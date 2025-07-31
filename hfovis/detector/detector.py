import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PyQt6.QtCore import QThread, pyqtSignal
from scipy.signal import butter, firwin, lfilter, sosfilt, sosfilt_zi
from streamz import Stream

from hfovis.data.buffering import RingBuffer
from hfovis.data.streaming import Streamer
from hfovis.detector.config import DetectorConfig


class RealTimeDetector(QThread):
    """
    Subclass of `QThread` Real-time detector for high-frequency oscillations (HFOs) in
    EEG data.

    Parameters
    ----------
    stream : Streamer
        A data stream object that has a `start()` method for starting the stream
        and a `read()` method for reading a chunk from the stream.

    parent : QObject, optional

    **kwargs
        Check `DetectorConfig` for available parameters.

    Attributes
    ----------
    new_event : pyqtSignal
        Signal emitted when a new event is detected.
    stream : Streamer
    config : DetectorConfig
    ring_buffer : RingBuffer
        Used to store raw data for when events are detected.

    Methods
    -------
    stop()
        Stops both the stream and the detector.
    run()
        Reads through the data stream and emits chunks tagged with a global index.
    build_graph_single_band()
        Builds the processing graph for single-band HFO detection.
    build_graph_dual_band()
        Builds the processing graph for dual-band HFO detection. Note: Still WIP.
    """

    new_event = pyqtSignal(dict)

    def __init__(self, stream: Streamer, parent=None, **kwargs):
        super().__init__(parent)
        self.stream = stream
        self.config = DetectorConfig()
        self.config.update(**kwargs)

        self.build_graph_single_band()
        self._running = True

    def stop(self):
        """
        Stops both the stream and the detector.
        """
        self._running = False
        self.stream.stop()
        self.wait()

    def run(self):
        """
        Reads through the data stream and emits chunks tagged with a global index.
        """
        self.stream.start()
        try:
            idx = 0  # Global index to tag chunks with global time stamp
            while self._running:
                chunk = self.stream.read()
                if chunk is None:  # Last chunk in stream should be None
                    self.stop()
                self.raw_stream.emit((idx, chunk))
                idx += len(chunk)  # Update global index
        except Exception as e:
            print(f"Error in internal loop: {e}")
            self.stop()

    def build_graph_single_band(self):
        self.raw_stream = Stream()

        # Ring buffer is used to temporarily store raw data so that it can later
        # be matched in time if events are detected.
        self.ring_buffer = RingBuffer(
            int(self.config.ring_buffer_size_s * self.config.fs),
            self.config.channels,
        )

        # Send raw data to the ring buffer
        self.raw_stream.sink(lambda pair: self.ring_buffer.write(pair[1]))

        # Function to change global indexing of chunks to global indexing of individual
        # samples. This will allow for precise matching of filtered data with raw data
        # in the ring buffer regardless of frame shifts.
        def explode(pair):
            start, chunk = pair
            idxs = np.arange(start, start + len(chunk), dtype=np.int64)
            return list(zip(idxs, chunk))

        # Create filters once
        dc_offset_sos = butter(
            2, self.config.low_band, fs=self.config.fs, btype="high", output="sos"
        )
        hfo_band_b = firwin(
            65,
            self.config.hfo_band,
            fs=self.config.fs,
            pass_zero="bandpass",
            window="hamming",
        )

        # Initialize filter states
        dc_zi_init = np.repeat(
            sosfilt_zi(dc_offset_sos)[:, :, np.newaxis], self.config.channels, axis=2
        )
        fir_zi_init = np.zeros((64, self.config.channels))

        # Define the processing functions
        def dc_block(pair, state=dc_zi_init):
            idx, chunk = pair
            # The state is maintained within this block and not reset every time the
            # function is called.
            y, state[:] = sosfilt(dc_offset_sos, chunk, axis=0, zi=state)
            return idx, y

        def hfo_band_filter_block(pair, state=fir_zi_init):
            idx, chunk = pair
            y, state[:] = lfilter(hfo_band_b, 1.0, chunk, axis=0, zi=state)
            return idx, y

        filtered = (
            self.raw_stream.map(dc_block)
            .map(hfo_band_filter_block)
            .map(explode)
            .flatten()  # Flatten the stream to emit individual samples
        )

        # Adaptive thresholding stream
        std_devs = (
            filtered.sliding_window(
                int(
                    self.config.adaptive_threshold_window_size_ms
                    / 1000
                    * self.config.fs  # Convert ms to samples
                ),
                return_partial=False,  # Prevents sending buffers that aren't full yet
            )
            .slice(
                step=int(
                    (
                        self.config.adaptive_threshold_window_size_ms
                        - self.config.adaptive_threshold_overlap_ms
                    )
                    / 1000
                    # Convert from overlap ms to step samples
                    * self.config.fs
                )
            )
            # Remove the global index since it is not needed for the threshold and
            # compute standard deviation over only the signal
            .map(lambda w: np.std([x[1] for x in w], axis=0))  #
        )
        thresholds = (
            std_devs.sliding_window(
                # In this case, it might take a while to get enough standard deviations,
                # so we will allow partial windows.
                self.config.adaptive_threshold_num_windows,
                return_partial=True,
            )
            .slice(
                step=self.config.adaptive_threshold_num_windows
                - self.config.adaptive_threshold_num_windows_overlap,
            )
            .map(lambda w: self.config.threshold_multiplier * np.median(w, axis=0))
        )

        # Detection stream
        def classify(pair):
            # Here we are given a threshold for each channel as well as the filtered
            # window.
            thr, data = pair

            # Replace thresholds that are too low with the minimum
            thr[thr < self.config.min_threshold] = self.config.min_threshold

            win_idx = data[0][0]
            win = np.stack([x[1] for x in data])

            # Split of left and right thirds of the window
            third_of_window = int(win.shape[0] / 3)
            left_window = win[:third_of_window]
            right_window = win[-third_of_window:]

            # Get the crossing matrices for left, right, and entire window. First for
            # the positive thresholds.
            left_p_crossings = self._threshold_crossings(left_window, thr)
            left_n_crossings = self._threshold_crossings(-left_window, thr)
            right_p_crossings = self._threshold_crossings(right_window, thr)
            right_n_crossings = self._threshold_crossings(-right_window, thr)
            all_p_crossings = self._threshold_crossings(win, thr)
            all_n_crossings = self._threshold_crossings(-win, thr)

            # Our left and right conditions are that there should not be more than the
            # minimum number of crossings in the left and right thirds of the window.
            max_crossings = self.config.side_max_crossings

            left_p_burst_condition = left_p_crossings.sum(axis=0) < max_crossings
            left_n_burst_condition = left_n_crossings.sum(axis=0) < max_crossings
            left_burst_condition = left_p_burst_condition & left_n_burst_condition

            right_p_burst_condition = right_p_crossings.sum(axis=0) < max_crossings
            right_n_burst_condition = right_n_crossings.sum(axis=0) < max_crossings
            right_burst_condition = right_p_burst_condition & right_n_burst_condition

            # Now we need to filter out channels where the crossings are not close
            # enough to be in our HFO band.
            min_sample_distance = int(round(self.config.fs / self.config.hfo_band[0]))
            all_p_burst_channels = self._sufficient_high_frequency_crossings(
                all_p_crossings,
                min_sample_distance,
                self.config.center_min_crossings,
            )
            all_n_burst_channels = self._sufficient_high_frequency_crossings(
                all_n_crossings,
                min_sample_distance,
                self.config.center_min_crossings,
            )
            all_burst_condition = all_p_burst_channels | all_n_burst_channels

            # This is a mask for the channels that meet the burst criteria to be HFO
            # candidates.
            burst_channels = (
                left_burst_condition & right_burst_condition & all_burst_condition
            )
            if not burst_channels.any():
                return None

            # Next, we use the mask to extract the relevant data
            channel_indices = np.where(burst_channels)[0]
            filtered_seg = win[:, burst_channels]
            raw_seg = self.ring_buffer.read(win_idx, len(win))[:, burst_channels]

            # Now we extract the indices of the center of the event.
            visualization_window_size = int(
                self.config.visualization_window_size_ms / 1000 * self.config.fs
            )
            peak_idx, center_indices = self._center_extraction_indices(
                filtered_seg,
                visualization_window_size,
            )

            # Now we need to convert the centers of the events to seconds.
            center = (peak_idx + win_idx) / self.config.fs

            return {
                "raw": np.take_along_axis(raw_seg, center_indices, axis=0),
                "filtered": np.take_along_axis(filtered_seg, center_indices, axis=0),
                "center": center,
                "channels": channel_indices,
                "threshold": thr[burst_channels],
            }

        # We use the same method as before to get overlapping windows for burst
        # detection.
        burst_win = filtered.sliding_window(
            int(self.config.burst_window_size_ms / 1000 * self.config.fs),
            return_partial=False,
        ).slice(
            step=int(
                (self.config.burst_window_size_ms - self.config.burst_window_overlap_ms)
                / 1000
                * self.config.fs
            )
        )
        burst_events = (
            # Combining in this way will use the previous threshold since computing
            # those will take longer.
            thresholds.combine_latest(burst_win, emit_on=burst_win)
            .map(classify)
            # The classify function returns None if no channels meet the criteria, so we
            # can filter those out.
            .filter(lambda x: x is not None)
        )
        # Finally we send the detected events to the GUI thread
        burst_events.sink(self.new_event.emit)

    def _threshold_crossings(self, sig: np.ndarray, thr: np.ndarray) -> np.ndarray:
        """
        Produce binary array of threshold crossings.

        Parameters:
        -----------
        sig: np.ndarray
            The signal data for which to count threshold crossings. This should have
            shape (n_samples, n_channels).
        thr: np.ndarray
            The threshold values for each channel.

        Returns:
        --------
        np.ndarray
            An array of shape (n_samples-1, n_channels) with 1's where there are
            threshold crossings and 0's otherwise.
        """
        above = sig > thr
        # We use np.diff to find the transitions from below to above the threshold
        return np.abs(np.diff(above.astype(int), axis=0))

    def _sufficient_high_frequency_crossings(
        self,
        crossings: np.ndarray,
        min_sample_distance: int,
        min_cluster_crossings: int,
    ) -> np.ndarray:
        # Minimum number of samples needed to detect the number of cluster crossings
        # with the minimum sample distance spacing.
        window_size = (min_cluster_crossings - 1) * min_sample_distance + 1

        # Create sliding windows so we can check the number of crossings in each
        # cluster.
        sw = sliding_window_view(crossings, window_size, axis=0)
        counts = sw.sum(axis=2)

        return np.any(counts >= min_cluster_crossings, axis=0)

    def _center_extraction_indices(
        self, win: np.ndarray, vis_win_length: int
    ) -> np.ndarray:
        half = vis_win_length // 2

        # Find index of absolute maximum. This will be the center of the event.
        peak_idx = np.argmax(np.abs(win), axis=0)

        # Calculate the start index for the visualization window. If the peak is too
        # close to the edge, we simply return the edge and compromise on having the peak
        # be in the center.
        start_idx = np.clip(peak_idx - half, 0, win.shape[0] - vis_win_length)

        # Individually index each sample in the new window
        offsets = np.arange(vis_win_length)[:, None]

        # Add the offsets to the start indices to get the indices of the windows for
        # each sample in our visualization window.
        return peak_idx, start_idx[None, :] + offsets

    def build_graph_dual_band(self):
        # Send raw data to the ring buffer
        self.raw_stream.sink(lambda pair: self.ring_buffer.write(pair[1]))

        # Create filters once
        dc_offset_sos = butter(
            2, self.config.low_band, fs=self.config.fs, btype="high", output="sos"
        )
        r_b = firwin(
            65,
            self.config.ripple_band,
            fs=self.config.fs,
            pass_zero="bandpass",
            window="hamming",
        )
        fr_b = firwin(
            65,
            self.config.fast_ripple_band,
            fs=self.config.fs,
            pass_zero="bandpass",
            window="hamming",
        )

        # Initialize filter states
        dc_zi_init = np.tile(sosfilt_zi(dc_offset_sos), (self.config.channels, 1)).T
        fir_zi_init = np.zeros((64, self.config.channels))

        # Define the processing functions
        def dc_block(pair, state=dc_zi_init):
            idx, chunk = pair
            y, state[:] = sosfilt(dc_offset_sos, chunk, axis=0, zi=state)
            return idx, y

        def ripple_filter(pair, state=fir_zi_init.copy()):
            idx, chunk = pair
            y, state[:] = lfilter(r_b, 1.0, chunk, axis=0, zi=state)
            return idx, y

        def fast_ripple_filter(pair, state=fir_zi_init.copy()):
            idx, chunk = pair
            y, state[:] = lfilter(fr_b, 1.0, chunk, axis=0, zi=state)
            return idx, y
