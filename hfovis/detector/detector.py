import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from scipy.signal import butter, filtfilt, firwin, sosfilt, sosfilt_zi, lfilter
import threading
from streamz import Stream
from .utils import get_adaptive_threshold, find_burst_events
from ..data.buffering import RingBuffer
from typing import Callable, Dict, Any
from tqdm import tqdm

get_adaptive_threshold = get_adaptive_threshold
find_burst_events = find_burst_events


class RealTimeDetector:
    """
    Runs detection for HFOs on real-time stream of data. Runs detection on a separate
    thread so that the main thread can continue to read data. All events will then be
    sent to a user set handler.
    """

    def __init__(self, stream: Any, handle: Callable, **kwargs):
        """
        Parameters:
        -----------
        stream
            A data stream object that has a `start()` method for starting the stream
            and a `read()` method for reading a chunk from the stream.
        handle: function
            A function that will be called with the detected events. The function should
            accept a single argument which is a dictionary containing the detected
            events. The dictionary will have the following keys:
                - 'raw': The raw data segment of the detected event.
                - 'filtered': The filtered data segment of the detected event.
                - 'center': The center time of the detected event in seconds.
                - 'channels': The indices of the channels where the event was
                  detected.
                - 'threshold': The threshold used for detection.
        """
        self.stream = stream
        self.handle = handle
        self.config = self._default_config()
        self.config.update(kwargs)
        self._validate_config()
        self.raw_stream = Stream()

        # Ring buffer is used to temporarily store raw data so that it can later
        # be matched in time if events are detected.
        self.ring_buffer = RingBuffer(
            int(self.config["ring_buffer_size_s"] * self.config["fs"]),
            self.config["channels"],
        )
        self._build_graph_single_band()
        self._running = False

    def _default_config(self) -> Dict[str, Any]:
        """
        Returns the default configuration for the detector.

        Explanation of configuration parameters:
            fs: float
                Sampling frequency of the data in Hz.
            channels: int
                Number of channels in the data.
            hfo_band: list of float
                Frequency band for HFO detection (in Hz).
            ripple_band: list of float
                Frequency band of ripple oscillations (in Hz).
            fast_ripple_band: list of float
                Frequency band of fast ripple oscillations (in Hz).
            adaptive_threshold_window_size_ms: float
                Window size over which standard deviations should be calculated for
                adaptive thresholding.
            adaptive_threshold_overlap_ms: float
                Overlap between consecutive standard deviation windows for adaptive
                thresholding.
            adaptive_threshold_num_windows: int
                Number of standard deviations to calculate median over to get adaptive
                threshold.
            adaptive_threshold_overlap_ms: int
                Number of standard deviations to overlap when calculating adaptive
                threshold.
            min_threshold: float
                Minimum threshold for detection. If the adaptive threshold is below this
                value, threshold will be set to this value.
            threshold_multiplier: float
                Multiplier for the adaptive threshold to set the detection threshold.
            burst_window_size_ms: float
                Size of the window over which to detect bursts in milliseconds.
            burst_window_overlap_ms: float
                Overlap between consecutive burst windows in milliseconds.
            side_max_crossings: int
                Maximum number of threshold crossings allowed on the sides (overlap
                region) of the burst window.
            center_min_crossings: int
                Minimum number of threshold crossings required in the center of the
                burst window to classify window as HFO candidate.
            visualization_window_size_ms: float
                Size of the visualization window in milliseconds. This is used to
                extract the center of the detected event for visualization.
            low_band: float
                Low band frequency for high-pass filtering to remove DC offset.
            ring_buffer_size_s: float
                Size of the ring buffer in seconds. This is used to store raw data so
                that it can be matched with detected events.
        """
        return {
            "fs": 2048.0,
            "channels": 1,
            "hfo_band": [80, 500],
            "ripple_band": [80, 270],
            "fast_ripple_band": [230, 600],
            "adaptive_threshold_window_size_ms": 500.0,
            "adaptive_threshold_overlap_ms": 200.0,
            "adaptive_threshold_num_windows": 100,
            "adaptive_threshold_num_windows_overlap": 50,
            "min_threshold": 5.0,
            "threshold_multiplier": 3.0,
            "burst_window_size_ms": 320.0,
            "burst_window_overlap_ms": 64.0,
            "side_max_crossings": 4,
            "center_min_crossings": 6,
            "visualization_window_size_ms": 200.0,
            "low_band": 1,
            "ring_buffer_size_s": 10.0,
        }

    def _validate_config(self):
        pass

    def start(self):
        """
        Starts both the stream and the detector. Do not start the stream prior to
        running this.
        """
        self._thread = threading.Thread(target=self._internal_loop, daemon=True)
        self._running = True
        self.stream.start()
        self._thread.start()

    def stop(self):
        """
        Stops both the stream and the detector.
        """
        if self._running:
            self._running = False
            self.stream.stop()
            self._thread.join()

    def _internal_loop(self):
        """
        Reads through the data stream and emits chunks tagged with a global index.
        """
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

    def _build_graph_single_band(self):
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
            2, self.config["low_band"], fs=self.config["fs"], btype="high", output="sos"
        )
        hfo_band_b = firwin(
            65,
            self.config["hfo_band"],
            fs=self.config["fs"],
            pass_zero="bandpass",
            window="hamming",
        )

        # Initialize filter states
        dc_zi_init = np.repeat(
            sosfilt_zi(dc_offset_sos)[:, :, np.newaxis], self.config["channels"], axis=2
        )
        fir_zi_init = np.zeros((64, self.config["channels"]))

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
                    self.config["adaptive_threshold_window_size_ms"]
                    / 1000
                    * self.config["fs"]  # Convert ms to samples
                ),
                return_partial=False,  # Prevents sending buffers that aren't full yet
            )
            .slice(
                step=int(
                    (
                        self.config["adaptive_threshold_window_size_ms"]
                        - self.config["adaptive_threshold_overlap_ms"]
                    )
                    / 1000
                    * self.config["fs"]  # Convert from overlap ms to step samples
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
                self.config["adaptive_threshold_num_windows"],
                return_partial=True,
            )
            .slice(
                step=self.config["adaptive_threshold_num_windows"]
                - self.config["adaptive_threshold_num_windows_overlap"],
            )
            .map(lambda w: self.config["threshold_multiplier"] * np.median(w, axis=0))
        )

        # Detection stream
        def classify(pair):
            # Here we are given a threshold for each channel as well as the filtered
            # window.
            thr, data = pair

            # Replace thresholds that are too low with the minimum
            thr[thr < self.config["min_threshold"]] = self.config["min_threshold"]

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
            max_crossings = self.config["side_max_crossings"]

            left_p_burst_condition = left_p_crossings.sum(axis=0) < max_crossings
            left_n_burst_condition = left_n_crossings.sum(axis=0) < max_crossings
            left_burst_condition = left_p_burst_condition & left_n_burst_condition

            right_p_burst_condition = right_p_crossings.sum(axis=0) < max_crossings
            right_n_burst_condition = right_n_crossings.sum(axis=0) < max_crossings
            right_burst_condition = right_p_burst_condition & right_n_burst_condition

            # Now we need to filter out channels where the crossings are not close
            # enough to be in our HFO band.
            min_sample_distance = int(
                round(self.config["fs"] / self.config["hfo_band"][0])
            )
            all_p_burst_channels = self._sufficient_high_frequency_crossings(
                all_p_crossings,
                min_sample_distance,
                self.config["center_min_crossings"],
            )
            all_n_burst_channels = self._sufficient_high_frequency_crossings(
                all_n_crossings,
                min_sample_distance,
                self.config["center_min_crossings"],
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
                self.config["visualization_window_size_ms"] / 1000 * self.config["fs"]
            )
            peak_idx, center_indices = self._center_extraction_indices(
                filtered_seg,
                visualization_window_size,
            )

            # Now we need to convert the centers of the events to seconds.
            center = (peak_idx + win_idx) / self.config["fs"]

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
            int(self.config["burst_window_size_ms"] / 1000 * self.config["fs"]),
            return_partial=False,
        ).slice(
            step=int(
                (
                    self.config["burst_window_size_ms"]
                    - self.config["burst_window_overlap_ms"]
                )
                / 1000
                * self.config["fs"]
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
        # Finally we send the detected events to the user defined handler.
        burst_events.sink(self.handle)

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

    def _build_graph_dual_band(self):
        # Send raw data to the ring buffer
        self.raw_stream.sink(lambda pair: self.ring_buffer.write(pair[1]))

        # Create filters once
        dc_offset_sos = butter(
            2, self.config["low_band"], fs=self.config["fs"], btype="high", output="sos"
        )
        r_b = firwin(
            65,
            self.config["ripple_band"],
            fs=self.config["fs"],
            pass_zero="bandpass",
            window="hamming",
        )
        fr_b = firwin(
            65,
            self.config["fast_ripple_band"],
            fs=self.config["fs"],
            pass_zero="bandpass",
            window="hamming",
        )

        # Initialize filter states
        dc_zi_init = np.tile(sosfilt_zi(dc_offset_sos), (self.config["channels"], 1)).T
        fir_zi_init = np.zeros((64, self.config["channels"]))

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


class AmplitudeThresholdDetectorV2:
    def __init__(self, data, **kwargs):
        self.data = data
        self.config = self.default_config()
        self.config.update(kwargs)
        self.validate_config()
        self.pool = {}

    def default_config(self):
        return {
            "montage": list(range(1, self.data.shape[1] + 1)),  # Default montage
            "hardWare": "Not Defined",
            "signalRange": "uV",  # Options: 'uV', 'mV'
            "fs": 2048,  # Default sampling frequency
            "resampleRate": 2048,  # Default resampling rate
            "LowBand": 1,  # Low band filter to remove DC offset
            "RippleBand": [80, 270],  # Ripple band
            "FastRippleBand": [230, 600],  # Fast Ripple band
            "HFOBand": [80, 500],  # HFO band
            "threshold_type": "Single",  # Options: 'Single', 'Dual'
            "thresholdMultiplier_HFO": [
                3,
                100,
            ],  # For 'Dual', it needs three values i.e [3, 4, 9]
            "thresholdMultiplier_R": [
                3,
                100,
            ],  # For 'Dual', it needs three values i.e [3, 4, 9]
            "thresholdMultiplier_FR": [
                3,
                100,
            ],  # For 'Dual', it needs three values i.e [3, 4, 9]
            "detection_type": "Single Band",  # Options: 'Single Band', 'Dual Band'
            "burstWindow_type": "fixed",  # Options: 'fixed', 'flexible'
            "burstWindow": 250,  # For 'fixed' window (in ms)
            "burstWindow_center": "max",  # Options: 'max', 'best'
            "minBurst_duration": 10,  # In ms
            "maxBurst_duration": 150,  # In ms
            "stitchTime": 50,  # In ms
            "adaptiveThreshold_type": "Std",  # Options: 'Std', 'max', 'mean', 'median'
            "adaptiveThreshold_WindowSize": 500,  # In ms
            "adaptiveThreshold_WindowOverlap": 200,  # In ms
            "adaptiveThreshold_WindowNo": 1800,  # Number of windows
            "adaptiveThreshold_WindowNoShift": 1000,  # Number of windows
            "symmetricGlobalSwing_type": "Full",  # Options: 'Full', 'No'
            "symmetricLocalSwing_type": "Half-Symmetric",  # Options: 'Full-Symmetric', 'Half-Symmetric', 'No-Symmetric'
            "removeSide": 4,  # Options: 0, k
            "min_symmetricGlobalSwing": 2,  # Minimum number of global swings
            "min_symmetricLocalSwing": 3,  # Minimum number of local swings
            "includeEvent": 0,  # Options: 1, 0
            "min_threshold_R": 5,  # Minimum threshold in Ripple Band
            "min_threshold_HFO": 5,  # Minimum threshold in HFO Band
            "min_threshold_FR": 5,  # Minimum threshold in Fast Ripple Band
            "max_threshold_FR": 100,  # Maximum threshold in FR Band
            "checkCentralized": 1,  # Options: 0, 1
            "backGroundMultiplier": 0,
            "noiseFloor_HFO_Band": 0,
            "noiseFloor_R_Band": 0,
            "noiseFloor_FR_Band": 0,
            "PSDCompute": 0,  # Options: 0, 1
        }

    def validate_config(self):
        valid_values = {
            "hardWare": [
                "Natus Quantum",
                "Nihon Kohden",
                "gtec",
                "BIC",
                "CadWell",
                "Ripple",
                "Not Defined",
            ],
            "signalRange": ["uV", "mV"],
            "threshold_type": ["Single", "Dual"],
            "detection_type": ["Single Band", "Dual Band"],
            "burstWindow_type": ["fixed", "flexible"],
            "burstWindow_center": ["max", "best"],
            "adaptiveThreshold_type": ["Std", "max", "mean", "median"],
            "symmetricGlobalSwing_type": ["Full", "No"],
            "symmetricLocalSwing_type": [
                "Full-Symmetric",
                "Half-Symmetric",
                "No-Symmetric",
            ],
        }

        config = self.config
        assert config["hardWare"] in valid_values["hardWare"], "Invalid hardware option"
        assert config["signalRange"] in valid_values["signalRange"], (
            "Invalid signal range option"
        )
        assert isinstance(config["fs"], (int, float)) and config["fs"] > 0, (
            "Sampling frequency must be a positive number"
        )
        assert (
            isinstance(config["resampleRate"], (int, float))
            and config["resampleRate"] >= 0
        ), "Resample rate must be a non-negative number"
        assert isinstance(config["LowBand"], (int, float)) and config["LowBand"] > 0, (
            "LowBand must be a positive number"
        )
        assert len(config["RippleBand"]) == 2, "RippleBand must be a 2-element list"
        assert len(config["FastRippleBand"]) == 2, (
            "FastRippleBand must be a 2-element list"
        )
        assert len(config["HFOBand"]) == 2, "HFOBand must be a 2-element list"
        assert config["threshold_type"] in valid_values["threshold_type"], (
            "Invalid threshold type option"
        )
        assert config["detection_type"] in valid_values["detection_type"], (
            "Invalid detection type option"
        )
        assert config["burstWindow_type"] in valid_values["burstWindow_type"], (
            "Invalid burst window type option"
        )
        assert isinstance(config["burstWindow"], int) and config["burstWindow"] > 0, (
            "burstWindow must be a positive number"
        )
        assert config["burstWindow_center"] in valid_values["burstWindow_center"], (
            "Invalid burst window center option"
        )
        assert (
            isinstance(config["minBurst_duration"], int)
            and config["minBurst_duration"] > 0
        ), "minBurst_duration must be positive"
        assert (
            isinstance(config["maxBurst_duration"], int)
            and config["maxBurst_duration"] > 0
        ), "maxBurst_duration must be positive"
        assert isinstance(config["stitchTime"], int) and config["stitchTime"] > 0, (
            "stitchTime must be positive"
        )
        assert (
            config["adaptiveThreshold_type"] in valid_values["adaptiveThreshold_type"]
        ), "Invalid adaptive threshold type"
        assert (
            isinstance(config["adaptiveThreshold_WindowSize"], int)
            and config["adaptiveThreshold_WindowSize"] > 0
        ), "adaptiveThreshold_WindowSize must be positive"
        assert (
            isinstance(config["adaptiveThreshold_WindowOverlap"], int)
            and config["adaptiveThreshold_WindowOverlap"] >= 0
        ), "adaptiveThreshold_WindowOverlap must be non-negative"
        assert (
            isinstance(config["adaptiveThreshold_WindowNo"], int)
            and config["adaptiveThreshold_WindowNo"] > 0
        ), "adaptiveThreshold_WindowNo must be positive"
        assert (
            isinstance(config["adaptiveThreshold_WindowNoShift"], int)
            and config["adaptiveThreshold_WindowNoShift"] >= 0
        ), "adaptiveThreshold_WindowNoShift must be non-negative"
        assert (
            config["symmetricGlobalSwing_type"]
            in valid_values["symmetricGlobalSwing_type"]
        ), "Invalid symmetric global swing type"
        assert (
            config["symmetricLocalSwing_type"]
            in valid_values["symmetricLocalSwing_type"]
        ), "Invalid symmetric local swing type"
        assert isinstance(config["removeSide"], int) and config["removeSide"] >= 0, (
            "removeSide must be non-negative"
        )
        assert (
            isinstance(config["min_symmetricGlobalSwing"], int)
            and config["min_symmetricGlobalSwing"] >= 0
        ), "min_symmetricGlobalSwing must be positive"
        assert (
            isinstance(config["min_symmetricLocalSwing"], int)
            and config["min_symmetricLocalSwing"] >= 0
        ), "min_symmetricLocalSwing must be positive"
        assert config["includeEvent"] in [0, 1], "includeEvent must be 0 or 1"
        assert (
            isinstance(config["min_threshold_R"], int) and config["min_threshold_R"] > 0
        ), "min_threshold_R must be positive"
        assert (
            isinstance(config["min_threshold_HFO"], int)
            and config["min_threshold_HFO"] > 0
        ), "min_threshold_HFO must be positive"
        assert (
            isinstance(config["min_threshold_FR"], int)
            and config["min_threshold_FR"] > 0
        ), "min_threshold_FR must be positive"
        assert (
            isinstance(config["max_threshold_FR"], int)
            and config["max_threshold_FR"] > 0
        ), "max_threshold_FR must be positive"
        assert config["checkCentralized"] in [0, 1], "checkCentralized must be 0 or 1"
        assert config["PSDCompute"] in [0, 1], "PSDCompute must be 0 or 1"

        # Recalculate config parameters based on resampleRate
        fs = self.config["resampleRate"]
        fs_reference = 2048

        self.config["burstWindow"] = round(self.config["burstWindow"] / 1000 * fs)
        self.config["minBurst_duration"] = round(
            self.config["minBurst_duration"] / 1000 * fs
        )
        self.config["maxBurst_duration"] = round(
            self.config["maxBurst_duration"] / 1000 * fs
        )
        self.config["stitchTime"] = round(self.config["stitchTime"] / 1000 * fs)

        self.config["adaptiveThreshold_WindowSize"] = round(
            self.config["adaptiveThreshold_WindowSize"] * fs / fs_reference
        )
        self.config["adaptiveThreshold_WindowOverlap"] = round(
            self.config["adaptiveThreshold_WindowOverlap"] * fs / fs_reference
        )
        self.config["adaptiveThreshold_WindowNo"] = round(
            self.config["adaptiveThreshold_WindowNo"] * fs / fs_reference
        )
        self.config["adaptiveThreshold_WindowNoShift"] = round(
            self.config["adaptiveThreshold_WindowNoShift"] * fs / fs_reference
        )

    def detect(self):
        if self.config["resampleRate"] != 0:
            self.resample_data()
        self.filter_data()
        if self.config["detection_type"] == "Single Band":
            return self.single_band_detection()
        elif self.config["detection_type"] == "Dual Band":
            return self.dual_band_detection()

    def resample_data(self):
        raw = np.copy(self.data)  # raw data

        if int(self.config["fs"] / 100) != int(
            self.config["resampleRate"] / 100
        ):  # 100 Hz resolution for resampling
            print(
                f"Resampling data from {self.config['fs']:.0f} Hz to {self.config['resampleRate']:.0f} Hz"
            )

            # If downsampling the data, first filter the data
            if round(self.config["fs"]) > round(self.config["resampleRate"]):
                b_tmp = firwin(
                    64,
                    (self.config["resampleRate"] / 2) / (self.config["fs"] / 2),
                    pass_zero="lowpass",
                )
                raw = filtfilt(b_tmp, 1, raw, axis=0)

            dp = raw.shape[0]
            ts_old = np.arange(1, dp + 1) / self.config["fs"]
            ts_new = np.arange(
                1 / self.config["resampleRate"],
                ts_old[-1] + 1 / self.config["resampleRate"],
                1 / self.config["resampleRate"],
            )

            data_tmp = np.zeros((len(ts_new), raw.shape[1]))  # Predefine temporary data

            for ch in range(raw.shape[1]):
                data_tmp[:, ch] = np.interp(ts_new, ts_old, raw[:, ch])

            self.data = data_tmp
            self.config["fs"] = round(self.config["resampleRate"])

    def filter_data(self):
        raw = self.data
        print("iEEG Filtering...")
        # Remove DC offset (High-Pass Filter)
        bL, aL = butter(
            2, self.config["LowBand"] / (self.config["fs"] / 2), btype="high"
        )
        raw = filtfilt(
            bL, aL, raw, axis=0, method="pad", padlen=3 * (max(len(bL), len(aL)) - 1)
        )

        # Filtering the analyzed bands based on detection type
        if self.config["detection_type"] == "Single Band":
            print("Single Band Filtering...")
            b_hfo = firwin(
                65,
                np.array(self.config["HFOBand"]) / (self.config["fs"] / 2),
                pass_zero="bandpass",
                window="hamming",
            )
            self.data = {
                "raw": raw,
                "hfo_band": filtfilt(
                    b_hfo,
                    1,
                    raw,
                    axis=0,
                    method="pad",
                    padlen=3 * (max(len(b_hfo), 1) - 1),
                ),
            }
        elif self.config["detection_type"] == "Dual Band":
            print("Dual Band Filtering...")
            b_r = firwin(
                65,
                np.array(self.config["RippleBand"]) / (self.config["fs"] / 2),
                pass_zero="bandpass",
                window="hamming",
            )
            b_fr = firwin(
                65,
                np.array(self.config["FastRippleBand"]) / (self.config["fs"] / 2),
                pass_zero="bandpass",
                window="hamming",
            )
            self.data = {
                "raw": raw,
                "ripple_band": filtfilt(
                    b_r, 1, raw, axis=0, method="pad", padlen=3 * (max(len(b_r), 1) - 1)
                ),
                "fripple_band": filtfilt(
                    b_fr,
                    1,
                    raw,
                    axis=0,
                    method="pad",
                    padlen=3 * (max(len(b_fr), 1) - 1),
                ),
            }
        else:
            self.data = {"raw": raw}

    def calculate_thresholds(self, data, config):
        backgroundNoise_list = []
        threshold = {}

        # Background Noise Calculation
        _, background_level = get_adaptive_threshold(
            data,
            config["adaptiveThreshold_WindowSize"],
            config["adaptiveThreshold_WindowOverlap"],
            config["adaptiveThreshold_WindowNo"],
            config["adaptiveThreshold_WindowNoShift"],
            config["adaptiveThreshold_type"],
            1,
        )

        backgroundNoise_list.append(background_level)
        threshold["backgroundLevel"] = background_level

        # Apply Dual or Single Thresholds
        if config["threshold_type"] == "Dual":
            threshold["th1"] = np.maximum(
                config["min_threshold"],
                config["thresholdMultiplier"][0] * background_level
                - config["backGroundMultiplier"] * config["noiseFloor_Band"],
            )
            threshold["th2"] = np.maximum(
                config["min_threshold"]
                * (config["thresholdMultiplier"][1] / config["thresholdMultiplier"][0]),
                config["thresholdMultiplier"][1] * background_level
                - config["backGroundMultiplier"] * config["noiseFloor_Band"],
            )
            threshold["th_rej"] = np.maximum(
                config["min_threshold"],
                config["thresholdMultiplier"][-1] * background_level
                - config["backGroundMultiplier"] * config["noiseFloor_Band"],
            )
        else:
            threshold["th1"] = np.maximum(
                config["min_threshold"],
                config["thresholdMultiplier"][0] * background_level
                - config["backGroundMultiplier"] * config["noiseFloor_Band"],
            )
            threshold["th_rej"] = np.maximum(
                config["min_threshold"],
                config["thresholdMultiplier"][-1] * background_level
                - config["backGroundMultiplier"] * config["noiseFloor_Band"],
            )

        return backgroundNoise_list, threshold

    def single_band_detection(self):
        raw = self.data["raw"]
        config = self.config
        print("Single-Band Event Detection")

        chN = raw.shape[1]  # Number of channels
        config["Band_of_Interest"] = "HFO"
        config["detectionBand"] = config["HFOBand"]
        config["min_threshold"] = config["min_threshold_HFO"]
        config["thresholdMultiplier"] = config["thresholdMultiplier_HFO"]
        config["noiseFloor_Band"] = config["noiseFloor_HFO_Band"]

        threshold_HFO = {}
        detected_events_master = []

        for channel in tqdm(range(chN), desc="Processing Channels"):
            # for channel in [2]:
            # Calculate thresholds
            backgroundNoise_HFO_list, threshold_HFO[channel] = (
                self.calculate_thresholds(self.data["hfo_band"][:, channel], config)
            )

            # Detect Events
            detected_events = find_burst_events(
                self.data["raw"][:, channel],
                self.data["hfo_band"][:, channel],
                threshold_HFO[channel],
                config,
            )
            if not detected_events.empty:
                detected_events["channel"] = channel
                detected_events_master.append(detected_events)

        if detected_events_master:
            df_master = pd.concat(detected_events_master, ignore_index=True)
            print(f"{df_master.shape[0]} event(s) detected in HFO band.")

            self.pool = {
                "event": np.stack(
                    (df_master["event_raw"], df_master["event_filtered"]), axis=1
                ),
                "threshold_event_HFOBand": df_master["threshold"],
                "backgroundNoise_HFO": backgroundNoise_HFO_list,
                "temporalDistribution": df_master["center"],
                "spatialDistribution": df_master["channel"],
                "burstLength": df_master["duration"],
                "number_of_channels": chN,
                "config": config,
                "channelMontage": config["montage"],
                "fs": config["fs"],
            }
        else:
            print("No HFO detected!")

            self.pool = {
                "event": np.array([]),
                "threshold_event_HFOBand": np.array([]),
                "backgroundNoise_HFO": backgroundNoise_HFO_list,
                "temporalDistribution": np.array([]),
                "spatialDistribution": np.array([]),
                "burstLength": np.array([]),
                "number_of_channels": chN,
                "config": config,
                "channelMontage": config["montage"],
                "fs": config["fs"],
            }

        # Save data into pool
        return self.data, self.pool

    def dual_band_detection(self):
        raw = self.data["raw"]
        config = self.config
        print("Dual Band (Ripple & Fast Ripple) Detection")

        chN = raw.shape[1]
        config["Band_of_Interest"] = "R"
        config["detectionBand"] = config["RippleBand"]
        config["min_threshold"] = config["min_threshold_R"]
        config["thresholdMultiplier"] = config["thresholdMultiplier_R"]
        config["noiseFloor_Band"] = config["noiseFloor_R_Band"]

        backgroundNoise_R_list = []
        threshold_R = {}
        detected_events_master = []

        for k in tqdm(range(chN), desc="Processing Channels"):
            # Calculate thresholds
            backgroundNoise_R_list, threshold_R[k] = self.calculate_thresholds(
                self.data["ripple_band"][:, k], config
            )

            # Detect Events
            detected_events = find_burst_events(
                self.data["raw"][:, k],
                self.data["ripple_band"][:, k],
                threshold_R[k],
                config,
            )

            if not detected_events.empty:
                detected_events["channel"] = k  # or channel+1 if you prefer 1-based
                detected_events_master.append(detected_events)

        # Combine results across all channels
        if detected_events_master:
            df_master_ripple = pd.concat(detected_events_master, ignore_index=True)
            print(f"{df_master_ripple.shape[0]} event(s) detected in the Ripple band.")
        else:
            print("No Ripples detected!")

        # **Step 2: Detect Fast Ripple Band Events**
        config["Band_of_Interest"] = "FR"  # Frequency bands
        config["detectionBand"] = config["FastRippleBand"]
        config["min_threshold"] = config["min_threshold_FR"]
        config["thresholdMultiplier"] = config["thresholdMultiplier_FR"]
        config["noiseFloor_Band"] = config["noiseFloor_FR_Band"]

        threshold_FR = {}
        backgroundNoise_FR_list = []
        detected_events_master = []

        for k in tqdm(range(chN), desc="Processing Channels"):
            # Calculate thresholds
            backgroundNoise_FR_list, threshold_FR[k] = self.calculate_thresholds(
                self.data["fripple_band"][:, k], config
            )

            # Detect Events
            detected_events = find_burst_events(
                self.data["raw"][:, k],
                self.data["fripple_band"][:, k],
                threshold_FR[k],
                config,
            )

            if not detected_events.empty:
                detected_events["channel"] = k  # or channel+1 if you prefer 1-based
                detected_events_master.append(detected_events)

        # Combine results across all channels
        if detected_events_master:
            df_master_fripple = pd.concat(detected_events_master, ignore_index=True)
            print(
                f"{df_master_fripple.shape[0]} event(s) detected in the Fast Ripple band."
            )

        else:
            print("No Fast Ripples detected!")

        print("Merging Ripple & Fast Ripple events...\n")

        # Ensure the dataframes exist; if empty, create empty dataframes with required columns
        if "df_master_ripple" not in locals() or df_master_ripple.empty:
            df_master_ripple = pd.DataFrame(
                columns=["center", "event_filtered", "threshold", "channel"]
            )
            print("No Ripples detected!")
        if "df_master_fripple" not in locals() or df_master_fripple.empty:
            df_master_fripple = pd.DataFrame(
                columns=["center", "event_filtered", "threshold", "channel"]
            )
            print("No Fast Ripples detected!")

        # --- CONFIGURATION ---
        burst_window = config["burstWindow"]  # Time threshold for merging

        # --- STORAGE ---
        timestamp_HFO = {ch: [] for ch in range(chN)}
        overlapping_HFO = {
            ch: [] for ch in range(chN)
        }  # Storage for overlapping events
        pairs = {}
        subGroup_R = []
        subGroup_FR = []

        # --- PROCESS EACH CHANNEL SEPARATELY ---
        for ch in range(chN):
            try:
                # Extract timestamps; if the dataframe is empty for the channel, use an empty array.
                t_R = (
                    df_master_ripple[df_master_ripple["channel"] == ch]["center"].values
                    if (
                        not df_master_ripple.empty
                        and ch in df_master_ripple["channel"].values
                    )
                    else np.array([])
                )
                t_FR = (
                    df_master_fripple[df_master_fripple["channel"] == ch][
                        "center"
                    ].values
                    if (
                        not df_master_fripple.empty
                        and ch in df_master_fripple["channel"].values
                    )
                    else np.array([])
                )
                TFR = []

                # Compute distance matrix if both arrays have data
                if t_R.size > 0 and t_FR.size > 0:
                    dist_matrix = np.abs(
                        t_R[:, None] - t_FR[None, :]
                    )  # Pairwise absolute differences
                    i, j = np.where(
                        dist_matrix < round(burst_window / 2 - 1)
                    )  # Find overlapping events
                    pairs[ch] = np.column_stack((i, j))  # Store index pairs
                    # Separate overlapping events
                    overlapping_HFO[ch] = np.column_stack(
                        (
                            t_R[i],
                            t_FR[j],
                        )
                    )  # Save overlapping event timestamps
                    # Remove overlapping events from Ripple group to keep only non-overlapping
                    t_R = np.delete(t_R, i, axis=0)
                    TFR = np.copy(t_FR)
                    TFR = np.delete(TFR, j, axis=0)

                # Merge timestamps for non-overlapping events
                timestamp_HFO[ch] = np.concatenate((t_R, t_FR))
                # Store the subgroups
                subGroup_R.extend(t_R)
                subGroup_FR.extend(TFR)

            except Exception as e:
                print(f"Skipping channel {ch} due to error: {e}")
                timestamp_HFO[ch] = []

        # --- CONCATENATE TIMESTAMPS ACROSS ALL CHANNELS ---
        non_empty_ts = [ts for ts in timestamp_HFO.values() if len(ts) > 0]
        if non_empty_ts:
            non_overlapping_timestamps = np.concatenate(non_empty_ts)
        else:
            non_overlapping_timestamps = np.array([])

        non_empty_arrays = [ov for ov in overlapping_HFO.values() if len(ov) > 0]
        if non_empty_arrays:
            overlapping_timestamps = np.concatenate(non_empty_arrays)
        else:
            overlapping_timestamps = np.array([])

        # --- SORT EVENTS BASED ON TIMESTAMP ---
        sorted_indices = (
            np.argsort(non_overlapping_timestamps)
            if non_overlapping_timestamps.size > 0
            else np.array([])
        )
        sorted_non_timestamps = (
            non_overlapping_timestamps[sorted_indices]
            if non_overlapping_timestamps.size > 0
            else np.array([])
        )
        sorted_overlap_indices = (
            np.argsort(overlapping_timestamps[:, 0])
            if overlapping_timestamps.size > 0
            else np.array([])
        )
        sorted_overlapping_timestamps = (
            overlapping_timestamps[sorted_overlap_indices]
            if overlapping_timestamps.size > 0
            else np.array([])
        )

        # --- STORE FINAL EVENTS ---
        event = {
            "HFO_timeStamp": sorted_non_timestamps,
            "Overlapping_HFO_timeStamp": sorted_overlapping_timestamps,  # NEW: Store overlapping timestamps
            "Ripple signal": {
                ch: df_master_ripple[df_master_ripple["channel"] == ch][
                    "event_filtered"
                ].values
                for ch in df_master_ripple["channel"].unique()
            },
            "FastRipple signal": {
                ch: df_master_fripple[df_master_fripple["channel"] == ch][
                    "event_filtered"
                ].values
                for ch in df_master_fripple["channel"].unique()
            },
        }

        # --- EXTRACT CHANNEL REFERENCES ---
        all_channels = (
            np.concatenate(
                [
                    np.full(len(ts), ch)
                    for ch, ts in timestamp_HFO.items()
                    if len(ts) > 0
                ]
            )
            if non_overlapping_timestamps.size > 0
            else np.array([])
        )
        sorted_channels = (
            all_channels[sorted_indices] if all_channels.size > 0 else np.array([])
        )
        event["HFO_channelInformation"] = sorted_channels

        # --- IDENTIFY CHANNELS FOR OVERLAPPING EVENTS ---
        non_empty_arrays = [
            np.full(len(ov), ch) for ch, ov in overlapping_HFO.items() if len(ov) > 0
        ]
        overlapping_channels = (
            np.concatenate(non_empty_arrays) if non_empty_arrays else np.array([])
        )
        sorted_overlapping_channels = (
            overlapping_channels[sorted_overlap_indices]
            if overlapping_timestamps.size > 0
            else np.array([])
        )
        event["Overlapping_HFO_channelInformation"] = (
            sorted_overlapping_channels  # NEW: Store channel references for overlapping events
        )

        # --- STORE DATA IN POOL ---
        self.pool = {
            "temporalDistribution": event["HFO_timeStamp"],
            "spatialDistribution": event["HFO_channelInformation"],
            "Overlapping_HFO_timeStamp": event["Overlapping_HFO_timeStamp"],
            "Overlapping_HFO_channelInformation": event[
                "Overlapping_HFO_channelInformation"
            ],
            "Ripple signal": event["Ripple signal"],
            "FastRipple signal": event["FastRipple signal"],
            "threshold_event_rippleBand": df_master_ripple["threshold"]
            if not df_master_ripple.empty
            else np.array([]),  # Ripple threshold
            "threshold_event_FRBand": df_master_fripple["threshold"]
            if not df_master_fripple.empty
            else np.array([]),  # Fast Ripple threshold
            "backgroundNoise_R": backgroundNoise_R_list,  # Noise levels for Ripples
            "backgroundNoise_FR": backgroundNoise_FR_list,  # Noise levels for Fast Ripples
            "overlappingTemporalDistribution": event["Overlapping_HFO_timeStamp"],
            "overlappingSpatialDistribution": event[
                "Overlapping_HFO_channelInformation"
            ],
            "number_of_channels": chN,
            "config": config,
            "channelMontage": config["montage"],
            "fs": config["fs"],
        }

        # --- PRINT SUMMARY ---
        print(
            f"{len(event['HFO_timeStamp'])} non-overlapping Ripple & FR events detected!"
        )
        print(
            f"{len(event['Overlapping_HFO_timeStamp'])} overlapping Ripple & FR events detected!"
        )
        print(f"{len(subGroup_R)} non-overlapping R events stored separately")
        print(f"{len(subGroup_FR)} non-overlapping FR events stored separately")

        return self.data, self.pool
