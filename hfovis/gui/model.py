import numpy as np
import pandas as pd


class EventModel:
    """
    Model for handling event data, including raw and filtered events, metadata,
    and denoise histogram counts.

    Parameters
    ----------
    n_channels : int
        Number of channels in the event data.

    Attributes
    ----------
    raw_events : np.ndarray
        Array of raw event data, shape (n_samples, n_channels).
    filtered_events : np.ndarray
        Array of filtered event data, shape (n_samples, n_channels).
    meta : pd.DataFrame
        DataFrame containing metadata for each event, including center, channel,
        threshold, and whether the event is real.

    Methods
    -------
    append_batch(event: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Appends a batch of events to the model and returns the raw and filtered
        event arrays, metadata DataFrame, and batch indices.
    append_denoise_counts(hist_batch: np.ndarray) -> np.ndarray:
        Appends a batch of denoise histogram counts to the model and returns the
        updated histogram counts.
    save(raw_filename: str, filt_filename: str, meta_filename: str):
        Saves the raw events, filtered events, and metadata to specified files.
    """

    def __init__(self, n_channels: int):
        self.raw_events = None
        self.filtered_events = None
        self.meta = None
        self._denoise_hist = np.zeros((2, n_channels), dtype=int)

    def append_batch(self, event: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Appends a batch of events to the model.

        Parameters
        ----------
        event : dict
            Dictionary containing the following keys:
            - "raw": Raw event data as a 2D array (n_samples, n
            - "filtered": Filtered event data as a 2D array (n_samples, n_channels).
            - "center": Center of the event as a 1D array (n_samples,).
            - "channels": Channels associated with the event as a 1D array (n_samples,).
            - "threshold": Threshold values for the event as a 1D array (n_samples,).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]
            - raw_batch: Raw event data for the batch as a 2D array
            (n_samples, n_channels).
            - filt_batch: Filtered event data for the batch as a 2D array (n_samples,
            n_channels).
            - batch_meta: DataFrame containing metadata
            - batch_indices: Indices of the events in the batch.
            - chan_vec: Vector of channels associated with the events in the batch.
        """
        raw_batch = event["raw"].T
        filt_batch = event["filtered"].T

        if self.raw_events is None:
            self.raw_events = raw_batch.copy()
            self.filtered_events = filt_batch.copy()
        else:
            self.raw_events = np.vstack((self.raw_events, raw_batch))
            self.filtered_events = np.vstack((self.filtered_events, filt_batch))

        center_arr = np.asarray(event["center"], dtype=float)
        chan_vec = np.asarray(event["channels"], dtype=int)
        threh_arr = np.asarray(event["threshold"], dtype=float)

        old_n = len(self.meta) if self.meta is not None else 0
        batch_indices = np.arange(old_n, old_n + len(center_arr), dtype=int)

        batch_meta = pd.DataFrame(
            {
                "center": center_arr,
                "channel": chan_vec,
                "threshold": threh_arr,
                "is_real": pd.NA,
            }
        )
        self.meta = (
            pd.concat([self.meta, batch_meta], ignore_index=True)
            if self.meta is not None
            else batch_meta
        )
        return raw_batch, filt_batch, batch_meta, batch_indices, chan_vec

    def append_denoise_counts(self, hist_batch: np.ndarray) -> np.ndarray:
        """
        Appends a batch of denoise histogram counts to the model.

        Parameters
        ----------
        hist_batch : np.ndarray
            Histogram batch of shape (2, n_channels) containing counts for pseudo-HFO
            and real-HFO across channels.

        Returns
        -------
        np.ndarray
            Updated denoise histogram counts, shape (2, n_channels).
        """
        self._denoise_hist += hist_batch
        return self._denoise_hist

    def save(self, raw_filename: str, filt_filename: str, meta_filename: str):
        """
        Saves the raw events, filtered events, and metadata to specified files.

        Parameters
        ----------
        raw_filename : str
            Filename to save the raw events (should be .pkl).
        filt_filename : str
            Filename to save the filtered events (should be .npy).
        meta_filename : str
            Filename to save the metadata DataFrame (should be .npy).
        """
        if self.meta is not None and meta_filename:
            self.meta.to_pickle(meta_filename)
        if self.raw_events is not None and raw_filename:
            np.save(raw_filename, self.raw_events)
        if self.filtered_events is not None and filt_filename:
            np.save(filt_filename, self.filtered_events)
