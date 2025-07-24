import pandas as pd
import numpy as np


class EventModel:
    def __init__(self, n_channels: int):
        self.raw_events = None
        self.filtered_events = None
        self.meta = None
        self._denoise_hist = np.zeros((2, n_channels), dtype=int)

    def append_batch(self, event: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        self._denoise_hist += hist_batch
        return self._denoise_hist
