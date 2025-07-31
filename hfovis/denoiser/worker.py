import os
import queue

import joblib
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from scipy.signal import butter

from .features import extract_omp_features

data_dir = os.path.join(os.path.dirname(__file__), "data")
classifier = joblib.load(os.path.join(data_dir, "random_forest_model.pkl"))


class DenoisingThread(QThread):
    """
    Subclass of `QThread`. Thread for denoising potential HFO candidates.

    Parameters
    ----------
    num_channels : int
        Number of channels in the EEG data.
    fs : float
        Sampling frequency of the EEG data.
    parent : QObject, optional

    Attributes
    ----------
    histReady : pyqtSignal
        Signal emitted when the histogram of classes is ready.
    classReady : pyqtSignal
        Signal emitted when the classification results are ready.
    num_channels : int
    bL, aL : np.ndarray
        Coefficients for the low-pass Butterworth filter.
    bH, aH : np.ndarray
        Coefficients for the band-pass Butterworth filter.

    Methods
    -------
    enqueue(raw_batch, batch_meta, batch_indices)
        Enqueues a batch of events to this denoising thread.
    run()
        Main loop of the thread that processes the queued events.
    stop()
        Stops the thread and waits for it to finish processing.
    """

    histReady = pyqtSignal(np.ndarray)
    classReady = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, num_channels: int, fs: float, parent=None):
        super().__init__(parent)
        self.num_channels = num_channels
        self.bL, self.aL = butter(2, 1, fs=fs, btype="high")
        self.bH, self.aH = butter(4, (80, 600), fs=fs, btype="bandpass")

        # replace single-slot with a queue
        self._queue = queue.Queue()
        self._running = True

    def enqueue(self, raw_batch, batch_meta, batch_indices):
        """
        Enqueues a batch of events to this denoising thread.

        Parameters
        ----------
        raw_batch : np.ndarray
        batch_meta : pd.DataFrame
        batch_indices : np.ndarray
            Indices in the Pandas DataFrame that correspond to the events
        """
        self._queue.put(
            (
                raw_batch.copy(),
                batch_meta.copy().reset_index(drop=True),
                batch_indices.copy(),
            )
        )

    def run(self):
        """
        Main loop of the thread that processes the queued events.
        """
        while self._running:
            try:
                raw, meta, indices = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            features = extract_omp_features(
                raw.T, bL=self.bL, aL=self.aL, bH=self.bH, aH=self.aH
            )
            classes = np.asarray(classifier.predict(features), dtype=int)

            batch_hist = np.zeros((2, self.num_channels), dtype=int)
            for cls, ch in zip(classes, meta["channel"].astype(int)):
                batch_hist[cls, ch] += 1

            self.classReady.emit(indices, classes)
            self.histReady.emit(batch_hist)

            self._queue.task_done()

    def stop(self):
        """
        Stops the thread and waits for it to finish processing.
        """
        self._running = False
        self.wait()
