import os
import joblib
import numpy as np
import queue
from scipy.signal import butter
from PyQt6.QtCore import QThread, pyqtSignal
from .features import extract_omp_features

data_dir = os.path.join(os.path.dirname(__file__), "data")
classifier = joblib.load(os.path.join(data_dir, "random_forest_model.pkl"))

class DenoisingThread(QThread):
    histReady  = pyqtSignal(np.ndarray)
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
        # push onto the queue (never overwritten)
        self._queue.put((
            raw_batch.copy(),
            batch_meta.copy().reset_index(drop=True),
            batch_indices.copy(),
        ))

    def run(self):
        while self._running:
            try:
                raw, meta, indices = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # your feature extraction + classification
            features = extract_omp_features(
                raw.T, bL=self.bL, aL=self.aL, bH=self.bH, aH=self.aH
            )
            classes = np.asarray(classifier.predict(features), dtype=int)

            # build per‐batch histogram
            batch_hist = np.zeros((2, self.num_channels), dtype=int)
            for cls, ch in zip(classes, meta["channel"].astype(int)):
                batch_hist[cls, ch] += 1

            # emit back to GUI
            self.classReady.emit(indices, classes)
            self.histReady.emit(batch_hist)

            self._queue.task_done()

    def stop(self):
        self._running = False
        self.wait()
