from PyQt6.QtCore import QObject, pyqtSignal, QRunnable
import numpy as np


class EventWorkerSignals(QObject):
    finished = pyqtSignal(object, object, object, object, object)


class EventWorker(QRunnable):
    def __init__(self, event: dict, fs: float):
        super().__init__()
        self.event = event
        self.fs = fs
        self.signals = EventWorkerSignals()

    def run(self):
        raw_batch = self.event["raw"].T
        filt_batch = self.event["filtered"].T

        center_arr = np.asarray(self.event["center"], dtype=float)
        chan_vec = np.asarray(self.event["channels"], dtype=int)
        thresh_arr = np.asarray(self.event["threshold"], dtype=float)

        self.signals.finished.emit(
            raw_batch, filt_batch, center_arr, chan_vec, thresh_arr
        )
