import sys
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6 import QtCore
from hfovis.interface import Ui_MainWindow
from hfovis.data import streaming, ieeg_loader
from hfovis.detector.detector import RealTimeDetector
from scipy import signal


class MainWindow(QMainWindow, Ui_MainWindow):
    """Real‑time HFO viewer with **NumPy‑backed signal arrays** and a
    **DataFrame for per‑event metadata**.

    Storage layout
    --------------
    ``self.raw_events``      → ``ndarray``  shape *(N, S)*  – raw traces
    ``self.filtered_events`` → ``ndarray``  shape *(N, S)*  – filtered traces
    ``self.meta``            → ``DataFrame`` columns [center, channel, threshold]

    ``N`` = number of detected events, ``S`` = samples per event.
    Row *i* of *meta* corresponds to row *i* of both arrays.
    """

    # ------------------------------------------------------------------
    new_event = QtCore.pyqtSignal(dict)  # worker → GUI thread

    # ------------------------------------------------------------------
    def __init__(self, fs: float, channel_names):
        super().__init__()
        self.setupUi(self)
        self.new_event.connect(
            self._on_event_received, QtCore.Qt.ConnectionType.QueuedConnection
        )

        # ─── State ----------------------------------------------------
        self.fs = fs
        self.channel_names = channel_names
        self.raw_events: np.ndarray | None = None  # (N, S)
        self.filtered_events: np.ndarray | None = None  # (N, S)
        self.meta: pd.DataFrame | None = None  # per‑event info
        self.event_t: np.ndarray | None = None  # x‑axis for traces
        self.show_latest = True
        self.raw_event = True  # toggle raw/filtered spectrogram source

        # ─── Time‑series plots ---------------------------------------
        self.rawPlot = self.rawEventPlot
        self.filteredPlot = self.filteredEventPlot
        self.rawCurve = self.rawPlot.plot(pen="w")
        self.filtCurve = self.filteredPlot.plot(pen="w")
        self.lowerThreshLine = pg.InfiniteLine(pen="r", angle=0)
        self.upperThreshLine = pg.InfiniteLine(pen="r", angle=0)
        self.filteredPlot.addItem(self.lowerThreshLine)
        self.filteredPlot.addItem(self.upperThreshLine)
        for c in (self.rawCurve, self.filtCurve):
            c.setClipToView(True)
            c.setDownsampling(auto=True, ds="peak")
        self.rawPlot.setLabel("left", "Raw")
        self.filteredPlot.setLabel("left", "Filtered")
        for p in (self.rawPlot, self.filteredPlot):
            p.setLabel("bottom", "Time", units="s")
            p.showGrid(x=True, y=True, alpha=0.3)

        # ─── Spectrogram setup ---------------------------------------
        self.specImg = pg.ImageItem()
        self.eventSpectrogram.addItem(self.specImg)
        cmap = pg.colormap.get("viridis")
        self.cbar = pg.ColorBarItem(colorMap=cmap)
        self.cbar.setImageItem(
            self.specImg, insert_in=self.eventSpectrogram.getPlotItem()
        )
        self.eventSpectrogram.setLabel("bottom", "Time", units="s")
        self.eventSpectrogram.setLabel("left", "Frequency", units="Hz")

        # ─── UI connections ------------------------------------------
        self.showRawSpectrogramButton.setChecked(True)
        self.showRawSpectrogramButton.toggled.connect(self.toggle_spectrogram)
        self.showFilteredSpectrogramButton.toggled.connect(self.toggle_spectrogram)
        self.eventNumBox.valueChanged.connect(self._on_event_index_changed)
        self.nextEventButton.clicked.connect(self.next_event)
        self.previousEventButton.clicked.connect(self.previous_event)
        self.lastEventButton.clicked.connect(self.last_event)
        self.firstEventButton.clicked.connect(self.first_event)

    # ==================================================================
    # Worker‑thread callback -------------------------------------------
    def handle(self, event: dict):
        """Slot for RealTimeDetector; executed in worker thread."""
        self.new_event.emit(event)

    # ==================================================================
    # GUI‑thread slot ---------------------------------------------------
    @QtCore.pyqtSlot(dict)
    def _on_event_received(self, event: dict):
        """Merge *event* batch into internal buffers."""

        n_ch = len(event["channels"])  # events in this batch
        raw_batch = event["raw"].T  # (n_ch, S)
        filt_batch = event["filtered"].T  # (n_ch, S)

        # Initialise time axis once ----------------------------------
        if self.event_t is None:
            self.event_t = np.arange(raw_batch.shape[1]) / self.fs
            self.eventNumBox.setMinimum(1)

        # Expand master arrays ---------------------------------------
        if self.raw_events is None:
            self.raw_events = raw_batch.copy()
            self.filtered_events = filt_batch.copy()
        else:
            self.raw_events = np.vstack((self.raw_events, raw_batch))
            self.filtered_events = np.vstack((self.filtered_events, filt_batch))

        # ----- Build metadata DataFrame -----------------------------
        center_val = np.asarray(event["center"]).flatten()
        center_vec = np.full(
            n_ch, center_val[0] if center_val.size == 1 else center_val, dtype=float
        )

        chan_vec = np.asarray(event["channels"], dtype=int).flatten()
        assert chan_vec.size == n_ch, "channels length mismatch"

        thresh_arr = np.asarray(event["threshold"]).flatten()
        thresh_vec = np.full(
            n_ch, thresh_arr[0] if thresh_arr.size == 1 else thresh_arr, dtype=float
        )

        batch_meta = pd.DataFrame(
            {
                "center": center_vec,
                "channel": chan_vec,
                "threshold": thresh_vec,
            }
        )

        self.meta = (
            pd.concat([self.meta, batch_meta], ignore_index=True)
            if self.meta is not None
            else batch_meta
        )

        # ----- UI widgets -------------------------------------------
        n_events = len(self.meta)
        self.eventNumBox.setMaximum(n_events)
        self.numEventsLabel.setText(f"of {n_events}")
        if self.show_latest:
            self.eventNumBox.setValue(n_events)  # triggers plot

    # ==================================================================
    # Navigation helpers ----------------------------------------------
    def _on_event_index_changed(self):
        if self.meta is None:
            return
        self.show_latest = self.eventNumBox.value() == len(self.meta)
        self.plot_event(self.eventNumBox.value() - 1)

    def last_event(self):
        if self.meta is not None:
            self.eventNumBox.setValue(len(self.meta))

    def first_event(self):
        if self.meta is not None:
            self.eventNumBox.setValue(1)

    def next_event(self):
        if self.meta is None:
            return
        i, n = self.eventNumBox.value(), len(self.meta)
        self.eventNumBox.setValue(i + 1 if i < n else 1)

    def previous_event(self):
        if self.meta is None:
            return
        i, n = self.eventNumBox.value(), len(self.meta)
        self.eventNumBox.setValue(i - 1 if i > 1 else n)

    # ==================================================================
    # Plotting ---------------------------------------------------------
    def plot_event(self, idx: int):
        if self.meta is None or not (0 <= idx < len(self.meta)):
            return

        raw = self.raw_events[idx]
        filt = self.filtered_events[idx]
        row = self.meta.iloc[idx]
        chan, cent, thresh = int(row.channel), float(row.center), float(row.threshold)

        self._update_window(raw, self.rawCurve, self.rawPlot, cent)
        self._update_window(filt, self.filtCurve, self.filteredPlot)
        self.lowerThreshLine.setPos(-thresh)
        self.upperThreshLine.setPos(thresh)

        self.plot_spectrogram(raw if self.raw_event else filt)
        self.channelLabel.setText(self.channel_names[chan])

    def _update_window(self, sig, curve, plot_widget, center=None):
        curve.setData(self.event_t, sig)
        if center is not None:
            plot_widget.setTitle(f"Center: {center:.3f} s")
        plot_widget.setXRange(0, sig.size / self.fs, padding=0)
        plot_widget.setYRange(np.min(sig), np.max(sig), padding=0.05)

    # ------------------------------------------------------------------
    def plot_spectrogram(self, sig):
        nfft = min(1024, sig.size)
        noverlap = int(nfft * 0.9)
        f, t, Zxx = signal.stft(sig, fs=self.fs, nperseg=nfft, noverlap=noverlap)
        mask = f <= 600
        Z = 20 * np.log10(np.abs(Zxx[mask]) + 1e-6)

        self.specImg.setImage(
            Z, autoLevels=False, autoDownsample=True, axes={"x": 1, "y": 0}
        )
        self.specImg.setRect(
            pg.QtCore.QRectF(t[0], f[0], t[-1] - t[0], f[mask][-1] - f[0])
        )
        self.cbar.setLevels((np.nanmin(Z), np.nanmax(Z)))

    # ==================================================================
    def toggle_spectrogram(self):
        self.raw_event = self.showRawSpectrogramButton.isChecked()
        if self.meta is not None:
            self.plot_event(self.eventNumBox.value() - 1)

    # ==================================================================
    def closeEvent(self, event):
        """Persist buffers on shutdown."""
        if self.meta is not None and self.raw_events is not None:
            np.save("raw_events.npy", self.raw_events)
            np.save("filtered_events.npy", self.filtered_events)
            self.meta.to_pickle("events_meta.pkl")
        event.accept()
