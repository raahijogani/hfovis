import re
from collections import defaultdict
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import QMainWindow
from PyQt6 import QtCore
from hfovis.interface import Ui_MainWindow
from scipy import signal
from scipy.signal import butter, filtfilt


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

        # Raster‑plot scroll state
        self.window_secs = 10.0  # x‑range shown (user adjustable)
        self.auto_scroll = True  # follow live data unless user pans
        self._raster_updating = False  # guard to avoid feedback loops

        # ─── Time‑series plots ---------------------------------------
        self._init_time_series_plots()
        self._init_spectrogram()
        self._init_raster_plot()
        self._init_frequency_plot()
        self._connect_ui()

    # ==================================================================
    # Initialisation helpers -------------------------------------------
    def _init_time_series_plots(self):
        self.rawPlot = self.rawEventPlot
        self.filteredPlot = self.filteredEventPlot

        self.rawPlot.setBackground("#f8f8f8")
        self.rawPlot.getPlotItem().getViewBox().setBackgroundColor("k")
        self.rawPlot.getPlotItem().getAxis("bottom").setTextPen("k")
        self.rawPlot.getPlotItem().getAxis("left").setTextPen("k")

        self.filteredPlot.setBackground("#f8f8f8")
        self.filteredPlot.getPlotItem().getViewBox().setBackgroundColor("k")
        self.filteredPlot.getPlotItem().getAxis("bottom").setTextPen("k")
        self.filteredPlot.getPlotItem().getAxis("left").setTextPen("k")

        self.rawPlot.setYRange(-250, 250)
        self.filteredPlot.setYRange(-25, 25)
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
            p.setMouseEnabled(x=False, y=True)  # disable mouse panning/zooming

    def _init_spectrogram(self):
        # Create high pass parameters for spectrogram
        self.spec_a, self.spec_b = butter(2, 16, fs=self.fs, btype="highpass")

        # Set light background
        self.eventSpectrogram.setBackground("#f8f8f8")
        self.eventSpectrogram.getPlotItem().getAxis("bottom").setPen("k")
        self.eventSpectrogram.getPlotItem().getAxis("left").setPen("k")
        self.eventSpectrogram.getPlotItem().getAxis("bottom").setTextPen("k")
        self.eventSpectrogram.getPlotItem().getAxis("left").setTextPen("k")
        self.eventSpectrogram.getPlotItem().getAxis("bottom").setTickPen(None)
        self.eventSpectrogram.getPlotItem().getAxis("left").setTickPen(None)

        # Get rid of padding
        self.eventSpectrogram.getViewBox().setDefaultPadding(0)
        self.eventSpectrogram.getPlotItem().setContentsMargins(0, 10, 0, 0)

        self.specImg = pg.ImageItem()
        self.eventSpectrogram.addItem(self.specImg)
        self.colormap = pg.colormap.get("inferno")
        self.cbar = pg.ColorBarItem(colorMap=self.colormap, label="Power (dB)")
        self.cbar.getAxis("right").setPen("k")
        self.cbar.getAxis("right").setTextPen("k")
        self.cbar.getAxis("left").setLabel(color="k")
        self.cbar.setImageItem(
            self.specImg, insert_in=self.eventSpectrogram.getPlotItem()
        )
        self.eventSpectrogram.setLabel("bottom", "Time", units="s")
        self.eventSpectrogram.setLabel("left", "Frequency", units="Hz")

        # Disable mouse
        self.eventSpectrogram.getPlotItem().setMouseEnabled(x=False, y=False)

    def _init_raster_plot(self):
        """Configure rasterPlot for scrolling spike‑like events."""
        self.rasterScatter = pg.ScatterPlotItem(size=4, brush="w", pen=None)

        self.rasterPlot.addItem(self.rasterScatter)

        self.rasterPlot.setBackground("#f8f8f8")
        self.rasterPlot.getPlotItem().getViewBox().setBackgroundColor("k")
        self.rasterPlot.getPlotItem().getAxis("bottom").setTextPen("k")
        self.rasterPlot.getPlotItem().getAxis("left").setTextPen("k")

        self.rasterPlot.setLabel("left", "Channel")
        self.rasterPlot.setLabel("bottom", "Time", units="s")
        self.rasterPlot.setYRange(-0.5, len(self.channel_names) - 0.5, padding=0)

        # Fixed y‑ticks with channel labels
        yticks = self._update_raster_ticks(self.channel_names)
        self.rasterPlot.getAxis("left").setTicks([yticks])

        self.rasterPlot.setMouseEnabled(x=False, y=False)  # disable mouse

    def _init_frequency_plot(self):
        self.freq_bins = np.linspace(0, 600, 61)  # 60 bins → 10 Hz resolution
        self.freq_hist = np.zeros((len(self.freq_bins) - 1, len(self.channel_names)))

        # Create ImageItem
        self.freqImg = pg.ImageItem()
        self.frequencyPlot.addItem(self.freqImg)

        # Configure axes
        self.frequencyPlot.setLabel("bottom", "Channel")
        self.frequencyPlot.setLabel("left", "Frequency (Hz)")

        # Set light background
        self.frequencyPlot.setBackground("#f8f8f8")
        self.frequencyPlot.getPlotItem().getAxis("bottom").setPen("k")
        self.frequencyPlot.getPlotItem().getAxis("left").setPen("k")
        self.frequencyPlot.getPlotItem().getAxis("bottom").setTextPen("k")
        self.frequencyPlot.getPlotItem().getAxis("left").setTextPen("k")
        self.frequencyPlot.getPlotItem().getAxis("bottom").setTickPen(None)
        self.frequencyPlot.getPlotItem().getAxis("left").setTickPen(None)

        # Get rid of padding
        self.frequencyPlot.getViewBox().setDefaultPadding(0)
        self.frequencyPlot.getPlotItem().setContentsMargins(0, 10, 0, 0)

        # Remove axis ticks
        self.frequencyPlot.getPlotItem().getAxis("bottom").setTickPen(None)
        self.frequencyPlot.getPlotItem().getAxis("left").setTickPen(None)

        # Set ticks
        xticks = self._update_raster_ticks(self.channel_names)
        self.frequencyPlot.getAxis("bottom").setTicks([xticks])

        # Colorbar
        self.freq_cbar = pg.ColorBarItem(colorMap=self.colormap, label="Count")
        self.freq_cbar.getAxis("right").setPen("k")
        self.freq_cbar.getAxis("right").setTextPen("k")
        self.freq_cbar.getAxis("left").setLabel(color="k")
        self.freq_cbar.setImageItem(
            self.freqImg, insert_in=self.frequencyPlot.getPlotItem()
        )

    def _update_raster_ticks(self, channel_names):
        """
        Group channels by shared prefix and show at most one label per group.
        """
        groups = defaultdict(list)
        for i, label in enumerate(channel_names):
            match = re.match(r"([A-Za-z]+)", label)
            key = match.group(1) if match else label
            groups[key].append(i)

        yticks = []
        for name, indices in groups.items():
            center_idx = indices[len(indices) // 2]
            yticks.append((center_idx, name))

        yticks.sort()

        return yticks

    def _connect_ui(self):
        self.showRawSpectrogramButton.setChecked(True)
        self.showRawSpectrogramButton.toggled.connect(self.toggle_spectrogram)
        self.showFilteredSpectrogramButton.toggled.connect(self.toggle_spectrogram)
        self.eventNumBox.valueChanged.connect(self._on_event_index_changed)
        self.nextEventButton.clicked.connect(self.next_event)
        self.previousEventButton.clicked.connect(self.previous_event)
        self.lastEventButton.clicked.connect(self.last_event)
        self.firstEventButton.clicked.connect(self.first_event)
        # If there is a spinBox (optional) named windowLengthSpinBox, wire it:
        if hasattr(self, "windowLengthSpinBox"):
            self.windowLengthSpinBox.setValue(self.window_secs)
            self.windowLengthSpinBox.valueChanged.connect(self.set_raster_window)

    # ==================================================================
    # Worker‑thread callback -------------------------------------------
    def handle(self, event: dict):
        """Slot for RealTimeDetector; executed in worker thread."""
        self.new_event.emit(event)

    # ==================================================================
    # GUI‑thread slot ---------------------------------------------------
    @QtCore.pyqtSlot(dict)
    def _on_event_received(self, event: dict):
        """Merge incoming batch and update plots (including raster)."""

        raw_batch = event["raw"].T
        filt_batch = event["filtered"].T

        # initialise x‑axis for waveforms
        if self.event_t is None:
            self.event_t = np.arange(raw_batch.shape[1]) / self.fs
            self.eventNumBox.setMinimum(1)

        # ------ Expand master buffers --------------------------------
        if self.raw_events is None:
            self.raw_events = raw_batch.copy()
            self.filtered_events = filt_batch.copy()
        else:
            self.raw_events = np.vstack((self.raw_events, raw_batch))
            self.filtered_events = np.vstack((self.filtered_events, filt_batch))

        # ------ Build metadata frame ---------------------------------
        center_arr = np.asarray(event["center"], dtype=float)
        chan_vec = np.asarray(event["channels"], dtype=int)
        thresh_arr = np.asarray(event["threshold"], dtype=float)

        batch_meta = pd.DataFrame(
            {
                "center": center_arr,
                "channel": chan_vec,
                "threshold": thresh_arr,
            }
        )
        self.meta = (
            pd.concat([self.meta, batch_meta], ignore_index=True)
            if self.meta is not None
            else batch_meta
        )

        # ------ Raster plot update -----------------------------------
        self._append_raster_points(center_arr, chan_vec)
        self._update_raster_view(center_arr.max())

        # ------ Frequency histogram update --------------------------
        self._update_frequency_histogram(filt_batch, chan_vec)

        # ------ GUI counters -----------------------------------------
        n_events = len(self.meta)
        self.eventNumBox.setMaximum(n_events)
        self.numEventsLabel.setText(f"of {n_events}")
        if self.show_latest:
            self.eventNumBox.setValue(n_events)  # jump to latest

    # ==================================================================
    # Raster helpers ---------------------------------------------------
    def _append_raster_points(self, times: np.ndarray, chans: np.ndarray):
        """Add new points to the scatter item."""
        self.rasterScatter.addPoints(x=times, y=chans)

    def _update_raster_view(self, newest_time: float):
        """Auto‑scroll unless the user has taken manual control."""
        if self.auto_scroll:
            self._raster_updating = True  # suppress pan callback
            self.rasterPlot.setXRange(
                newest_time - self.window_secs, newest_time, padding=0
            )
            self._raster_updating = False

    def set_raster_window(self, secs: float):
        """Setter for the visible time window (callable from UI)."""
        self.window_secs = float(secs)
        # Force update to current view if in live mode
        if self.meta is not None and self.auto_scroll:
            self._update_raster_view(self.meta["center"].iloc[-1])

    def catch_up_live(self):
        """Re‑enable auto‑scroll and jump to newest data (bind to a button)."""
        self.auto_scroll = True
        if self.meta is not None:
            self._update_raster_view(self.meta["center"].iloc[-1])

    # ==================================================================
    # Frequency Plot Helper ----------------------------------------------
    def _update_frequency_histogram(self, filtered_batch, chan_vec):
        # Compute dominant freq for each event
        freqs = np.fft.rfftfreq(filtered_batch.shape[1], d=1 / self.fs)
        fft_vals = np.fft.rfft(filtered_batch, axis=1)
        mags = np.abs(fft_vals)
        dom_idx = np.argmax(mags, axis=1)
        dom_freqs = freqs[dom_idx]

        # Update histogram
        for freq, chan in zip(dom_freqs, chan_vec):
            bin_idx = np.searchsorted(self.freq_bins, freq, side="right") - 1
            if 0 <= bin_idx < len(self.freq_bins) - 1:
                self.freq_hist[bin_idx, chan] += 1

        # Update image
        self.freqImg.setImage(self.freq_hist.T, autoLevels=True)

        # Set axes
        self.freqImg.setRect(
            pg.QtCore.QRectF(
                0,
                self.freq_bins[0],
                len(self.channel_names),
                self.freq_bins[-1] - self.freq_bins[0],
            )
        )

        # Set colorbar levels
        self.freq_cbar.setLevels((0, np.max(self.freq_hist)))

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
            plot_widget.setTitle(f"Center: {center:.3f} s", color="k")
        plot_widget.setXRange(0, sig.size / self.fs, padding=0)

    # ------------------------------------------------------------------
    def plot_spectrogram(self, sig):
        sig = filtfilt(self.spec_a, self.spec_b, sig, axis=0)
        sig = signal.detrend(sig, type="constant")
        # Normalize by l2 norm
        sig = sig / np.linalg.norm(sig)
        f, t, Zxx = signal.spectrogram(
            sig,
            fs=self.fs,
            window="hann",
            nperseg=128,
            noverlap=127,
            nfft=1024,
            scaling="density",
            mode="magnitude",
        )
        mask = f <= 600
        Zxx = np.abs(Zxx[mask])
        Zxx /= np.max(Zxx)
        Z = 20 * np.log10(Zxx + 1e-6)

        self.specImg.setImage(
            Z, autoLevels=False, autoDownsample=True, axes={"x": 1, "y": 0}
        )
        self.specImg.setRect(
            pg.QtCore.QRectF(t[0], f[0], t[-1] - t[0], f[mask][-1] - f[0])
        )
        # self.cbar.setLevels((np.nanmin(Z), np.nanmax(Z)))
        self.cbar.setLevels((-30, 0))

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
