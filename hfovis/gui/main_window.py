import re
from PyQt6 import QtCore
from PyQt6.QtWidgets import QMainWindow
from pyqtgraph import colormap

import numpy as np
import pandas as pd

from hfovis.interface import Ui_MainWindow
from hfovis.gui.model import EventModel
from hfovis.gui.time_series import TimeSeriesPlot
from hfovis.gui.denoising_heatmap import DenoisingHeatmapPlot
from hfovis.gui.spectrogram import SpectrogramPlot
from hfovis.gui.raster import RasterPlot
from hfovis.gui.frequency import FrequencyPlot
from hfovis.gui.config_menu import ConfigMenu
from hfovis.gui.config import GeneralConfig

from hfovis.data.streaming import Streamer

from hfovis.detector import RealTimeDetector
from hfovis.denoiser import DenoisingThread


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(
        self,
        fs: float,
        streamer: Streamer,
        channel_names: list[str] | None = None,
        n_channels: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.setupUi(self)

        self.fs = fs
        self.streamer = streamer

        self.config = GeneralConfig()

        if self.config.montage_location:
            with open(self.config.montage_location, "r") as f:
                self.channel_names = [line.strip()
                                      for line in f if line.strip()]
            self.channel_groups = self._create_channel_groups(channel_names)
            n_channels = len(self.channel_names)
        elif channel_names:
            self.channel_names = channel_names
            self.channel_groups = self._create_channel_groups(channel_names)
            n_channels = len(self.channel_names)
        elif n_channels:
            self.channel_names = [f"{i + 1}" for i in range(n_channels)]
            self.channel_groups = [(i, f"{i + 1}") for i in range(n_channels)]
        else:
            raise ValueError(
                "No channel names provided and no montage file specified in config."
            )

        self.show_latest = True

        self.detector_thread = RealTimeDetector(
            streamer,
            fs=self.fs,
            channels=len(self.channel_names),
            **kwargs,
        )

        self.config_menu = ConfigMenu(
            self.configScrollAreaWidgetContents,
            [self.config, self.detector_thread.config],
        )

        self.model = EventModel(n_channels)

        # Plots
        self.timeSeriesPlot = TimeSeriesPlot(
            self.rawEventPlot, self.filteredEventPlot, fs
        )
        self.denoisingHeatmapPlot = DenoisingHeatmapPlot(
            self.denoisingHeatmap,
            self.channel_groups,
            n_channels,
            colormap.get("inferno"),
        )
        self.spectrogramPlot = SpectrogramPlot(
            self.eventSpectrogram, self.fs, colormap.get("turbo")
        )
        self.rasterPlot = RasterPlot(
            self.rasterPlot,
            self.channel_groups,
            n_channels,
            self.model,
            self.showPseudoEventBox,
            self.eventNumBox,
        )
        self.frequencyPlot = FrequencyPlot(
            self.frequencyPlot,
            self.channel_groups,
            n_channels,
            self.fs,
            colormap.get("inferno"),
        )

        self._connect_ui()

    # Slots
    @QtCore.pyqtSlot(dict)
    def _on_event_received(self, event: dict):
        old_len = len(self.model.meta) if self.model.meta is not None else 0
        was_at_end = self.eventNumBox.value() == old_len

        raw_batch, filt_batch, batch_meta, batch_indices, chan_vec = (
            self.model.append_batch(event)
        )

        self.frequencyPlot.update(filt_batch, chan_vec)

        new_len = len(self.model.meta)
        self.eventNumBox.setRange(1, new_len)
        self.numEventsLabel.setText(f"of {new_len}")

        if was_at_end:
            self.eventNumBox.setValue(new_len)
            self.rasterPlot.update()

        self.denoise_thread.enqueue(raw_batch, batch_meta, batch_indices)

    @QtCore.pyqtSlot(np.ndarray)
    def _on_hist_ready(self, hist_batch: np.ndarray):
        self.denoisingHeatmapPlot.update(hist_batch)

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def _on_classification_ready(self, indices: np.ndarray, classes: np.ndarray):
        self.model.meta.loc[indices, "is_real"] = classes == 1
        self._update_classification_label()
        self.rasterPlot.update()

    # Navigation methods
    def _on_event_index_changed(self):
        if self.model.meta is None:
            return
        self.show_latest = self.eventNumBox.value() == len(self.model.meta)
        self._plot_event(self.eventNumBox.value() - 1)
        self.rasterPlot.update()

    def last_event(self):
        if self.model.meta is not None:
            self.eventNumBox.setValue(len(self.model.meta))

    def first_event(self):
        if self.model.meta is not None:
            self.eventNumBox.setValue(1)

    def next_event(self):
        if self.model.meta is None:
            return
        i, n = self.eventNumBox.value(), len(self.model.meta)
        self.eventNumBox.setValue(i + 1 if i < n else 1)

    def previous_event(self):
        if self.model.meta is None:
            return
        i, n = self.eventNumBox.value(), len(self.model.meta)
        self.eventNumBox.setValue(i - 1 if i > 1 else n)

    def _connect_ui(self):
        self.eventNumBox.valueChanged.connect(self._on_event_index_changed)

        self.nextEventButton.clicked.connect(self.next_event)
        self.previousEventButton.clicked.connect(self.previous_event)
        self.lastEventButton.clicked.connect(self.last_event)
        self.firstEventButton.clicked.connect(self.first_event)

        self.showPseudoEventBox.toggled.connect(self.rasterPlot.update)
        self.windowLengthSpinBox.valueChanged.connect(
            self.rasterPlot.set_raster_window)

        self.startButton.clicked.connect(self._start)
        self.saveButton.clicked.connect(self.save)

        self.applyConfigButton.clicked.connect(self.config_menu.apply_changes)
        self.resetDefaultsButton.clicked.connect(
            self.config_menu.reset_defaults)

    def save(self):
        self.model.save(
            raw_filename=self.config.raw_data_filename,
            filt_filename=self.config.filtered_data_filename,
            meta_filename=self.config.metadata_filename,
        )

    def _start(self):
        self._start_detector_thread(self.streamer)
        self._start_denoise_thread()
        self.startButton.setEnabled(False)
        self.config_menu.disable_editing()

    def _start_detector_thread(self, streamer: Streamer):
        self.detector_thread.new_event.connect(
            self._on_event_received, QtCore.Qt.ConnectionType.QueuedConnection
        )
        self.detector_thread.start()

    def _start_denoise_thread(self):
        self.denoise_thread = DenoisingThread(
            num_channels=len(self.channel_names),
            fs=self.fs,
        )

        self.denoise_thread.histReady.connect(self.denoisingHeatmapPlot.update)
        self.denoise_thread.classReady.connect(self._on_classification_ready)

        self.denoise_thread.start()

    # Additional helpers
    def _create_channel_groups(self, channel_names: list[str]) -> list[tuple[int, str]]:
        """
        Group bipolar channels by shared prefix (e.g., LA, LAH, LPH) and
        return ticks at the first occurrence of each group.
        """
        groups = {}  # prefix -> first index

        for i, label in enumerate(channel_names):
            match = re.match(r"([A-Z]+)", label)  # Extract "LA", "LAH", etc.
            if not match:
                continue
            prefix = match.group(1)
            if prefix not in groups:
                groups[prefix] = i  # Only store the first occurrence

        # Convert to sorted list of ticks
        yticks = sorted((idx, prefix) for prefix, idx in groups.items())
        return yticks

    def _update_classification_label(self):
        cur = self.eventNumBox.value() - 1
        val = self.model.meta.at[cur, "is_real"]

        if pd.isna(val):
            text, color = "Classification\npending", "black"
        elif val:
            text, color = "Real event", "green"
        else:
            text, color = "Pseudo event", "red"

        self.eventClassificationLabel.setText(text)
        self.eventClassificationLabel.setStyleSheet(f"color: {color};")

    def _plot_event(self, idx: int):
        if self.model.meta is None or not (0 <= idx < len(self.model.meta)):
            return

        raw = self.model.raw_events[idx]
        filt = self.model.filtered_events[idx]
        row = self.model.meta.iloc[idx]
        chan, cent, thresh = int(row.channel), float(
            row.center), float(row.threshold)

        self.timeSeriesPlot.update(raw, filt, cent, thresh)
        self.spectrogramPlot.update(raw)

        self.channelLabel.setText(self.channel_names[chan])
        self._update_classification_label()

    def closeEvent(self, event):
        self.save()
        event.accept()
