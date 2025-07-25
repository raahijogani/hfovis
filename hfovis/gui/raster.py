import numpy as np
import pandas as pd
import pyqtgraph as pg
from hfovis.gui.model import EventModel
from PyQt6.QtWidgets import QCheckBox


class RasterPlot:
    def __init__(
        self,
        plot_widget: pg.PlotWidget,
        channel_groups: list[tuple[int, str]],
        num_channels: int,
        model: EventModel,
        pseudo_check_box: QCheckBox,
        event_num_box: pg.SpinBox,
    ):
        self.rasterPlot = plot_widget
        self.model = model
        self.showPseudoEventBox = pseudo_check_box
        self.eventNumBox = event_num_box
        self.window_secs = 10.0

        self.rasterScatter = pg.ScatterPlotItem(size=4, brush="w", pen=None)

        self.rasterPlot.addItem(self.rasterScatter)

        self.rasterPlot.setBackground("#f8f8f8")
        self.rasterPlot.getPlotItem().getViewBox().setBackgroundColor("k")
        self.rasterPlot.getPlotItem().getAxis("bottom").setTextPen("k")
        self.rasterPlot.getPlotItem().getAxis("left").setTextPen("k")

        self.rasterPlot.setLabel("left", "Channel")
        self.rasterPlot.setLabel("bottom", "Time", units="s")

        # hide auto-scale and menu buttons
        self.rasterPlot.getPlotItem().hideButtons()

        # add a special scatter for the “selected” event
        self.selectedScatter = pg.ScatterPlotItem(
            size=10,
            symbol="x",
            pen="w",  # default, will be updated per‐point
            brush=None,
        )
        self.rasterPlot.addItem(self.selectedScatter)

        # Fixed y‑ticks with channel labels
        self.rasterPlot.getAxis("left").setTicks([channel_groups])
        self.rasterPlot.setYRange(-0.5, num_channels - 0.5, padding=0)

        self.rasterPlot.setMouseEnabled(x=False, y=False)  # disable mouse

    def update(self):
        if self.model.meta is None:
            return

        # — build color list for *all* events —
        vals = self.model.meta["is_real"]
        brushes = []
        for v in vals:
            if pd.isna(v):
                brushes.append("w")  # pending
            elif v:
                brushes.append("g")  # real
            else:
                brushes.append("r")  # pseudo

        # — mask out pseudo unless checkbox ticked —
        if self.showPseudoEventBox.isChecked():
            mask = np.ones_like(vals.to_numpy(), dtype=bool)
        else:
            mask = (vals.isna() | vals).to_numpy()

        times = self.model.meta["center"].to_numpy()[mask]
        chans = self.model.meta["channel"].to_numpy()[mask]
        colors = [brushes[i] for i in np.nonzero(mask)[0]]

        # redraw main dots
        self.rasterScatter.setData(x=times, y=chans, brush=colors)

        # — now draw the “×” at the selected event —
        idx = self.eventNumBox.value() - 1
        if 0 <= idx < len(self.model.meta):
            t_sel = float(self.model.meta["center"].iat[idx])
            chan_sel = int(self.model.meta["channel"].iat[idx])
            v = self.model.meta["is_real"].iat[idx]

            if pd.isna(v):
                sel_color = "w"
            elif v:
                sel_color = "g"
            else:
                sel_color = "r"

            # always draw exactly one “×”
            self.selectedScatter.setData(x=[t_sel], y=[chan_sel], pen=sel_color)

        if self.eventNumBox.value() == len(self.model.meta):
            self._update_raster_view(
                max(times.max(), self.model.meta["center"].iat[idx])
            )
        else:
            self._center_raster_on_event(float(self.model.meta["center"].iat[idx]))

    def _center_raster_on_event(self, t_center: float):
        T = self.window_secs
        t_last = float(self.model.meta["center"].iloc[-1])
        half = T / 2

        lo = t_center - half
        hi = t_center + half

        if lo < 0:
            lo, hi = 0, min(T, t_last)
        elif hi > t_last:
            hi, lo = t_last, max(0, t_last - T)

        self.rasterPlot.setXRange(lo, hi, padding=0)

    def _update_raster_view(self, newest_time: float):
        self.rasterPlot.setXRange(
            newest_time - self.window_secs, newest_time, padding=0
        )

    def set_raster_window(self, secs: float):
        """Setter for the visible time window (callable from UI)."""
        self.window_secs = float(secs)
        if self.model.meta is not None:
            self.update()
