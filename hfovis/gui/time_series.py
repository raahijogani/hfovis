import numpy as np
import pyqtgraph as pg


class TimeSeriesPlot:
    def __init__(
        self,
        raw_plot: pg.PlotWidget,
        filtered_plot: pg.PlotWidget,
        fs: float,
    ):
        self.fs = fs
        self.rawPlot = raw_plot
        self.filteredPlot = filtered_plot

        self.rawPlot.setBackground("#f8f8f8")
        self.rawPlot.getPlotItem().getViewBox().setBackgroundColor("k")
        self.rawPlot.getPlotItem().getAxis("bottom").setTextPen("k")
        self.rawPlot.getPlotItem().getAxis("left").setTextPen("k")

        self.filteredPlot.setBackground("#f8f8f8")
        self.filteredPlot.getPlotItem().getViewBox().setBackgroundColor("k")
        self.filteredPlot.getPlotItem().getAxis("bottom").setTextPen("k")
        self.filteredPlot.getPlotItem().getAxis("left").setTextPen("k")

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

    def update(self, raw: np.ndarray, filt: np.ndarray, center: float, thr: float):
        self._update_window(raw, self.rawCurve, self.rawPlot, center)
        self._update_window(filt, self.filtCurve, self.filteredPlot)
        self.lowerThreshLine.setPos(-thr)
        self.upperThreshLine.setPos(thr)

    def _update_window(
        self,
        sig: np.ndarray,
        curve: pg.PlotCurveItem,
        plot_widget: pg.PlotWidget,
        center: float | None = None,
    ):
        event_t = np.arange(sig.size) / self.fs
        curve.setData(event_t, sig)
        if center is not None:
            plot_widget.setTitle(f"Center: {center:.3f} s", color="k")

        plot_widget.setXRange(0, sig.size / self.fs, padding=0)

        # Add vertical padding
        y_min = np.min(sig)
        y_max = np.max(sig)
        y_range = y_max - y_min
        if y_range == 0:
            y_min -= 1
            y_max += 1
        else:
            pad = 0.5 * y_range
            y_min -= pad
            y_max += pad
        plot_widget.setYRange(y_min, y_max, padding=0)
