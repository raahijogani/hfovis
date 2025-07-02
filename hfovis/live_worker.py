from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
import pyqtgraph as pg
import numpy as np

class LivePlotWorker(QObject):
    plot_ready = pyqtSignal(pg.PlotWidget)

    def __init__(self, channel_names, spacing=1000.0):
        super().__init__()
        self.channel_names = channel_names
        self.spacing = spacing
        self._setup_plot()

    def _setup_plot(self):
        self.plot = pg.PlotWidget()
        self.plot.setBackground("k")

        self.curves = []
        self.offsets = np.arange(len(self.channel_names)) * self.spacing

        for _ in self.channel_names:
            c = self.plot.plot(pen="w")
            c.setClipToView(True)
            c.setDownsampling(auto=True, ds="peak")
            self.curves.append(c)

        yticks = [(offset, name) for offset, name in zip(self.offsets, self.channel_names)]
        self.plot.getAxis("left").setTicks([yticks])

        self.plot.setYRange(-self.spacing / 2, self.offsets[-1] + self.spacing / 2, padding=0)
        self.plot.setLabel("left", "Channel")
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        self.plot_ready.emit(self.plot)

    @pyqtSlot(object)
    def update_plot(self, data):
        t, y = data
        y = y + self.offsets
        for curve, ycol in zip(self.curves, y.T):
            curve.setData(t, ycol)

        if len(t) > 0:
            self.plot.setXRange(t[0], t[-1], padding=0)
