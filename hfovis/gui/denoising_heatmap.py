import numpy as np
import pyqtgraph as pg


class DenoisingHeatmapPlot:
    def __init__(
        self,
        plot_widget: pg.PlotWidget,
        channel_groups: list[tuple[int, str]],
        num_channels: int,
        colormap: pg.ColorMap,
    ):
        """Set up the two‐row heatmap (pseudo vs real)."""
        self.denoisingHeatmap = plot_widget
        self.denoiseImg = pg.ImageItem()
        self.denoisingHeatmap.addItem(self.denoiseImg)
        self.denoisingHeatmap.setBackground("#f8f8f8")
        self.denoisingHeatmap.getPlotItem().getAxis("left").setPen("k")
        self.denoisingHeatmap.getPlotItem().getAxis("bottom").setPen("k")
        self.denoisingHeatmap.getPlotItem().getAxis("left").setTextPen("k")
        self.denoisingHeatmap.getPlotItem().getAxis("bottom").setTextPen("k")
        self.denoisingHeatmap.getPlotItem().getAxis("bottom").setTickPen(None)
        self.denoisingHeatmap.getPlotItem().getAxis("left").setTickPen(None)
        self.denoisingHeatmap.getPlotItem().getAxis("bottom").setStyle(
            hideOverlappingLabels=False
        )
        self.denoisingHeatmap.setLabel("left", "")
        self.denoisingHeatmap.setLabel("bottom", "Channel")
        # fixed y-ticks at 0 and 1
        self.denoisingHeatmap.getPlotItem().getAxis("left").setTicks(
            [[(0, "Pseudo-HFO"), (1, "Real-HFO")]]
        )
        # Set x-ticks
        self.denoisingHeatmap.getAxis("bottom").setTicks([channel_groups])

        self.denoisingHeatmap.getViewBox().setDefaultPadding(0)
        self.denoisingHeatmap.getPlotItem().setContentsMargins(0, 10, 0, 0)

        # add colorbar
        self.denoise_cbar = pg.ColorBarItem(
            colorMap=colormap,
            label="Count",
            interactive=False,
            pen=None,
            hoverPen=None,
            hoverBrush=None,
        )
        self.denoise_cbar.setImageItem(
            self.denoiseImg, insert_in=self.denoisingHeatmap.getPlotItem()
        )
        self.denoise_cbar.getAxis("right").setTextPen("k")
        self.denoise_cbar.getAxis("left").setTextPen("k")

        # cumulative counts across all batches
        self._denoise_hist = np.zeros((2, num_channels), dtype=int)

    def update(self, hist_batch: np.ndarray):
        self._denoise_hist += hist_batch
        self.denoiseImg.setImage(self._denoise_hist, autoLevels=True)

        h, w = self._denoise_hist.shape
        self.denoiseImg.setRect(pg.QtCore.QRectF(0, 0, w, h))

        self.denoise_cbar.setLevels((0, np.max(self._denoise_hist)))
        axis = self.denoise_cbar.getAxis("right")
        maxv = float(self._denoise_hist.max())
        axis.setTicks([[(maxv, str(int(maxv)))]])
