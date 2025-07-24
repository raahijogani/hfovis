import numpy as np
import pyqtgraph as pg


class FrequencyPlot:
    def __init__(
        self,
        plot_widget: pg.PlotWidget,
        channel_groups: list[tuple[int, str]],
        num_channels: int,
        fs: float,
        colormap: pg.ColorMap,
    ):
        self.fs = fs
        self.frequencyPlot = plot_widget
        self.num_channels = num_channels

        self.freq_bins = np.linspace(0, 600, 61)  # 60 bins → 10 Hz resolution
        self.freq_hist = np.zeros((len(self.freq_bins) - 1, self.num_channels), dtype=int)

        # Create ImageItem
        self.freqImg = pg.ImageItem()
        self.frequencyPlot.addItem(self.freqImg)

        # Configure axes
        self.frequencyPlot.setLabel("bottom", "Channel")
        self.frequencyPlot.setLabel("left", "Frequency (Hz)")
        self.frequencyPlot.getPlotItem().getAxis("bottom").setStyle(
            hideOverlappingLabels=False
        )

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
        self.frequencyPlot.getAxis("bottom").setTicks([channel_groups])

        # Colorbar
        self.freq_cbar = pg.ColorBarItem(
            colorMap=colormap,
            label="Count",
            interactive=False,
            pen=None,
            hoverPen=None,
            hoverBrush=None,
        )
        self.freq_cbar.getAxis("right").setPen("k")
        self.freq_cbar.getAxis("right").setTextPen("k")
        self.freq_cbar.getAxis("left").setLabel(color="k")
        self.freq_cbar.setImageItem(
            self.freqImg, insert_in=self.frequencyPlot.getPlotItem()
        )

    def update(self, filtered_batch: np.ndarray, chan_vec: np.ndarray) -> np.ndarray:
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
                self.num_channels,
                self.freq_bins[-1] - self.freq_bins[0],
            )
        )

        # Set colorbar levels
        self.freq_cbar.setLevels((0, np.max(self.freq_hist)))
