import numpy as np
from scipy.signal import butter, filtfilt, detrend, spectrogram
import pyqtgraph as pg


class SpectrogramPlot:
    def __init__(self, plot_widget: pg.PlotWidget, fs: float, colormap: pg.ColorMap):
        self.eventSpectrogram = plot_widget
        self.fs = fs

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
        self.cbar = pg.ColorBarItem(colorMap=colormap, label="Power (dB)")
        self.cbar.getAxis("right").setPen("k")
        self.cbar.getAxis("right").setTextPen("k")
        self.cbar.getAxis("left").setLabel(color="k")
        self.cbar.setImageItem(
            self.specImg, insert_in=self.eventSpectrogram.getPlotItem()
        )
        self.eventSpectrogram.setLabel("bottom", "Time", units="s")
        self.eventSpectrogram.setLabel("left", "Frequency", units="Hz")

    def update(self, sig: np.ndarray):
        sig = filtfilt(self.spec_a, self.spec_b, sig, axis=0)
        sig = detrend(sig, type="constant")

        # Normalize by l2 norm
        sig = sig / np.linalg.norm(sig)

        f, t, Zxx = spectrogram(
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
        self.cbar.setLevels((-35, 0))
