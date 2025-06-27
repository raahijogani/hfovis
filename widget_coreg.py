import sys
import numpy as np
import nibabel as nib

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QComboBox, QPushButton, QFileDialog
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import pyvista as pv  # pip install pyvista


class SliceViewer(QWidget):
    def __init__(self, mri_data, ct_data):
        super().__init__()

        # store volumes
        self.mri_data = mri_data
        self.ct_data  = ct_data

        # compute window-level percentiles
        self.ct_min,  self.ct_max  = np.percentile(ct_data, (1, 99))
        self.mri_min, self.mri_max = np.percentile(mri_data, (1, 99))

        # defaults
        self.orientation   = "Axial"
        self.current_slice = mri_data.shape[2] // 2
        self.blend_ratio   = 0.6

        self.initUI()

    def initUI(self):
        self.setWindowTitle("MRI–CT Viewer with Blend & 3D")
        layout = QVBoxLayout()

        # 2D Matplotlib canvas
        self.canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.ax     = self.canvas.figure.add_subplot(111)
        self.ax.axis("off")
        layout.addWidget(self.canvas)

        # Controls row
        ctrls = QHBoxLayout()

        # Load buttons
        self.load_mri_btn = QPushButton("Load MRI")
        self.load_mri_btn.clicked.connect(self.load_mri)
        ctrls.addWidget(self.load_mri_btn)

        self.load_ct_btn = QPushButton("Load CT")
        self.load_ct_btn.clicked.connect(self.load_ct)
        ctrls.addWidget(self.load_ct_btn)

        # Slice slider
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self.mri_data.shape[2] - 1)
        self.slice_slider.setValue(self.current_slice)
        self.slice_slider.valueChanged.connect(self.update_slice)
        ctrls.addWidget(QLabel("Slice:"))
        ctrls.addWidget(self.slice_slider)

        # Orientation selector
        self.orientation_selector = QComboBox()
        self.orientation_selector.addItems(["Axial", "Coronal", "Sagittal"])
        self.orientation_selector.currentTextChanged.connect(self.change_orientation)
        ctrls.addWidget(QLabel("Orientation:"))
        ctrls.addWidget(self.orientation_selector)

        # Blend slider
        self.blend_slider = QSlider(Qt.Orientation.Horizontal)
        self.blend_slider.setMinimum(0)
        self.blend_slider.setMaximum(100)
        self.blend_slider.setValue(int(self.blend_ratio * 100))
        self.blend_slider.valueChanged.connect(self.update_blend)
        ctrls.addWidget(QLabel("MRI %:"))
        ctrls.addWidget(self.blend_slider)

        # Export & 3D buttons
        self.export_btn   = QPushButton("Export Screenshot")
        self.export_btn.clicked.connect(self.save_screenshot)
        ctrls.addWidget(self.export_btn)

        self.viewer3d_btn = QPushButton("Open 3D Viewer")
        self.viewer3d_btn.clicked.connect(self.open_3d_viewer)
        ctrls.addWidget(self.viewer3d_btn)

        layout.addLayout(ctrls)

        # Slice info label
        self.slice_label = QLabel()
        layout.addWidget(self.slice_label)

        self.setLayout(layout)

        # First draw
        self.update_slice(self.current_slice)

    def load_mri(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select MRI file", "", "NIfTI (*.nii *.nii.gz)"
        )
        if not path:
            return

        img = nib.load(path)
        self.mri_data = img.get_fdata()
        self.mri_min, self.mri_max = np.percentile(self.mri_data, (1, 99))

        # update slider range for current orientation
        dims = {
            "Axial":    self.mri_data.shape[2],
            "Coronal":  self.mri_data.shape[1],
            "Sagittal": self.mri_data.shape[0],
        }[self.orientation]
        self.slice_slider.setMaximum(dims - 1)

        # redraw (will show a warning if shapes mismatch)
        self.update_slice(self.current_slice)

    def load_ct(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CT file", "", "NIfTI (*.nii *.nii.gz)"
        )
        if not path:
            return

        img = nib.load(path)
        self.ct_data = img.get_fdata()
        self.ct_min, self.ct_max = np.percentile(self.ct_data, (1, 99))

        # now shapes should match—reinitialize slice & redraw
        self.change_orientation(self.orientation)

    def get_slice_data(self, vol, orient, idx):
        if orient == "Axial":
            return vol[:, :, idx]
        if orient == "Coronal":
            return vol[:, idx, :]
        return vol[idx, :, :]

    def update_slice(self, idx):
        # Guard against mismatched volumes
        if self.mri_data.shape != self.ct_data.shape:
            self.ax.clear()
            self.ax.text(
                0.5, 0.5,
                "MRI/CT shape mismatch\nPlease load matching volumes",
                ha="center", va="center", wrap=True
            )
            self.ax.axis("off")
            self.canvas.draw()
            return

        # Normal operation
        self.current_slice = idx
        self.slice_label.setText(f"{self.orientation} slice {idx}")

        mri_slice = self.get_slice_data(self.mri_data, self.orientation, idx)
        ct_slice  = self.get_slice_data(self.ct_data,  self.orientation, idx)

        # window-level normalize
        mri_norm = np.clip((mri_slice - self.mri_min)/(self.mri_max - self.mri_min), 0, 1)
        ct_norm  = np.clip((ct_slice  - self.ct_min)/(self.ct_max  - self.ct_min), 0, 1)

        # blend
        blended = np.clip(
            mri_norm * self.blend_ratio +
            ct_norm  * (1 - self.blend_ratio),
            0, 1
        )

        self.ax.clear()
        self.ax.imshow(blended.T, cmap="gray", origin="lower")
        self.ax.set_title(f"Blended ({int(self.blend_ratio*100)}% MRI)")
        self.ax.axis("off")
        self.canvas.draw()

    def change_orientation(self, orient):
        self.orientation = orient
        dims = {
            "Axial":    self.mri_data.shape[2],
            "Coronal":  self.mri_data.shape[1],
            "Sagittal": self.mri_data.shape[0],
        }[orient]
        self.slice_slider.setMaximum(dims - 1)
        mid = dims // 2
        self.slice_slider.setValue(mid)
        self.update_slice(mid)

    def update_blend(self, val):
        self.blend_ratio = val / 100.0
        self.update_slice(self.current_slice)

    def save_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "", "PNG Files (*.png)")
        if path:
            self.canvas.figure.savefig(path, dpi=300)
            print(f"Saved to {path}")

    def open_3d_viewer(self):
        # same guard applies—only proceed if shapes match
        if self.mri_data.shape != self.ct_data.shape:
            return

        # full‐volume normalize + blend
        mri_vol = np.clip((self.mri_data - self.mri_min)/(self.mri_max - self.mri_min), 0, 1)
        ct_vol  = np.clip((self.ct_data  - self.ct_min)/(self.ct_max  - self.ct_min), 0, 1)
        vol = mri_vol*self.blend_ratio + ct_vol*(1 - self.blend_ratio)

        grid = pv.ImageData(dimensions=vol.shape, spacing=(1,1,1), origin=(0,0,0))
        grid["values"] = vol.flatten(order="F")

        plotter = pv.Plotter()
        plotter.add_volume(grid, scalars="values", cmap="gray", opacity="linear")
        plotter.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 1) Ask for both files up front
    mri_path, _ = QFileDialog.getOpenFileName(
        None, "Select MRI file", "", "NIfTI (*.nii *.nii.gz)"
    )
    ct_path, _  = QFileDialog.getOpenFileName(
        None, "Select CT file", "", "NIfTI (*.nii *.nii.gz)"
    )
    if not mri_path or not ct_path:
        sys.exit("Must select both MRI and CT to continue.")

    # 2) Load volumes
    mri_data = nib.load(mri_path).get_fdata()
    ct_data  = nib.load(ct_path).get_fdata()

    # 3) Launch viewer
    viewer = SliceViewer(mri_data, ct_data)
    viewer.show()
    sys.exit(app.exec())
