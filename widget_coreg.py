import sys
import numpy as np
import pandas as pd
import nibabel as nib
import tabula  # for reading PDF tables; install with `pip install tabula-py`

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QComboBox, QPushButton, QFileDialog
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import pyvista as pv  # pip install pyvista


class SliceViewer(QWidget):
    def __init__(self, mri_img, ct_img):
        super().__init__()

        # load volumes + keep the MRI affine for world→voxel
        self.mri_data   = mri_img.get_fdata()
        self.mri_affine = mri_img.affine
        self.ct_data    = ct_img.get_fdata()

        # compute 1–99th percentile windows
        self.ct_min,  self.ct_max  = np.percentile(self.ct_data, (1, 99))
        self.mri_min, self.mri_max = np.percentile(self.mri_data, (1, 99))

        # defaults
        self.orientation    = "Axial"
        self.current_slice  = self.mri_data.shape[2] // 2
        self.blend_ratio    = 0.6

        # for event‐heatmap overlay
        self.events_df = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle("MRI–CT Viewer + Event Heatmap")
        layout = QVBoxLayout()

        # 2D Matplotlib canvas
        self.canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.ax     = self.canvas.figure.add_subplot(111)
        self.ax.axis("off")
        layout.addWidget(self.canvas)

        # Controls row
        ctrls = QHBoxLayout()

        # Load MRI / CT
        self.load_mri_btn = QPushButton("Load MRI")
        self.load_mri_btn.clicked.connect(self.load_mri)
        ctrls.addWidget(self.load_mri_btn)

        self.load_ct_btn = QPushButton("Load CT")
        self.load_ct_btn.clicked.connect(self.load_ct)
        ctrls.addWidget(self.load_ct_btn)

        # Load events file
        self.load_events_btn = QPushButton("Load Events")
        self.load_events_btn.clicked.connect(self.load_events)
        ctrls.addWidget(self.load_events_btn)

        # Channel selector
        ctrls.addWidget(QLabel("Channel:"))
        self.channel_selector = QComboBox()
        ctrls.addWidget(self.channel_selector)

        # Slice slider
        ctrls.addWidget(QLabel("Slice:"))
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self.mri_data.shape[2] - 1)
        self.slice_slider.setValue(self.current_slice)
        self.slice_slider.valueChanged.connect(self.update_slice)
        ctrls.addWidget(self.slice_slider)

        # Orientation dropdown
        ctrls.addWidget(QLabel("Orientation:"))
        self.orientation_selector = QComboBox()
        self.orientation_selector.addItems(["Axial", "Coronal", "Sagittal"])
        self.orientation_selector.currentTextChanged.connect(self.change_orientation)
        ctrls.addWidget(self.orientation_selector)

        # Blend slider
        ctrls.addWidget(QLabel("MRI %:"))
        self.blend_slider = QSlider(Qt.Orientation.Horizontal)
        self.blend_slider.setRange(0, 100)
        self.blend_slider.setValue(int(self.blend_ratio * 100))
        self.blend_slider.valueChanged.connect(self.update_blend)
        ctrls.addWidget(self.blend_slider)

        # Export screenshot & 3D viewer
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
        self.update_slice(self.current_slice)

    def load_mri(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select MRI file", "", "NIfTI (*.nii *.nii.gz)")
        if not path:
            return
        img = nib.load(path)
        self.mri_data   = img.get_fdata()
        self.mri_affine = img.affine
        self.mri_min, self.mri_max = np.percentile(self.mri_data, (1, 99))

        dims = {
            "Axial":    self.mri_data.shape[2],
            "Coronal":  self.mri_data.shape[1],
            "Sagittal": self.mri_data.shape[0],
        }[self.orientation]
        self.slice_slider.setMaximum(dims - 1)
        self.update_slice(self.current_slice)

    def load_ct(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CT file", "", "NIfTI (*.nii *.nii.gz)")
        if not path:
            return
        img = nib.load(path)
        self.ct_data     = img.get_fdata()
        self.ct_min, self.ct_max = np.percentile(self.ct_data, (1, 99))
        self.change_orientation(self.orientation)

    def load_events(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select events file", "", "Data (*.csv *.txt *.pdf)")
        if not path:
            return

        # Read table from PDF or CSV/TXT
        if path.lower().endswith(".pdf"):
            tables = tabula.read_pdf(path, pages="all", guess=False)
            df = tables[0]
        else:
            df = pd.read_csv(path)

        df = df.rename(columns=str.strip)
        needed = ["index", "channel", "x", "y", "z", "event_count"]
        if not all(col in df.columns for col in needed):
            raise ValueError(f"Events file needs columns: {needed}")

        self.events_df = df[needed]
        channels = sorted(self.events_df["channel"].unique())
        self.channel_selector.clear()
        self.channel_selector.addItems([str(c) for c in channels])

    def get_slice_data(self, vol, orient, idx):
        if orient == "Axial":    return vol[:, :, idx]
        if orient == "Coronal":  return vol[:, idx, :]
        return vol[idx, :, :]

    def update_slice(self, idx):
        if self.mri_data.shape != self.ct_data.shape:
            self.ax.clear()
            self.ax.text(0.5, 0.5,
                         "MRI/CT shape mismatch\nLoad matching volumes",
                         ha="center", va="center", wrap=True)
            self.ax.axis("off")
            self.canvas.draw()
            return

        self.current_slice = idx
        self.slice_label.setText(f"{self.orientation} slice {idx}")

        m = self.get_slice_data(self.mri_data, self.orientation, idx)
        c = self.get_slice_data(self.ct_data,  self.orientation, idx)
        m_norm = np.clip((m - self.mri_min)/(self.mri_max - self.mri_min), 0,1)
        c_norm = np.clip((c - self.ct_min )/(self.ct_max  - self.ct_min ), 0,1)
        blended = np.clip(m_norm*self.blend_ratio + c_norm*(1-self.blend_ratio), 0,1)

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

    def open_3d_viewer(self):
        # guard: only if shapes match
        if self.mri_data.shape != self.ct_data.shape:
            return

        # 1) Build the blended volume exactly as before
        m_vol = np.clip((self.mri_data - self.mri_min)/(self.mri_max - self.mri_min), 0,1)
        c_vol = np.clip((self.ct_data  - self.ct_min )/(self.ct_max  - self.ct_min ), 0,1)
        vol   = m_vol*self.blend_ratio + c_vol*(1 - self.blend_ratio)

        grid = pv.ImageData(
            dimensions=vol.shape,
            spacing=(1,1,1),
            origin=(0,0,0),
        )
        grid["values"] = vol.flatten(order="F")

        # 2) Render volume with NO scalar bar
        plotter = pv.Plotter()
        plotter.add_volume(
            grid,
            scalars="values",
            cmap="gray",
            opacity="linear",
            show_scalar_bar=False
        )

        # 3) Overlay your events (if loaded), using raw voxel coords
        if self.events_df is not None:
            chan = self.channel_selector.currentText()
            df   = self.events_df[self.events_df["channel"] == chan]

            # direct voxel indices from your CSV
            pts = df[["x","y","z"]].to_numpy()
            mesh = pv.PolyData(pts)
            mesh["event_count"] = df["event_count"].to_numpy()

            # add only *one* colorbar here
            plotter.add_mesh(
                mesh,
                scalars="event_count",
                cmap="hot",
                point_size=6,
                render_points_as_spheres=True,
                scalar_bar_args={
                    "title": "Event Count",
                    "vertical": False,
                    "position_x": 0.3,
                    "position_y": 0.05,
                }
            )

        plotter.show()



if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Prompt for MRI & CT before launching
    mri_path, _ = QFileDialog.getOpenFileName(None, "Select MRI file", "", "NIfTI (*.nii *.nii.gz)")
    ct_path,  _ = QFileDialog.getOpenFileName(None, "Select CT file",  "", "NIfTI (*.nii *.nii.gz)")
    if not mri_path or not ct_path:
        sys.exit("Must select both MRI and CT to continue.")

    # Load nibabel images
    mri_img = nib.load(mri_path)
    ct_img  = nib.load(ct_path)

    viewer = SliceViewer(mri_img, ct_img)
    viewer.show()
    sys.exit(app.exec())
