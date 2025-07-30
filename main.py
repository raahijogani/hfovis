import sys

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication

from hfovis.data import ieeg_loader, streaming
from hfovis.gui import MainWindow

if __name__ == "__main__":
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, useOpenGL=True)

    # ------------------------------------------------------------------
    file_info = {
        "Filename_interictal": "demo_data.mat",
        "subject": "sub-01",
        "outcome": "seizure-free",
    }
    data_path = "data"
    annotations_path = "annotations.json"

    print("Loading data…")
    data, fs, channel_names, _ = ieeg_loader.load_ieeg_from_fileinfo(
        file_info, data_path, annotations_path, baseline="interictal"
    )

    streamer = streaming.DataStreamer(data, chunk_size=fs, interval_s=1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    main = MainWindow(streamer, channel_names, fs=fs)

    main.show()
    sys.exit(app.exec())
