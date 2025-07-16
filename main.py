import sys
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
from hfovis.data import ieeg_loader, streaming
from hfovis.gui import MainWindow
from hfovis.detector import RealTimeDetector

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

    streamer = streaming.DataStreamer(data, chunk_size=fs, fs=fs)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    main = MainWindow(fs, channel_names)

    detector = RealTimeDetector(streamer, main.handle, fs=fs, channels=data.shape[1])
    app.aboutToQuit.connect(detector.stop)
    detector.start()

    main.show()
    sys.exit(app.exec())
