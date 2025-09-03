from __future__ import annotations

# ruff: noqa: N802 Function name should be lowercase
import argparse
import json
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from trailcam_classifier.util import find_images


class ViewerWindow(QMainWindow):
    def __init__(self, directory: str):
        super().__init__()
        self.setWindowTitle("Trailcam Classifier Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.image_paths = sorted(find_images([directory]))
        self.current_image_index = 0

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.load_image()

    def load_image(self):
        if not self.image_paths:
            return

        image_path = self.image_paths[self.current_image_index]
        pixmap = QPixmap(str(image_path))

        # Load metadata and draw bounding boxes
        json_path = image_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path) as f:
                metadata = json.load(f)

            painter = QPainter(pixmap)
            for bboxes in metadata.values():
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.end()

        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Right:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
            self.load_image()
        elif event.key() == Qt.Key_Left:
            self.current_image_index = (self.current_image_index - 1 + len(self.image_paths)) % len(self.image_paths)
            self.load_image()

    def resizeEvent(self, event):
        del event
        self.load_image()


def main():
    parser = argparse.ArgumentParser(description="Image viewer for trailcam classifier.")
    parser.add_argument("directory", nargs="?", default=None, help="Directory containing images and metadata.")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    if args.directory:
        directory = args.directory
    else:
        directory = QFileDialog.getExistingDirectory(None, "Select Directory")
        if not directory:
            return

    window = ViewerWindow(directory)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
