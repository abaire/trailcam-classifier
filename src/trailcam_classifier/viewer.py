from __future__ import annotations

# ruff: noqa: N802 Function name should be lowercase
import argparse
import json
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPixmap
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
        json_path = image_path.with_suffix(".json")

        if json_path.exists():
            with open(json_path) as f:
                metadata = json.load(f)

            painter = QPainter(pixmap)
            font = QFont("Helvetica", 16)
            painter.setFont(font)

            color_map = {
                "animal": QColor("green"),
                "person": QColor("red"),
                "vehicle": QColor("blue"),
            }
            default_color = QColor("yellow")

            for label, bboxes in metadata.items():
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                    confidence = bbox.get("confidence", 0.0)

                    color = color_map.get(label, default_color)

                    pen = painter.pen()
                    pen.setColor(QColor("black"))
                    pen.setWidth(4)
                    painter.setPen(pen)
                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)

                    pen.setColor(color)
                    pen.setWidth(2)
                    painter.setPen(pen)
                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)

                    label_text = f"{label}: {confidence:.2f}"
                    metrics = painter.fontMetrics()
                    text_width = metrics.horizontalAdvance(label_text)
                    text_height = metrics.height()

                    text_x = x1
                    text_y = y1 - 5

                    painter.fillRect(text_x, text_y - text_height, text_width, text_height, QColor("black"))

                    text_x = x1
                    text_y = y1 - 5

                    painter.fillRect(text_x, text_y - text_height, text_width, text_height, QColor("black"))

                    pen.setColor(color)
                    painter.setPen(pen)
                    painter.drawText(text_x, text_y, label_text)

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
