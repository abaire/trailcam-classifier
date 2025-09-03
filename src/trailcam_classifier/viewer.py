from __future__ import annotations

# ruff: noqa: N802 Function name should be lowercase
import argparse
import json
import sys

from PySide6.QtCore import QRect, QSize, Qt
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


class AnnotationLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metadata = None
        self.class_colors = {}

    def set_metadata(self, metadata, class_colors):
        self.metadata = metadata
        self.class_colors = class_colors
        self.update()

    def sizeHint(self):
        return QSize(100, 100)

    def minimumSizeHint(self):
        return QSize(10, 10)

    def paintEvent(self, _event):
        if not self.pixmap() or self.pixmap().isNull():
            return

        painter = QPainter(self)
        cr = self.contentsRect()
        pm = self.pixmap()

        pm_size = pm.size()
        pm_size.scale(cr.size(), Qt.KeepAspectRatio)

        target_rect = QRect(0, 0, pm_size.width(), pm_size.height())
        target_rect.moveCenter(cr.center())

        painter.drawPixmap(target_rect, pm)

        if not self.metadata:
            return

        font = QFont("Helvetica")
        font.setPointSize(16)
        painter.setFont(font)

        original_pixmap_size = pm.size()
        scale = min(
            target_rect.width() / original_pixmap_size.width(),
            target_rect.height() / original_pixmap_size.height(),
        )

        offset_x = target_rect.x()
        offset_y = target_rect.y()

        for label, bboxes in self.metadata.items():
            color = self.class_colors.get(label, QColor("yellow"))
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

                scaled_x1 = x1 * scale + offset_x
                scaled_y1 = y1 * scale + offset_y
                scaled_w = (x2 - x1) * scale
                scaled_h = (y2 - y1) * scale

                confidence = bbox.get("confidence", 0.0)

                pen = painter.pen()
                pen.setColor(QColor("black"))
                pen.setWidth(4)
                painter.setPen(pen)
                painter.drawRect(int(scaled_x1), int(scaled_y1), int(scaled_w), int(scaled_h))

                pen.setColor(color)
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawRect(int(scaled_x1), int(scaled_y1), int(scaled_w), int(scaled_h))

                label_text = f"{label}: {confidence:.2f}"
                metrics = painter.fontMetrics()
                text_width = metrics.horizontalAdvance(label_text)
                text_height = metrics.height()

                text_x = scaled_x1
                if text_x + text_width > self.width():
                    text_x = self.width() - text_width
                if text_x < 0:
                    text_x = 0

                text_y = scaled_y1 - 5
                if text_y - text_height < 0:
                    text_y = scaled_y1 + text_height

                painter.fillRect(
                    int(text_x),
                    int(text_y - text_height),
                    int(text_width),
                    int(text_height),
                    QColor("black"),
                )

                pen.setColor(color)
                painter.setPen(pen)
                painter.drawText(int(text_x), int(text_y), label_text)


class ViewerWindow(QMainWindow):
    GOLDEN_RATIO_CONJUGATE = 0.618033988749895

    def __init__(self, directory: str):
        super().__init__()
        self.setWindowTitle("Trailcam Classifier Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.image_paths = sorted(find_images([directory]))
        self.current_image_index = 0
        self.image_label = AnnotationLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.class_colors = {}
        self.next_color_hue = 0.0
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.load_image()

    def load_image(self):
        if not self.image_paths:
            self.image_label.setPixmap(QPixmap())
            self.image_label.set_metadata(None, {})
            self.setWindowTitle("Trailcam Classifier Viewer")
            return

        image_path = self.image_paths[self.current_image_index]
        pixmap = QPixmap(str(image_path))

        filename = image_path.name
        self.setWindowTitle(f"Trailcam Classifier Viewer - {filename} ({pixmap.width()}x{pixmap.height()})")

        metadata = None
        json_path = image_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path) as f:
                metadata = json.load(f)
            for label in metadata:
                if label not in self.class_colors:
                    self.next_color_hue += self.GOLDEN_RATIO_CONJUGATE
                    self.next_color_hue %= 1
                    self.class_colors[label] = QColor.fromHsvF(self.next_color_hue, 0.7, 0.95)

        self.image_label.setPixmap(pixmap)
        self.image_label.set_metadata(metadata, self.class_colors)

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
