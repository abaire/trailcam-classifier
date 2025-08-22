from __future__ import annotations

# ruff: noqa: N802 Function name should be lowercase
import asyncio
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from trailcam_classifier.main import ClassificationConfig, run_classification


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = QSettings()

        layout = QVBoxLayout(self)

        # Output directory
        output_dir_layout = QHBoxLayout()
        output_dir_label = QLabel("Output Directory:")
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(self.settings.value("output_directory", "classified_output"))
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_output_directory)
        output_dir_layout.addWidget(output_dir_label)
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(browse_button)
        layout.addLayout(output_dir_layout)

        # Save and Cancel buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def browse_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)

    def accept(self):
        self.settings.setValue("output_directory", self.output_dir_edit.text())
        super().accept()


class Worker(QObject):
    """A worker object that runs the classification in a separate thread."""

    log_message = Signal(str)
    finished = Signal()

    def __init__(self, config: ClassificationConfig):
        super().__init__()
        self.config = config

    def run(self):
        """Runs the classification."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_classification(self.config, logger=self.log_message.emit))
        finally:
            loop.close()
        self.finished.emit()


class DropLabel(QLabel):
    def __init__(self, text, parent: MainWindow):
        super().__init__(text, parent)
        self.parent_widget = parent
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            """
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                padding: 20px;
                font-size: 16px;
            }
        """
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return

        # For now, we only support dropping a single folder.
        url = urls[0]
        if url.isLocalFile():
            folder_path = url.toLocalFile()
            if Path(folder_path).is_dir():
                self.parent_widget.start_classification(folder_path)
            else:
                self.parent_widget.log("Please drop a folder, not a file.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trailcam Classifier")
        self.setGeometry(100, 100, 600, 400)

        self.settings = QSettings()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.drop_label = DropLabel("Drop a folder here", self)
        layout.addWidget(self.drop_label)

        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        layout.addWidget(self.log_widget)

        self.thread = None
        self.worker = None

        self._create_menus()

    def _create_menus(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        settings_action = file_menu.addAction("&Settings")
        settings_action.triggered.connect(self.open_settings)
        exit_action = file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)

    def open_settings(self):
        dialog = SettingsDialog(self)
        dialog.exec()

    def log(self, message: str):
        self.log_widget.append(message)

    def start_classification(self, folder_path: str):
        if self.thread and self.thread.isRunning():
            self.log("A classification process is already running.")
            return

        self.log_widget.clear()
        self.log(f"Starting classification for folder: {folder_path}")

        output_directory = self.settings.value("output_directory", "classified_output")

        config = ClassificationConfig(dirs=[folder_path], output=output_directory)
        self.thread = QThread()
        self.worker = Worker(config)
        self.worker.moveToThread(self.thread)

        self.worker.log_message.connect(self.log)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()


def run_gui():
    QApplication.setOrganizationName("BearBrains")
    QApplication.setApplicationName("trailcam-classifier")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
