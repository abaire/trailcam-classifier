from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from trailcam_classifier.training import save_class_names, train_model
from trailcam_classifier.util import MODEL_SAVE_FILENAME


@pytest.fixture
def dataset_yaml(tmp_path: Path) -> Path:
    yaml_path = tmp_path / "dataset.yaml"
    data = {
        "names": {
            0: "coyote",
            1: "deer",
            2: "fox",
        }
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    return yaml_path


def test_save_class_names(dataset_yaml: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    class_names = save_class_names(dataset_yaml, output_dir)
    assert class_names == ["coyote", "deer", "fox"]

    class_names_file = output_dir / "class_names.txt"
    assert class_names_file.exists()
    assert class_names_file.read_text().strip().splitlines() == ["coyote", "deer", "fox"]


@patch("trailcam_classifier.training.YOLO")
def test_train_model_callback_saves_best_checkpoint(
    mock_yolo_cls: MagicMock, dataset_yaml: Path, tmp_path: Path
) -> None:
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    output_dir = tmp_path / "output"
    weights_dir = tmp_path / "runs" / "weights"
    weights_dir.mkdir(parents=True)
    best_pt = weights_dir / "best.pt"
    best_pt.write_text("model_v1")

    callbacks: dict[str, list[Any]] = {}

    def add_callback(event: str, func: Any) -> None:
        callbacks.setdefault(event, []).append(func)

    mock_model.add_callback.side_effect = add_callback

    mock_results = MagicMock()
    mock_results.save_dir = str(tmp_path / "runs")
    mock_model.train.return_value = mock_results

    train_model(dataset=dataset_yaml, output_dir=output_dir, epochs=10, batch=8)

    assert "on_fit_epoch_end" in callbacks
    on_fit_epoch_end = callbacks["on_fit_epoch_end"][0]

    dest_model = output_dir / MODEL_SAVE_FILENAME
    assert (output_dir / "class_names.txt").exists()

    # Simulate epoch 1 with best fitness 0.8
    trainer_mock = MagicMock()
    trainer_mock.epoch = 0
    trainer_mock.epochs = 10
    trainer_mock.metrics = {"metrics/mAP50(B)": 0.8}
    trainer_mock.best_fitness = 0.8
    trainer_mock.best = best_pt
    trainer_mock.save_dir = str(tmp_path / "runs")

    on_fit_epoch_end(trainer_mock)
    assert dest_model.exists()
    assert dest_model.read_text() == "model_v1"

    # Simulate epoch 2 with worse fitness 0.75 - should not rewrite
    best_pt.write_text("model_v2_not_best")
    trainer_mock.epoch = 1
    trainer_mock.metrics = {"metrics/mAP50(B)": 0.75}
    trainer_mock.best_fitness = 0.8  # unchanged

    on_fit_epoch_end(trainer_mock)
    assert dest_model.read_text() == "model_v1"

    # Simulate epoch 3 with improved fitness 0.85
    best_pt.write_text("model_v3_best")
    trainer_mock.epoch = 2
    trainer_mock.metrics = {"metrics/mAP50(B)": 0.85}
    trainer_mock.best_fitness = 0.85

    on_fit_epoch_end(trainer_mock)
    assert dest_model.read_text() == "model_v3_best"


@patch("trailcam_classifier.training.YOLO")
def test_train_model_handles_keyboard_interrupt(mock_yolo_cls: MagicMock, dataset_yaml: Path, tmp_path: Path) -> None:
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    output_dir = tmp_path / "output"
    weights_dir = tmp_path / "runs" / "weights"
    weights_dir.mkdir(parents=True)
    best_pt = weights_dir / "best.pt"
    best_pt.write_text("epoch_2_best_weights")

    callbacks: dict[str, list[Any]] = {}

    def add_callback(event: str, func: Any) -> None:
        callbacks.setdefault(event, []).append(func)

    mock_model.add_callback.side_effect = add_callback

    def fake_train(*_args: Any, **_kwargs: Any) -> None:
        # Simulate on_fit_epoch_end happening during training
        on_fit_epoch_end = callbacks["on_fit_epoch_end"][0]
        trainer_mock = MagicMock()
        trainer_mock.epoch = 1
        trainer_mock.epochs = 10
        trainer_mock.metrics = {"fitness": 0.9}
        trainer_mock.best_fitness = 0.9
        trainer_mock.best = best_pt
        trainer_mock.save_dir = str(tmp_path / "runs")
        on_fit_epoch_end(trainer_mock)

        # User interrupts before training finishes
        raise KeyboardInterrupt

    mock_model.train.side_effect = fake_train

    train_model(dataset=dataset_yaml, output_dir=output_dir, epochs=10)

    dest_model = output_dir / MODEL_SAVE_FILENAME
    assert dest_model.exists()
    assert dest_model.read_text() == "epoch_2_best_weights"
    assert (output_dir / "class_names.txt").exists()
