from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from trailcam_classifier.training import main, save_class_names, train_model
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
    last_pt = weights_dir / "last.pt"
    last_pt.write_text("epoch_2_last_weights")

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
        trainer_mock.last = last_pt
        trainer_mock.save_dir = str(tmp_path / "runs")
        on_fit_epoch_end(trainer_mock)

        # User interrupts before training finishes
        raise KeyboardInterrupt

    mock_model.train.side_effect = fake_train

    train_model(dataset=dataset_yaml, output_dir=output_dir, epochs=10)

    dest_model = output_dir / MODEL_SAVE_FILENAME
    last_model = output_dir / "last.pt"
    assert dest_model.exists()
    assert dest_model.read_text() == "epoch_2_best_weights"
    assert last_model.exists()
    assert (output_dir / "class_names.txt").exists()


@patch("trailcam_classifier.training.YOLO")
def test_train_model_resume_default(mock_yolo_cls: MagicMock, dataset_yaml: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    last_pt = output_dir / "last.pt"
    last_pt.write_text("saved_checkpoint_state")

    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_results = MagicMock()
    mock_results.save_dir = str(tmp_path / "runs")
    mock_model.train.return_value = mock_results

    train_model(dataset=dataset_yaml, output_dir=output_dir, epochs=200, resume=True)

    mock_yolo_cls.assert_called_once_with(str(last_pt))
    mock_model.train.assert_called_once_with(
        data=str(dataset_yaml),
        epochs=200,
        batch=16,
        imgsz=1024,
        resume=True,
    )


@patch("trailcam_classifier.training.YOLO")
def test_train_model_resume_custom_path(mock_yolo_cls: MagicMock, dataset_yaml: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    custom_checkpoint = tmp_path / "custom" / "checkpoint_epoch_50.pt"
    custom_checkpoint.parent.mkdir(parents=True)
    custom_checkpoint.write_text("custom_checkpoint_state")

    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_results = MagicMock()
    mock_results.save_dir = str(tmp_path / "runs")
    mock_model.train.return_value = mock_results

    train_model(dataset=dataset_yaml, output_dir=output_dir, epochs=200, resume=custom_checkpoint)

    mock_yolo_cls.assert_called_once_with(str(custom_checkpoint))
    mock_model.train.assert_called_once_with(
        data=str(dataset_yaml),
        epochs=200,
        batch=16,
        imgsz=1024,
        resume=True,
    )


def test_train_model_resume_not_found(dataset_yaml: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    with pytest.raises(FileNotFoundError, match="Checkpoint to resume from not found"):
        train_model(dataset=dataset_yaml, output_dir=output_dir, epochs=200, resume=True)


@patch("trailcam_classifier.training.YOLO")
def test_train_model_custom_batch(mock_yolo_cls: MagicMock, dataset_yaml: Path, tmp_path: Path) -> None:
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model
    output_dir = tmp_path / "output"

    train_model(dataset=dataset_yaml, output_dir=output_dir, epochs=5, batch=32)

    mock_model.train.assert_called_once_with(
        data=str(dataset_yaml),
        epochs=5,
        batch=32,
        imgsz=1024,
        resume=False,
    )


@patch("trailcam_classifier.training.train_model")
def test_main_batch_argument(mock_train_model: MagicMock, dataset_yaml: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    with patch(
        "sys.argv",
        [
            "train",
            str(dataset_yaml),
            str(output_dir),
            "--batch",
            "32",
        ],
    ):
        main()

    mock_train_model.assert_called_once_with(
        dataset=str(dataset_yaml),
        output_dir=str(output_dir),
        epochs=175,
        batch=32,
        model_name="yolo26x.pt",
        resume=None,
    )
