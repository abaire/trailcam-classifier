from __future__ import annotations

# ruff: noqa: T201 `print` found
import argparse
import shutil
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from trailcam_classifier.util import MODEL_SAVE_FILENAME

# _DEFAULT_MODEL = "yolo11m.pt"
# _DEFAULT_MODEL = "yolo26l.pt"
_DEFAULT_MODEL = "yolo26x.pt"


def save_class_names(dataset_yaml_path: Path | str, output_dir: Path | str) -> list[str]:
    output_dir = Path(output_dir)
    with open(dataset_yaml_path) as f:
        data_yaml = yaml.safe_load(f)
    class_names = list(data_yaml["names"].values())
    class_names_path = output_dir / "class_names.txt"
    with open(class_names_path, "w") as f:
        f.write("\n".join(class_names))
    print(f"Class names saved to {class_names_path}")
    return class_names


def train_model(
    dataset: Path | str,
    output_dir: Path | str,
    epochs: int = 175,
    batch: int = 16,
    model_name: str = _DEFAULT_MODEL,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dest_path = output_path / MODEL_SAVE_FILENAME

    save_class_names(dataset, output_path)

    model = YOLO(model_name)

    best_state: dict[str, Any] = {"fitness": None, "epoch": None}

    def on_fit_epoch_end(trainer: Any) -> None:
        current_epoch = trainer.epoch + 1  # trainer.epoch is 0-indexed
        total_epochs = trainer.epochs
        metrics = trainer.metrics or {}

        if trainer.best_fitness is not None and trainer.best_fitness != best_state["fitness"]:
            best_state["fitness"] = trainer.best_fitness
            best_state["epoch"] = current_epoch

            best_model_path = getattr(trainer, "best", None) or (Path(trainer.save_dir) / "weights" / "best.pt")
            best_model_path = Path(best_model_path)
            if best_model_path.exists():
                shutil.copy(best_model_path, dest_path)

        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
        fitness_str = (
            f"{best_state['fitness']:.6f}"
            if isinstance(best_state["fitness"], (int, float))
            else str(best_state["fitness"])
        )
        print(
            f"\n[Epoch {current_epoch}/{total_epochs}]  {metric_str}"
            f"\n  Best so far: epoch {best_state['epoch']}  fitness={fitness_str}"
        )

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    try:
        results = model.train(data=str(dataset), epochs=epochs, batch=batch, imgsz=1024)
        if results and getattr(results, "save_dir", None):
            best_model_path = Path(results.save_dir) / "weights" / "best.pt"
            if best_model_path.exists():
                shutil.copy(best_model_path, dest_path)
        print(f"Best model saved to {dest_path}")
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user. Best model saved so far is at {dest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO model.")
    parser.add_argument("dataset", help="Path to the dataset.yaml file.")
    parser.add_argument("output_dir", help="Directory into which the final model should be saved.")

    parser.add_argument("--epochs", type=int, default=175, help="Number of epochs to train for.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")

    parser.add_argument("--model", default=_DEFAULT_MODEL, help="The YOLO model to use.")

    args = parser.parse_args()

    train_model(
        dataset=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch=args.batch,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
