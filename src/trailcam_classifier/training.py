from __future__ import annotations

# ruff: noqa: T201 `print` found
import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

from trailcam_classifier.util import MODEL_SAVE_FILENAME


# _DEFAULT_MODEL = "yolo11m.pt"
#_DEFAULT_MODEL = "yolo26l.pt"
_DEFAULT_MODEL = "yolo26x.pt"

def main():
    parser = argparse.ArgumentParser(description="Train a YOLO model.")
    parser.add_argument("dataset", help="Path to the dataset.yaml file.")
    parser.add_argument("output_dir", help="Directory into which the final model should be saved.")

    parser.add_argument("--epochs", type=int, default=175, help="Number of epochs to train for.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")

    parser.add_argument("--model", default=_DEFAULT_MODEL, help="The YOLO model to use.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    best_state: dict = {"fitness": None, "epoch": None}

    def on_fit_epoch_end(trainer) -> None:
        current_epoch = trainer.epoch + 1  # trainer.epoch is 0-indexed
        total_epochs = trainer.epochs
        metrics = trainer.metrics or {}

        if trainer.best_fitness is not None and trainer.best_fitness != best_state["fitness"]:
            best_state["fitness"] = trainer.best_fitness
            best_state["epoch"] = current_epoch

        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
        print(
            f"\n[Epoch {current_epoch}/{total_epochs}]  {metric_str}"
            f"\n  Best so far: epoch {best_state['epoch']}  fitness={best_state['fitness']:.6f}"
        )

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    results = model.train(data=args.dataset, epochs=args.epochs, imgsz=1024)

    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    dest_path = output_dir / MODEL_SAVE_FILENAME
    shutil.copy(best_model_path, dest_path)
    print(f"Best model saved to {dest_path}")

    with open(args.dataset) as f:
        data_yaml = yaml.safe_load(f)
    class_names = list(data_yaml["names"].values())
    class_names_path = output_dir / "class_names.txt"
    with open(class_names_path, "w") as f:
        f.write("\n".join(class_names))
    print(f"Class names saved to {class_names_path}")


if __name__ == "__main__":
    main()
