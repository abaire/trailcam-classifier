from __future__ import annotations

# ruff: noqa: T201 `print` found
import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

from trailcam_classifier.util import MODEL_SAVE_FILENAME


def main():
    parser = argparse.ArgumentParser(description="Train a YOLO model.")
    parser.add_argument("dataset", help="Path to the dataset.yaml file.")
    parser.add_argument("output_dir", help="Directory into which the final model should be saved.")

    parser.add_argument("--epochs", type=int, default=125, help="Number of epochs to train for.")

    parser.add_argument("--model", default="yolo11m.pt", help="The YOLO model to use.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

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
