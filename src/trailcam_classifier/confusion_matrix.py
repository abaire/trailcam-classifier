from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sn
import torch
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from ultralytics import YOLO

from trailcam_classifier.util import find_images


def main():
    parser = argparse.ArgumentParser(
        description="""
        Generate a confusion matrix from a trained model and a dataset.
        """
    )
    parser.add_argument("model", help="Path to the trained model file.")
    parser.add_argument("dataset", help="Path to the dataset.yaml file.")
    parser.add_argument("output_dir", help="Directory to save the confusion matrix.")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    class_names = list(model.names.values())

    with open(args.dataset) as f:
        data_yaml = yaml.safe_load(f)
    dataset_dir = Path(args.dataset).parent
    val_path = dataset_dir / data_yaml["val"]
    val_images = find_images([str(val_path)])
    y_true = []
    y_pred = []

    label_dir = val_path.parent / "labels"
    for image_path in tqdm(val_images, desc="Generating predictions"):
        label_path = label_dir / image_path.with_suffix(".txt").name
        if not label_path.exists():
            continue

        with open(label_path) as f:
            lines = f.readlines()
            if not lines:
                continue
            true_class_idx = int(lines[0].split()[0])
            y_true.append(true_class_idx)

        results = model.predict(image_path, verbose=False)
        pred_class_idx = int(results[0].probs.top1)
        y_pred.append(pred_class_idx)

    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(output_dir / "confusion_matrix.csv")
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(output_dir / "confusion_matrix.png")

if __name__ == "__main__":
    main()
