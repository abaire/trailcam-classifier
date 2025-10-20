from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from ultralytics import YOLO

from trailcam_classifier.main import predict_image
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

    label_dir = val_path / "labels"
    for image_path in tqdm(val_images, desc="Generating predictions"):
        label_path = label_dir / image_path.with_suffix(".txt").name
        if not label_path.exists():
            continue

        gt_labels = []
        gt_boxes = []
        with open(label_path) as infile:
            for line in infile:
                elements = line.split()

                true_class_idx = int(elements[0])
                left = float(elements[1])
                top = float(elements[2])
                right = float(elements[3])
                bottom = float(elements[4])

                gt_labels.append(true_class_idx)
                gt_boxes.append((left, top, right, bottom))

        # TODO: Support verifying that no false positives are found.
        if not gt_labels:
            continue

        pred_labels, pred_scores, pred_boxes = predict_image(image_path, model, confidence_threshold=0.0)

        # TODO: For each predicted box:
        #  1. Match with the box in gt_boxes that has the largest area of overlap with the predicted box.
        #     - If there is no overlap, insert a sentinel using the "unknown" class into y_true and the predicted label
        #       from pred_labels corresponding to the predicted box to y_pred.
        #     - If there is a valid match, populate y_true with the gt_labels entry corresponding to the box and add the
        #       pred_labels corresponding to the predicted box to y_pred.
        #  2. For each box in gt_boxes that has no overlap with a predicted box, add the gt_labels entry corresponding
        #     to the box to y_true and a sentinel entry using the "unknown" class into y_pred.

    if not (y_true or y_pred):
        msg = "Failed to find test images in dataset"
        raise ValueError(msg)

    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(output_dir / "confusion_matrix.csv")
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(output_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
