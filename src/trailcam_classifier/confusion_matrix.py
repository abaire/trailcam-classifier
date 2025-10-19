from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

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
    class_names = [*list(model.names.values()), "_EMPTY_"]

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

        pred_labels, pred_scores, pred_boxes = predict_image(image_path, model, confidence_threshold=0.0)
        empty_class_idx = len(model.names)

        if not gt_labels:
            # No ground truth labels, so all predictions are false positives
            for pred_label in pred_labels:
                y_true.append(empty_class_idx)
                y_pred.append(pred_label)
            continue

        if not pred_labels:
            # No predictions, so all ground truth labels are false negatives
            for gt_label in gt_labels:
                y_true.append(gt_label)
                y_pred.append(empty_class_idx)
            continue

        # Match predicted boxes with ground truth boxes
        iou = box_iou(torch.tensor(pred_boxes), torch.tensor(gt_boxes))
        max_iou, gt_idx = iou.max(1)
        matched_gt = set()

        for i, pred_label in enumerate(pred_labels):
            if max_iou[i] > 0:
                # A valid match is found
                matched_gt.add(int(gt_idx[i]))
                y_true.append(gt_labels[gt_idx[i]])
                y_pred.append(pred_label)
            else:
                # No overlap, this is a false positive
                y_true.append(empty_class_idx)
                y_pred.append(pred_label)

        # Find false negatives (ground truth boxes with no overlap)
        for i, gt_label in enumerate(gt_labels):
            if i not in matched_gt:
                y_true.append(gt_label)
                y_pred.append(empty_class_idx)

    if not (y_true or y_pred):
        msg = "Failed to find test images in dataset"
        raise ValueError(msg)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(output_dir / "confusion_matrix.csv")
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(output_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
