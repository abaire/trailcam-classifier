from __future__ import annotations

# ruff: noqa: T201 `print` found
import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from PIL import Image
from tqdm import tqdm

from trailcam_classifier.util import find_images

if TYPE_CHECKING:
    from collections.abc import Collection

logger = logging.getLogger(__name__)


def convert_bbox_to_yolo(img_width, img_height, bbox):
    """Converts a bounding box from [x1, y1, x2, y2] to YOLO format."""
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

    if x1 > x2:
        tmp = x1
        x1 = x2
        x2 = tmp
    if y1 > y2:
        tmp = y1
        y1 = y2
        y2 = tmp

    dw = 1.0 / img_width
    dh = 1.0 / img_height
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    x_center_norm = x_center * dw
    y_center_norm = y_center * dh
    width_norm = width * dw
    height_norm = height * dh

    return x_center_norm, y_center_norm, width_norm, height_norm


def _discover_images(data_dirs: list[str]) -> tuple[set[Path], list[str]]:
    all_image_paths = find_images(data_dirs)
    class_names = set()
    for img_path in tqdm(all_image_paths, desc="Discovering classes"):
        json_path = img_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path) as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logger.exception("Exception while processing '%s'", json_path)
                    sys.exit(1)

                class_names.update(data.keys())

    class_names = sorted(class_names)

    return all_image_paths, class_names


def _process_metadata_files(all_image_paths: Collection[Path], class_to_idx: dict[str, int]) -> list[Path]:
    labeled_image_paths = []
    for img_path in tqdm(all_image_paths, desc="Generating labels"):
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            labeled_image_paths.append(img_path)
            continue

        json_path = img_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)

            with open(txt_path, "w") as f_out:
                img = Image.open(img_path)
                for class_name, bboxes in data.items():
                    if class_name in class_to_idx:
                        class_idx = class_to_idx[class_name]
                        for bbox in bboxes:
                            yolo_bbox = convert_bbox_to_yolo(img.width, img.height, bbox)
                            f_out.write(f"{class_idx} {' '.join(map(str, yolo_bbox))}\n")

            labeled_image_paths.append(img_path)
        else:
            print(f"Warning: No .txt or .json file found for {img_path}. Skipping file.")

    return labeled_image_paths


def _group_new_images(new_image_paths: Collection[Path], val_split: float) -> tuple[set[Path], set[Path]]:
    new_class_to_images = defaultdict(list)
    for img_path in tqdm(new_image_paths, desc="Grouping new images by class"):
        json_path = img_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
                for class_name in data:
                    new_class_to_images[class_name].append(img_path)

    new_val_paths = set()
    for images_in_class in new_class_to_images.values():
        random.shuffle(images_in_class)
        val_count = int(len(images_in_class) * val_split)
        if val_count == 0 and len(images_in_class) > 0:
            val_count = 1
        new_val_paths.update(images_in_class[:val_count])

    new_train_paths = set(new_image_paths) - new_val_paths
    return new_train_paths, new_val_paths


def _move_entries(new_train_paths: set[Path], new_val_paths: set[Path], output_dir: Path):
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    def _move_dataset(items: set[Path], target_dir: Path):
        image_dir = target_dir / "images"
        label_dir = target_dir / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for image in items:
            target_image = image_dir / image.name

            label = image.with_suffix(".txt")
            target_label = label_dir / label.name

            image.rename(target_image)
            label.rename(target_label)

    _move_dataset(new_train_paths, train_dir)
    _move_dataset(new_val_paths, val_dir)


def main():
    parser = argparse.ArgumentParser(
        description="""
        Prepare a directory of images and JSON annotations for YOLO training.

        This script processes a directory of images and corresponding JSON files and generates
        a YOLO formatted dataset.
        """
    )
    parser.add_argument(
        "data_dir",
        nargs="+",
        help="Path to a directory containing images and metadata that should be moved to the output dataset.",
    )
    parser.add_argument(
        "--dataset-dir", "-o", default="dataset", help="Directory into which a YOLO dataset should be organized."
    )
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio for new images.")
    args = parser.parse_args()

    output_dir = Path(args.dataset_dir)
    val_split = args.val_split

    # Check for existing classes
    yaml_path = output_dir / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
            if "names" in data:
                # Note that PyYAML may load this as a list or a dict.
                if isinstance(data["names"], dict):
                    existing_class_names = set(data["names"].values())
                else:
                    existing_class_names = set(data["names"])
    else:
        existing_class_names = set()

    # Find all images and discover classes from their JSON files
    all_image_paths, new_class_names = _discover_images(args.data_dir)

    # Merge class lists
    class_names = sorted(existing_class_names.union(new_class_names))
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # Generate .txt label files from .json files if they don't exist
    labeled_image_paths = _process_metadata_files(all_image_paths, class_to_idx)

    random.seed(42)
    new_train_paths, new_val_paths = _group_new_images(labeled_image_paths, val_split)

    _move_entries(new_train_paths, new_val_paths, output_dir)

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir.resolve()}\n")
        f.write("train: train\n")
        f.write("val: val\n")
        f.write("\n")
        f.write("names:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {i}: {name}\n")

    print(f"Dataset successfully prepared at '{output_dir}'")


if __name__ == "__main__":
    main()
