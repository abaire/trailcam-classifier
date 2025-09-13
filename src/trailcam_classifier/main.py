from __future__ import annotations

# ruff: noqa: T201 `print` found
# ruff: noqa: PLR2004 Magic value used in comparison
# ruff: noqa: DTZ007 Naive datetime constructed using `datetime.datetime.strptime()` without %z
import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
from producer_graph import NO_OUTPUT, Pipeline, standard_node
from tqdm import tqdm
from ultralytics import YOLO

from trailcam_classifier.util import (
    MODEL_SAVE_FILENAME,
    find_images,
    get_image_datetime,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ClassificationConfig:
    """Configuration for the classification process."""

    dirs: list[str]
    model: str = MODEL_SAVE_FILENAME
    output: str = "classified_output"
    print_only: bool = False
    copy: bool = False
    confidence_threshold: float = 0.5
    keep_empty: bool = False


def load_detector(model_path: str, class_names_path: str, logger: Callable[[str], None] = print):
    """Loads the fine-tuned model and corresponding class names."""
    if not os.path.exists(model_path):
        logger(f"Error: Model file not found at {model_path}")
        return None, None
    if not os.path.exists(class_names_path):
        logger(f"Error: Class names file not found at {class_names_path}")
        return None, None

    with open(class_names_path) as f:
        class_names = [line.strip() for line in f]

    model = YOLO(model_path)
    logger(f"Loaded model and {len(class_names)} classes.")
    return model, class_names


_device_name: str | None = None


def predict_image(
    image_path: str, model: YOLO, confidence_threshold: float = 0.5
) -> tuple[list[Any], list[Any], list[Any]] | None:
    """Opens an image, preprocesses it, and returns the model's prediction."""
    global _device_name  # noqa: PLW0603 Using the global statement to update `_device_name` is discouraged
    try:
        if _device_name:
            results = model.predict(image_path, verbose=False, device=_device_name)
        else:
            try:
                results = model.predict(image_path, verbose=False, device="cuda")
                _device_name = "cuda"
            except ValueError:
                if sys.platform == "darwin":
                    device_to_try = "mps"
                else:
                    device_to_try = "cpu"
                try:
                    results = model.predict(image_path, verbose=False, device=device_to_try)
                    _device_name = device_to_try
                except ValueError:
                    _device_name = "cpu"
                    results = model.predict(image_path, verbose=False, device="cpu")
    except cv2.error:
        logger.exception("Failed to process image %s", image_path)
        return None

    result = results[0]

    boxes = result.boxes
    pred_boxes = []
    pred_labels = []
    pred_scores = []
    if boxes:
        for box in boxes:
            if box.conf[0] > confidence_threshold:
                class_id = int(box.cls[0])
                pred_labels.append(class_id)
                pred_scores.append(float(box.conf[0]))
                pred_boxes.append([float(coord) for coord in box.xyxy[0]])

    return pred_labels, pred_scores, pred_boxes


async def run_classification(
    config: ClassificationConfig,
    logger: Callable[[str], None] = print,
    progress_update: Callable[[str, int], None] | None = None,
):
    """Runs the image classification process."""
    model_path = config.model
    class_names_path = os.path.join(os.path.dirname(model_path), "class_names.txt")
    output_root = os.path.abspath(os.path.expanduser(config.output))

    model, class_names = load_detector(model_path, class_names_path, logger)
    if not model:
        return 1

    os.makedirs(output_root, exist_ok=True)

    logger(f"Looking for images in dirs {config.dirs}")
    image_paths = find_images(config.dirs, [config.output])
    if not image_paths:
        logger("No images found to classify.")
        return 0

    total_images = len(image_paths)
    logger(f"\nFound {total_images} images. Starting classification...")

    pbar = tqdm(total=total_images, desc="Classifying images")

    def update_progress(image_file: Path):
        pbar.update(1)
        if progress_update:
            progress_update(str(image_file), total_images)

    def _calculate_output_filename(image_path: Path) -> tuple[Path, str]:
        filename = os.path.basename(image_path)
        date_taken = get_image_datetime(image_path)
        if date_taken:
            base, ext = os.path.splitext(filename)
            timestamp_string = f"{date_taken.strftime('%Y%m%d-%H%M%S')}_"
            if not filename.startswith(timestamp_string):
                filename = f"{timestamp_string}{base}{ext}"
        return image_path, filename

    def _classify(input_data: tuple[Path, str]):
        image_path, output_filename = input_data
        prediction = predict_image(str(image_path), model, config.confidence_threshold)
        if not prediction:
            return NO_OUTPUT
        predicted_indices, confidences, bboxes = prediction

        if not predicted_indices:
            if config.keep_empty:
                return image_path, output_filename, None
            update_progress(image_path)
            return NO_OUTPUT

        # Create a list of (class_name, confidence, bounding_box) tuples
        detections = []
        for i, index in enumerate(predicted_indices):
            class_name = class_names[index]
            detections.append((class_name, confidences[i], bboxes[i]))

        if not detections:
            if config.keep_empty:
                return image_path, output_filename, None
            update_progress(image_path)
            return NO_OUTPUT

        return image_path, output_filename, detections

    def _save_output(result: tuple[Path, str, list[tuple[str, float, list[float]]] | None]) -> None:
        image_path, output_filename, detections = result

        if detections is None:
            # This is an empty image
            if config.print_only:
                logger(f"mv '{image_path.name}' '_empty/{output_filename}'")
                update_progress(image_path)
                return

            empty_dir = os.path.join(output_root, "_empty_")
            os.makedirs(empty_dir, exist_ok=True)
            dest_path = os.path.join(empty_dir, output_filename)
            if config.copy:
                shutil.copy2(image_path, dest_path)
            else:
                shutil.move(image_path, dest_path)
            update_progress(image_path)
            return

        class_counts: dict[str, int] = {}
        for class_name, _, _ in detections:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        sorted_class_names = sorted(class_counts.keys())

        rename_suffix = ""
        for class_name in sorted_class_names:
            count = class_counts[class_name]
            rename_suffix += f"__{count}{class_name}"

        base, ext = os.path.splitext(output_filename)
        filename = f"{base}{rename_suffix}{ext}"

        json_data = defaultdict(list)
        for class_name, confidence, bbox in detections:
            x1, y1, x2, y2 = bbox
            json_data[class_name].append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": confidence})

        if config.print_only:
            logger(f"mv '{image_path.name}' '{filename}'")
            logger(json.dumps(json_data, indent=2))
            update_progress(image_path)
            return

        base, ext = os.path.splitext(filename)
        counter = 1
        dest_path = os.path.join(output_root, filename)
        while os.path.exists(dest_path):
            filename = f"{base}_{counter}{ext}"
            dest_path = os.path.join(output_root, filename)
            counter += 1

        json_base, _ = os.path.splitext(filename)
        json_filename = f"{json_base}.json"
        json_dest_path = os.path.join(output_root, json_filename)
        with open(json_dest_path, "w") as f:
            json.dump(json_data, f, indent=2)

        if config.copy:
            try:
                shutil.copy2(image_path, dest_path)
            except PermissionError:
                shutil.copy(image_path, dest_path)
        else:
            try:
                shutil.move(image_path, dest_path)
            except PermissionError:
                shutil.copy(image_path, dest_path)
                os.unlink(image_path)
        update_progress(image_path)

    producer_graph = [
        standard_node(name="augment_filename", transform=_calculate_output_filename, num_workers=2, max_queue_size=128),
        standard_node(
            name="classify",
            transform=_classify,
            spawn_thread=True,
            num_workers=1,
            max_queue_size=1000,
            input_node="augment_filename",
        ),
        standard_node(
            name="save_output",
            transform=_save_output,
            spawn_thread=True,
            num_workers=2,
            max_queue_size=100,
            input_node="classify",
        ),
    ]

    pipeline = Pipeline(producer_graph)
    await pipeline.run(image_paths)

    if pbar:
        pbar.close()

    logger("Completed successfully")

    return 0


async def main():
    parser = argparse.ArgumentParser(
        description="A utility to automatically classify JPG images based on their contents.",
        epilog="Example: python classify.py ./photos_to_classify",
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        help="One or more source directories to search for images recursively.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=MODEL_SAVE_FILENAME,
        help="Path to the .pth model weights file (default: image_classifier_model.pth).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="classified_output",
        help="Base directory into which classified files should be moved (default: classified_output).",
    )
    parser.add_argument(
        "--print-only", action="store_true", help="Print the classification for each file instead of moving them."
    )
    parser.add_argument("--copy", "-c", action="store_true", help="Copy files to outputs instead of moving them.")
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Move/copy images with no detections to a special '_empty_' subdirectory.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for displaying detections.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    config = ClassificationConfig(
        dirs=args.dirs,
        model=args.model,
        output=args.output,
        print_only=args.print_only,
        copy=args.copy,
        confidence_threshold=args.confidence_threshold,
        keep_empty=args.keep_empty,
    )
    return await run_classification(config)


def run():
    """Entry point for the `trailcamclassify` script."""
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    run()
