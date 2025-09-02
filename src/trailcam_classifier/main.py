from __future__ import annotations

# ruff: noqa: T201 `print` found
# ruff: noqa: PLR2004 Magic value used in comparison
# ruff: noqa: DTZ007 Naive datetime constructed using `datetime.datetime.strptime()` without %z
import argparse
import asyncio
import os
import shutil
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from producer_graph import NO_OUTPUT, Pipeline, standard_node
from ultralytics import YOLO

from trailcam_classifier.util import (
    MODEL_SAVE_FILENAME,
    find_images,
    get_image_datetime,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class ClassificationConfig:
    """Configuration for the classification process."""

    dirs: list[str]
    model: str = MODEL_SAVE_FILENAME
    output: str = "classified_output"
    print_only: bool = False
    copy: bool = False
    omit_confidence: bool = False
    confidence_first: bool = False
    multiclass: bool = False
    confidence_threshold: float = 0.5


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


def predict_image(image_path: str, model: YOLO, confidence_threshold: float = 0.5):
    """Opens an image, preprocesses it, and returns the model's prediction."""
    results = model.predict(image_path, verbose=False)
    result = results[0]

    boxes = result.boxes
    pred_boxes = []
    pred_labels = []
    pred_scores = []
    for box in boxes:
        if box.conf[0] > confidence_threshold:
            class_id = int(box.cls[0])
            pred_labels.append(class_id)
            pred_scores.append(float(box.conf[0]))
            pred_boxes.append([float(coord) for coord in box.xyxy[0]])

    return pred_labels, pred_scores, pred_boxes


async def run_classification(config: ClassificationConfig, logger: Callable[[str], None] = print):
    """Runs the image classification process."""
    model_path = config.model
    class_names_path = os.path.join(os.path.dirname(model_path), "class_names.txt")
    output_root = os.path.abspath(os.path.expanduser(config.output))

    model, class_names = load_detector(model_path, class_names_path, logger)
    if not model:
        return 1

    for name in class_names:
        os.makedirs(os.path.join(output_root, name), exist_ok=True)

    logger(f"Looking for images in dirs {config.dirs}")
    image_paths = find_images(config.dirs, [config.output])
    if not image_paths:
        logger("No images found to classify.")
        return 0

    logger(f"\nFound {len(image_paths)} images. Starting classification...")

    def _calculate_output_filename(image_path: Path) -> tuple[Path, str]:
        filename = os.path.basename(image_path)
        date_taken = get_image_datetime(image_path)
        if date_taken:
            base, ext = os.path.splitext(filename)
            timestamp_string = f"{date_taken.strftime('%Y%m%d-%H%M%S')}_"
            if not filename.startswith(timestamp_string):
                filename = f"{timestamp_string}{base}{ext}"
        return image_path, filename

    def _classify(input_data: tuple[Path, str]) -> tuple[Path, str, list[tuple[str, float]]]:
        image_path, output_filename = input_data
        predicted_indices, confidences, _ = predict_image(str(image_path), model, config.confidence_threshold)

        if not predicted_indices:
            return NO_OUTPUT

        # Create a list of (class_name, confidence) tuples
        predictions = []
        for i, index in enumerate(predicted_indices):
            class_name = class_names[index]
            predictions.append((class_name, confidences[i]))

        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)

        if not predictions:
            return NO_OUTPUT

        return image_path, output_filename, predictions

    def _save_output(result: tuple[Path, str, list[tuple[str, float]]]) -> None:
        image_path, output_filename, predictions = result
        primary_class, primary_confidence = predictions[0]

        base, ext = os.path.splitext(output_filename)
        if not config.omit_confidence:
            if config.multiclass and len(predictions) > 1:
                secondary_class, secondary_confidence = predictions[1]
                confidence_str = (
                    f"C{round(primary_confidence * 100)}_{primary_class}_"
                    f"C{round(secondary_confidence * 100)}_{secondary_class}"
                )
            else:
                confidence_str = f"C{round(primary_confidence * 100)}"

            if config.confidence_first:
                base = f"{confidence_str}_{base}"
            else:
                base += f"_{confidence_str}"

        filename = f"{base}{ext}"
        dest_dir = os.path.join(output_root, primary_class)
        dest_path = os.path.join(dest_dir, filename)

        counter = 1
        while os.path.exists(dest_path):
            filename = f"{base}_{counter}{ext}"
            dest_path = os.path.join(dest_dir, filename)
            counter += 1

        if config.print_only:
            logger(f"mv '{filename}' to '{primary_class}' (Confidence: {primary_confidence:.2%})")
        elif config.copy:
            shutil.copy2(image_path, dest_path)
            logger(f"✅ Copied '{filename}' to '{primary_class}' (Confidence: {primary_confidence:.2%})")
        else:
            shutil.move(image_path, dest_path)
            logger(f"✅ Moved '{filename}' to '{primary_class}' (Confidence: {primary_confidence:.2%})")

    producer_graph = [
        standard_node(name="augment_filename", transform=_calculate_output_filename, num_workers=4, max_queue_size=128),
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
    parser.add_argument("--omit-confidence", action="store_true", help="Do not add model confidence to filenames.")
    parser.add_argument("--confidence-first", "-F", action="store_true", help="Prefix filenames with model confidence.")
    parser.add_argument("--multiclass", action="store_true", help="Add primary and secondary classes to the filename.")
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
        omit_confidence=args.omit_confidence,
        confidence_first=args.confidence_first,
        multiclass=args.multiclass,
        confidence_threshold=args.confidence_threshold,
    )
    return await run_classification(config)


def run():
    """Entry point for the `trailcamclassify` script."""
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    run()
