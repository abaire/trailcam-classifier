from __future__ import annotations

# ruff: noqa: T201 `print` found
# ruff: noqa: PLR2004 Magic value used in comparison
# ruff: noqa: DTZ007 Naive datetime constructed using `datetime.datetime.strptime()` without %z
import argparse
import asyncio
import os
import shutil
import sys
from typing import TYPE_CHECKING

import torch
from PIL import Image
from producer_graph import NO_OUTPUT, Pipeline, standard_node
from torchvision.models import efficientnet_v2_s

from trailcam_classifier.util import (
    MODEL_SAVE_FILENAME,
    find_images,
    get_best_device,
    get_classification_transforms,
    get_image_datetime,
)

if TYPE_CHECKING:
    from pathlib import Path


def load_classifier(model_path: str, class_names_path: str):
    """Loads the fine-tuned model and corresponding class names."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(class_names_path):
        print(f"Error: Class names file not found at {class_names_path}", file=sys.stderr)
        sys.exit(1)

    with open(class_names_path) as f:
        class_names = [line.strip() for line in f]
    num_classes = len(class_names)

    device = get_best_device()

    model = efficientnet_v2_s(weights=None, num_classes=num_classes)
    transform = get_classification_transforms()

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    print(f"Loaded model and {num_classes} classes. Using device: {device}")
    return model, class_names, device, transform


def predict_image(image_path: str, model, device, transform):
    """Opens an image, preprocesses it, and returns the model's prediction."""
    img = Image.open(image_path).convert("RGB")

    img_tensor = transform(img)

    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()

    return predicted_idx, confidence


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
    args = parser.parse_args()

    model_path = args.model
    class_names_path = os.path.join(os.path.dirname(model_path), "class_names.txt")
    output_root = os.path.abspath(os.path.expanduser(args.output))

    model, class_names, device, transform = load_classifier(model_path, class_names_path)

    for name in class_names:
        os.makedirs(os.path.join(output_root, name), exist_ok=True)

    print(f"Looking for images in dirs {args.dirs}")
    image_paths = find_images(args.dirs, [args.output])
    if not image_paths:
        print("No images found to classify.")
        return 0

    print(f"\nFound {len(image_paths)} images. Starting classification...")

    def _calculate_output_filename(image_path: Path) -> tuple[Path, str]:
        filename = os.path.basename(image_path)
        date_taken = get_image_datetime(image_path)
        if date_taken:
            base, ext = os.path.splitext(filename)
            timestamp_string = f"{date_taken.strftime('%Y%m%d-%H%M%S')}_"
            if not filename.startswith(timestamp_string):
                filename = f"{timestamp_string}{base}{ext}"
        return image_path, filename

    def _classify(input_data: tuple[Path, str]) -> tuple[Path, str, str, float]:
        image_path, output_filename = input_data
        predicted_idx, confidence = predict_image(str(image_path), model, device, transform)

        if predicted_idx is None:
            return NO_OUTPUT

        predicted_class = class_names[predicted_idx]
        return image_path, output_filename, predicted_class, confidence

    def _save_output(result: tuple[Path, str, str, float]) -> None:
        image_path, output_filename, predicted_class, confidence = result

        base, ext = os.path.splitext(output_filename)
        if not args.omit_confidence:
            confidence_str = f"C{round(confidence * 100)}"
            if args.confidence_first:
                base = f"{confidence_str}_{base}"
            else:
                base += f"_{confidence_str}"

        filename = f"{base}{ext}"
        dest_dir = os.path.join(output_root, predicted_class)
        dest_path = os.path.join(dest_dir, filename)

        counter = 1
        while os.path.exists(dest_path):
            filename = f"{base}_{counter}{ext}"
            dest_path = os.path.join(dest_dir, filename)
            counter += 1

        if args.print_only:
            print(f"mv '{filename}' to '{predicted_class}' (Confidence: {confidence:.2%})")
        elif args.copy:
            shutil.copy2(image_path, dest_path)
            print(f"✅ Copied '{filename}' to '{predicted_class}' (Confidence: {confidence:.2%})")
        else:
            shutil.move(image_path, dest_path)
            print(f"✅ Moved '{filename}' to '{predicted_class}' (Confidence: {confidence:.2%})")

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


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
