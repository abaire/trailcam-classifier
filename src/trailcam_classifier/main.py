from __future__ import annotations

# ruff: noqa: T201 `print` found
import argparse
import itertools
import os
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


def load_classifier(model_path: str, class_names_path: str):
    """Loads the fine-tuned model and corresponding class names."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(class_names_path):
        print(f"Error: Class names file not found at {class_names_path}", file=sys.stderr)
        sys.exit(1)

    # Load the class names
    with open(class_names_path) as f:
        class_names = [line.strip() for line in f]
    num_classes = len(class_names)

    # Recreate the model structure
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=None, num_classes=num_classes)

    # Load the trained weights (the state_dict)
    model.load_state_dict(torch.load(model_path))

    # Set up device and put model in evaluation mode
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)
    model.eval()

    print(f"Loaded model and {num_classes} classes. Using device: {device}")
    return model, class_names, device, weights.transforms()


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


def _find_sources(input_dirs: list[str], output_dir: str) -> set[str]:
    input_dirs = [Path(os.path.expanduser(input_dir)) for input_dir in input_dirs]

    combined_results = itertools.chain.from_iterable(base_path.rglob("*.*") for base_path in input_dirs)
    all_files = set(combined_results)

    output_path = Path(output_dir)

    def keep_file(filename: Path) -> bool:
        if filename.is_relative_to(output_path):
            return False

        if not filename.is_file():
            return False

        return filename.suffix[1:].lower() in {"jpg", "jpeg"}

    return {filename for filename in all_files if keep_file(filename)}


def main():
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
        default="image_classifier_model.pth",
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
    args = parser.parse_args()

    model_path = args.model
    class_names_path = os.path.join(os.path.dirname(model_path), "class_names.txt")
    output_root = os.path.abspath(os.path.expanduser(args.output))

    model, class_names, device, transform = load_classifier(model_path, class_names_path)

    for name in class_names:
        os.makedirs(os.path.join(output_root, name), exist_ok=True)

    image_paths = _find_sources(args.dirs, args.output)
    if not image_paths:
        print("No images found to classify.")
        return 0

    print(f"\nFound {len(image_paths)} images. Starting classification...")

    for image_path in image_paths:
        predicted_idx, confidence = predict_image(image_path, model, device, transform)

        if predicted_idx is None:
            continue

        predicted_class = class_names[predicted_idx]
        filename = os.path.basename(image_path)
        dest_dir = os.path.join(output_root, predicted_class)
        dest_path = os.path.join(dest_dir, filename)

        if args.print_only:
            print(f"mv '{filename}' to '{predicted_class}' (Confidence: {confidence:.2%})")
        else:
            shutil.move(image_path, dest_path)
            print(f"✅ Moved '{filename}' to '{predicted_class}' (Confidence: {confidence:.2%})")

    print("\nClassification complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
