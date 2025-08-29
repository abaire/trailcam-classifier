from __future__ import annotations

# ruff: noqa: DTZ007 Naive datetime constructed using `datetime.datetime.strptime()` without %z
# ruff: noqa: S311 Standard pseudo-random generators are not suitable for cryptographic purposes
import itertools
import os
import random
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from PIL.ExifTags import TAGS
from torchvision import transforms

DEFAULT_IMAGE_EXTENSIONS = {"jpg", "jpeg"}

MODEL_SAVE_FILENAME = "trailcam_classifier_model.pth"

NORMALIZATION = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_best_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def find_images(
    input_dirs: list[str], ignore_dirs: list[str] | None = None, extensions: set[str] | None = None
) -> set[Path]:
    """Recursively finds all images in the given input_dirs."""

    if not extensions:
        extensions = DEFAULT_IMAGE_EXTENSIONS

    input_dirs = [Path(os.path.expanduser(input_dir)) for input_dir in input_dirs]

    combined_results = itertools.chain.from_iterable(base_path.rglob("*.*") for base_path in input_dirs)
    all_files = set(combined_results)

    if ignore_dirs is None:
        ignore_dirs = []
    ignored_paths = [Path(ignored) for ignored in ignore_dirs]

    def keep_file(filename: Path) -> bool:
        if any(filename.is_relative_to(ignored) for ignored in ignored_paths):
            return False

        if not filename.is_file():
            return False

        return filename.suffix[1:].lower() in extensions

    return {filename for filename in all_files if keep_file(filename)}


def get_image_datetime(image_path) -> datetime | None:
    """
    Extracts the best available timestamp from an image's EXIF data and
    returns it as a datetime object. It checks for 'DateTimeOriginal',
    'DateTimeDigitized', and 'DateTime' in that order.
    """
    image = Image.open(image_path)
    exif_data = image.getexif()

    if not exif_data:
        return None

    tag_dict = {TAGS[key]: val for key, val in exif_data.items() if key in TAGS}

    date_tags = ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]
    date_str = None

    for tag in date_tags:
        if tag in tag_dict:
            date_str = tag_dict[tag]
            break

    if not date_str:
        return None

    return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")


class CropInfoBar:
    """A transform to clip the info bar off the bottom of an image."""

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        clip_heights = {
            1080: 1008,
            1512: 1411,
            2376: 2217,
        }
        target_height = clip_heights.get(height)
        if not target_height:
            msg = f"Unexpected image height {height}"
            raise ValueError(msg)

        crop_box = (0, 0, width, target_height)
        return img.crop(crop_box)


def get_classification_transforms() -> transforms.Compose:
    """
    Returns a composition of transforms for preprocessing an image for classification.
    """
    image_size = 384  # From EfficientNetV2-S spec

    return transforms.Compose(
        [
            CropInfoBar(),
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            NORMALIZATION,
        ]
    )


def slice_with_feature(img: Image.Image, feature_coords: dict[str, int]) -> Image.Image:
    """
    Slices the image to a random new size, ensuring that at least 50% of
    the feature is visible.
    """
    img_width, img_height = img.size
    x1, y1, x2, y2 = (
        feature_coords["x1"],
        feature_coords["y1"],
        feature_coords["x2"],
        feature_coords["y2"],
    )

    feature_width = x2 - x1
    feature_height = y2 - y1

    if feature_width <= 0 or feature_height <= 0:
        return img

    # 1. Determine the minimum required overlap dimensions (e.g., sqrt(0.5))
    min_overlap_width = int(feature_width * 0.707)
    min_overlap_height = int(feature_height * 0.707)

    # 2. Choose a random actual overlap size, between minimum and full feature.
    overlap_w = random.randint(min_overlap_width, feature_width)
    overlap_h = random.randint(min_overlap_height, feature_height)

    # 3. Randomly place this overlap box within the feature area.
    overlap_x1 = random.randint(x1, x2 - overlap_w)
    overlap_y1 = random.randint(y1, y2 - overlap_h)
    overlap_x2 = overlap_x1 + overlap_w
    overlap_y2 = overlap_y1 + overlap_h

    # 4. Randomly define the crop box boundaries, making sure it contains the overlap box.
    crop_x1 = random.randint(0, overlap_x1)
    crop_y1 = random.randint(0, overlap_y1)
    crop_x2 = random.randint(overlap_x2, img_width)
    crop_y2 = random.randint(overlap_y2, img_height)

    return img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
