from __future__ import annotations

# ruff: noqa: DTZ007 Naive datetime constructed using `datetime.datetime.strptime()` without %z
import itertools
import os
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from PIL.ExifTags import TAGS

DEFAULT_IMAGE_EXTENSIONS = {"jpg", "jpeg"}

MODEL_SAVE_FILENAME = "trailcam_classifier_model.pth"


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
