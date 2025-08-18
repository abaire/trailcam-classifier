from __future__ import annotations

import itertools

# ruff: noqa: T201 `print` found
# ruff: noqa: PLR2004 Magic value used in comparison
# ruff: noqa: DTZ007 Naive datetime constructed using `datetime.datetime.strptime()` without %z
# ruff: noqa: TRY300 Consider moving this statement to an `else` block
# ruff: noqa: BLE001 Do not catch blind exception: `Exception`
import os
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

DEFAULT_IMAGE_EXTENSIONS = {"jpg", "jpeg"}


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

    # Create a dictionary of human-readable tag names
    tag_dict = {TAGS[key]: val for key, val in exif_data.items() if key in TAGS}

    # Define the order of EXIF tags to check
    date_tags = ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]
    date_str = None

    for tag in date_tags:
        if tag in tag_dict:
            date_str = tag_dict[tag]
            break  # Stop when the first valid tag is found

    if not date_str:
        return None  # No usable date tag was found

    # Parse the EXIF date string ('YYYY:MM:DD HH:MM:SS') into a datetime object
    return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")


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


def _calculate_output_filename(image_path: Path) -> str:
    filename = os.path.basename(image_path)

    date_taken = get_image_datetime(image_path)
    if date_taken:
        base, ext = os.path.splitext(filename)
        timestamp_string = f"{date_taken.strftime('%Y%m%d-%H%M%S')}_"
        if not filename.startswith(timestamp_string):
            filename = f"{timestamp_string}{base}{ext}"
    return filename


if __name__ == "__main__":
    for image in find_images(["."]):
        output_filename = _calculate_output_filename(image)
        if not output_filename or output_filename == os.path.basename(image):
            continue

        output_path = os.path.join(os.path.dirname(image), output_filename)
        print(f"mv {image} {output_path}")
        # shutil.move(image, output_path)
