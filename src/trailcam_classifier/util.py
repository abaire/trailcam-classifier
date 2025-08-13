from __future__ import annotations

import itertools
import os
from pathlib import Path

DEFAULT_IMAGE_EXTENSIONS = {"jpg", "jpeg"}


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
