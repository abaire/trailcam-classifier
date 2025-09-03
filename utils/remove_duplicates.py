#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
from pathlib import Path

# ruff: noqa: T201 `print` found

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


def hash_file(path: Path) -> str:
    """Calculates the SHA256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def remove_duplicate_images(directory: str, *, dry_run: bool = False, golden_dirs: list[str] | None = None):
    """
    Recursively finds and removes duplicate images in a directory.

    Duplicates are identified by having the same file size and hash.
    """
    if dry_run:
        print("Dry run mode enabled. No files will be deleted.")

    hashes: dict[tuple[int, str], Path] = {}

    def _calculate_key(image_path: Path) -> tuple[int, str] | None:
        try:
            file_size = image_path.stat().st_size
            file_hash = hash_file(image_path)
        except FileNotFoundError:
            # This can happen if a file is deleted during the process
            # (e.g. it was a duplicate of another file that was processed earlier)
            return None

        return (file_size, file_hash)

    if golden_dirs:
        print(f"Scanning golden images in {golden_dirs}...")
        for image_path in find_images(golden_dirs):
            key = _calculate_key(image_path)
            if not key:
                continue
            hashes[key] = image_path

    print(f"Scanning for duplicate images in {directory}...")
    images = find_images([directory])
    duplicates_found = 0

    for image_path in images:
        key = _calculate_key(image_path)
        if not key:
            continue

        if key in hashes:
            if dry_run:
                print(f"Duplicate found: {image_path} is a duplicate of {hashes[key]} (would be deleted)")
            else:
                print(f"Duplicate found: {image_path} is a duplicate of {hashes[key]}")
                image_path.unlink()
            duplicates_found += 1
        else:
            hashes[key] = image_path

    if dry_run:
        print(f"Scan complete. Found {duplicates_found} duplicate images that would be removed.")
    else:
        print(f"Scan complete. Found and removed {duplicates_found} duplicate images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively find and remove duplicate images in a directory.")
    parser.add_argument(
        "directory",
        type=str,
        help="The directory to scan for duplicate images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be deleted, but do not delete them.",
    )
    parser.add_argument(
        "--golden-dir",
        "-g",
        nargs="+",
        help="Directories containing images that must be kept. Duplicate images outside of this directory will be removed.",
    )
    args = parser.parse_args()

    remove_duplicate_images(args.directory, dry_run=args.dry_run, golden_dirs=args.golden_dir)
