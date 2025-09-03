import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir_with_images():
    test_dir = tempfile.mkdtemp()
    subdir = Path(test_dir) / "subdir"
    subdir.mkdir()

    image1_path = Path(test_dir) / "image1.jpg"
    image2_path = Path(test_dir) / "image2.jpg"
    image3_path = Path(test_dir) / "image3.jpg"
    image4_path = subdir / "image4.jpg"
    text_file_path = Path(test_dir) / "notes.txt"

    image1_path.write_bytes(b"image data 1")
    image2_path.write_bytes(b"image data 2")
    image3_path.write_bytes(b"image data 1")
    image4_path.write_bytes(b"image data 1")
    text_file_path.write_text("this is not an image")

    yield test_dir, image1_path, image2_path, image3_path, image4_path, text_file_path

    shutil.rmtree(test_dir)


def test_remove_duplicates_script(temp_dir_with_images):
    test_dir, image1_path, image2_path, image3_path, image4_path, text_file_path = temp_dir_with_images
    script_path = Path(__file__).parent.parent / "utils" / "remove_duplicates.py"

    result = subprocess.run(
        [sys.executable, str(script_path), test_dir],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"Script failed with output:\n{result.stdout}\n{result.stderr}"

    assert image1_path.exists()
    assert image2_path.exists()
    assert not image3_path.exists()
    assert not image4_path.exists()
    assert text_file_path.exists()


def test_remove_duplicates_script_dry_run(temp_dir_with_images):
    test_dir, image1_path, image2_path, image3_path, image4_path, text_file_path = temp_dir_with_images
    script_path = Path(__file__).parent.parent / "utils" / "remove_duplicates.py"

    result = subprocess.run(
        [sys.executable, str(script_path), test_dir, "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"Script failed with output:\n{result.stdout}\n{result.stderr}"

    assert f"Duplicate found: {image3_path} is a duplicate of {image1_path} (would be deleted)" in result.stdout
    assert f"Duplicate found: {image4_path} is a duplicate of {image1_path} (would be deleted)" in result.stdout
    assert "Found 2 duplicate images that would be removed." in result.stdout

    assert image1_path.exists()
    assert image2_path.exists()
    assert image3_path.exists()
    assert image4_path.exists()
    assert text_file_path.exists()
