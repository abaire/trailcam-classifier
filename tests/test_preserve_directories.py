import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from trailcam_classifier.main import ClassificationConfig, run_classification


@pytest.fixture
def temp_image_dir_structure():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a subdirectory structure
        sub_dir = Path(temp_dir) / "subir"
        sub_dir.mkdir()

        # Create dummy image file
        image = Image.new("RGB", (100, 100), color="blue")
        image.save(sub_dir / "image.jpg")
        yield temp_dir, sub_dir


@patch("trailcam_classifier.main.YOLO")
def test_preserve_directories_json_location(mock_yolo, temp_image_dir_structure):
    """
    Test that when preserve_directories is True, the JSON file is saved in the preserved subdirectory,
    not in the root output directory.
    """
    root_dir, sub_dir = temp_image_dir_structure

    # Mock the YOLO model to return a detection
    mock_model = MagicMock()
    # Return one detection: class 0 with 0.9 confidence
    mock_box = MagicMock()
    mock_box.conf = [0.9]
    mock_box.cls = [0.0]
    mock_box.xyxy = [[0.0, 0.0, 50.0, 50.0]]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]

    mock_model.predict.return_value = [mock_result]
    mock_yolo.return_value = mock_model

    output_dir = os.path.join(root_dir, "output")
    config = ClassificationConfig(
        dirs=[root_dir], output=output_dir, keep_empty=False, preserve_directories=True, model="dummy.pt", copy=True
    )

    # Create dummy model and class names files
    Path(config.model).touch()
    with open("class_names.txt", "w") as f:
        f.write("test_class\n")

    try:
        asyncio.run(run_classification(config))

        # Define expected paths
        # Structure is root_dir/subdir/image.jpg
        # Output should be output_dir/subdir/image__1test_class.jpg and .json

        rel_subdir = Path(sub_dir).relative_to(root_dir)
        expected_output_subdir = Path(output_dir) / rel_subdir

        # Check that the image was moved/copied correctly
        assert expected_output_subdir.exists(), "Output subdirectory should exist"
        images = list(expected_output_subdir.glob("*.jpg"))
        assert len(images) == 1, "Should have one classified image in the subdirectory"

        # Check for JSON file
        # The bug is that JSON is in output_dir, not expected_output_subdir
        json_files_in_subdir = list(expected_output_subdir.glob("*.json"))
        json_files_in_root = list(Path(output_dir).glob("*.json"))

        # This assertions is expected to FAIL until the bug is fixed
        assert len(json_files_in_subdir) == 1, (
            f"JSON file should be in {expected_output_subdir}, but found {len(json_files_in_subdir)}"
        )
        assert len(json_files_in_root) == 0, (
            f"JSON file should NOT be in root {output_dir}, but found {len(json_files_in_root)}"
        )

    finally:
        # Cleanup dummy files
        if os.path.exists(config.model):
            os.remove(config.model)
        if os.path.exists("class_names.txt"):
            os.remove("class_names.txt")
