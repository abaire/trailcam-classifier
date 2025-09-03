import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from trailcam_classifier.main import ClassificationConfig, run_classification


@pytest.fixture
def temp_image_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy image files
        for i in range(3):
            image = Image.new("RGB", (100, 100), color="red")
            image.save(Path(temp_dir, f"image{i}.jpg"))
        yield temp_dir


@patch("trailcam_classifier.main.YOLO")
def test_keep_empty_moves_files(mock_yolo, temp_image_dir):
    # Mock the YOLO model to return no detections
    mock_model = MagicMock()
    mock_model.predict.return_value = [MagicMock(boxes=[])]
    mock_yolo.return_value = mock_model

    output_dir = os.path.join(temp_image_dir, "output")
    config = ClassificationConfig(
        dirs=[temp_image_dir],
        output=output_dir,
        keep_empty=True,
        model="dummy.pt",
    )

    # Create dummy model and class names files
    Path(config.model).touch()
    with open("class_names.txt", "w") as f:
        f.write("class1\n")

    asyncio.run(run_classification(config))

    empty_dir = Path(output_dir) / "_empty_"
    assert empty_dir.exists()
    assert len(list(empty_dir.glob("*.jpg"))) == 3

    # Cleanup dummy files
    os.remove(config.model)
    os.remove("class_names.txt")


@patch("trailcam_classifier.main.YOLO")
def test_keep_empty_with_copy_copies_files(mock_yolo, temp_image_dir):
    # Mock the YOLO model to return no detections
    mock_model = MagicMock()
    mock_model.predict.return_value = [MagicMock(boxes=[])]
    mock_yolo.return_value = mock_model

    output_dir = os.path.join(temp_image_dir, "output")
    config = ClassificationConfig(
        dirs=[temp_image_dir],
        output=output_dir,
        keep_empty=True,
        copy=True,
        model="dummy.pt",
    )

    # Create dummy model and class names files
    Path(config.model).touch()
    with open("class_names.txt", "w") as f:
        f.write("class1\n")

    asyncio.run(run_classification(config))

    empty_dir = Path(output_dir) / "_empty_"
    assert empty_dir.exists()
    assert len(list(empty_dir.glob("*.jpg"))) == 3
    # Check that original files still exist
    assert len(list(Path(temp_image_dir).glob("*.jpg"))) == 3

    # Cleanup dummy files
    os.remove(config.model)
    os.remove("class_names.txt")
