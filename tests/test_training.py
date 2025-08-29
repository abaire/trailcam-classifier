import json
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from trailcam_classifier.training import ImageDataset


def test_image_slicing_with_json():
    """
    Tests that if a JSON file with the same name as an image exists, the image
    is sliced according to the coordinates in the JSON file.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a subdirectory for a class
        class_dir = Path(temp_dir) / "test_class"
        class_dir.mkdir()

        # Create a dummy image
        image_path = class_dir / "test_image.jpg"
        original_image = Image.new("RGB", (200, 200), color="red")
        original_image.save(image_path)

        # Create a corresponding JSON file
        json_path = class_dir / "test_image.json"
        feature_coords = {"x1": 50, "y1": 50, "x2": 150, "y2": 150}
        with open(json_path, "w") as f:
            json.dump(feature_coords, f)

        # Create an ImageDataset
        dataset = ImageDataset(temp_dir)

        # Get the item
        image, _ = dataset[0]

        # Assert that the image has been sliced
        # The slice is random, so we can't check for exact dimensions.
        # But it should be smaller than the original.
        assert image.size[0] < original_image.size[0]
        assert image.size[1] < original_image.size[1]

        # Also assert that the crop is not empty
        assert image.size[0] > 0
        assert image.size[1] > 0

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


def test_image_loading_without_json():
    """
    Tests that if no JSON file exists, the image is loaded without slicing.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a subdirectory for a class
        class_dir = Path(temp_dir) / "test_class"
        class_dir.mkdir()

        # Create a dummy image
        image_path = class_dir / "test_image.jpg"
        original_image = Image.new("RGB", (200, 200), color="red")
        original_image.save(image_path)

        # Create an ImageDataset
        dataset = ImageDataset(temp_dir)

        # Get the item
        image, _ = dataset[0]

        # Assert that the image has not been sliced
        assert image.size == original_image.size

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
