import asyncio
import json
import shutil
import tempfile
from pathlib import Path

import torch
from PIL import Image
from torchvision import tv_tensors

from trailcam_classifier.main import run_classification, ClassificationConfig
from trailcam_classifier.training import ImageDataset, train_model, SchedulerMode


def setup_test_environment(num_images=5):
    """Creates a temporary directory with dummy images and JSON annotations."""
    temp_dir = tempfile.mkdtemp()
    class_dir = Path(temp_dir) / "cat"
    class_dir.mkdir()

    for i in range(num_images):
        # Create a dummy image
        image_path = class_dir / f"test_image_{i}.jpg"
        image = Image.new("RGB", (200, 200), color="red")
        image.save(image_path)

        # Create a corresponding JSON file
        json_path = class_dir / f"test_image_{i}.json"
        feature_coords = {"cat": [{"x1": 50, "y1": 50, "x2": 150, "y2": 150}]}
        with open(json_path, "w") as f:
            json.dump(feature_coords, f)

    return temp_dir


def test_dataset_loading():
    """Tests that the dataset loads images and annotations correctly."""
    temp_dir = setup_test_environment(num_images=5)
    try:
        dataset = ImageDataset(temp_dir)
        assert len(dataset) == 5
        assert dataset.classes == ["cat"]

        image, target = dataset[0]
        assert isinstance(image, tv_tensors.Image)
        assert "boxes" in target
        assert "labels" in target
        assert target["boxes"].shape == (1, 4)
        assert target["labels"].shape == (1,)
        assert target["labels"][0] == 1  # 1 for 'cat', 0 is background

    finally:
        shutil.rmtree(temp_dir)


def test_training_run():
    """Tests that the training script runs for one epoch without errors."""
    temp_dir = setup_test_environment(num_images=5)
    output_dir = tempfile.mkdtemp()
    try:
        train_model(
            data_dir=temp_dir,
            output_dir=output_dir,
            num_epochs=5,
            batch_size=2,
            loader_workers=0,
            scheduler_mode=SchedulerMode.COSINE_ANNEALING,
        )
        # Check for output files
        assert (Path(output_dir) / "trailcam_classifier_model.pth").exists()
        assert (Path(output_dir) / "class_names.txt").exists()

    finally:
        shutil.rmtree(temp_dir)
        shutil.rmtree(output_dir)


def test_inference_run():
    """Tests that the inference script can classify an image."""
    temp_dir = setup_test_environment(num_images=5)
    output_dir = tempfile.mkdtemp()
    model_output_dir = tempfile.mkdtemp()
    try:
        # First, train a model
        train_model(
            data_dir=temp_dir,
            output_dir=model_output_dir,
            num_epochs=5,
            batch_size=2,
            loader_workers=0,
            scheduler_mode=SchedulerMode.COSINE_ANNEALING,
        )
        model_path = Path(model_output_dir) / "trailcam_classifier_model.pth"

        # Now, run inference
        config = ClassificationConfig(
            dirs=[temp_dir],
            model=str(model_path),
            output=output_dir,
            confidence_threshold=0.01,  # Lower threshold for test model
        )
        asyncio.run(run_classification(config))

        # Check that at least one file was moved to the correct directory
        classified_dir = Path(output_dir) / "cat"
        assert classified_dir.exists()
        assert len(list(classified_dir.glob("*.jpg"))) > 0

    finally:
        shutil.rmtree(temp_dir)
        shutil.rmtree(output_dir)
        shutil.rmtree(model_output_dir)


if __name__ == "__main__":
    print("Running test_dataset_loading...")
    test_dataset_loading()
    print("OK")

    print("\nRunning test_training_run...")
    test_training_run()
    print("OK")

    print("\nRunning test_inference_run...")
    test_inference_run()
    print("OK")

    print("\nAll tests passed!")
