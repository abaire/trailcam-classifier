import json
import random
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

from trailcam_classifier.import_dataset import (
    _discover_images,
    _group_new_images,
    _move_entries,
    _process_metadata_files,
    convert_bbox_to_yolo,
    main,
)


def test_main(tmp_path: Path, monkeypatch):
    # Create source directory with dummy files
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for i in range(10):
        img_path = source_dir / f"img{i}.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)
        json_path = source_dir / f"img{i}.json"
        class_name = "cat" if i < 5 else "dog"
        json_path.write_text(json.dumps({class_name: [{"x1": 10, "y1": 10, "x2": 20, "y2": 20}]}))

    output_dir = tmp_path / "output"

    # Patch sys.argv
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import_dataset.py",
            str(source_dir),
            "--dataset-dir",
            str(output_dir),
            "--val-split",
            "0.4",
        ],
    )

    main()

    # Check directory structure
    assert (output_dir / "train" / "images").exists()
    assert (output_dir / "train" / "labels").exists()
    assert (output_dir / "val" / "images").exists()
    assert (output_dir / "val" / "labels").exists()
    assert (output_dir / "data.yaml").exists()

    # Check file counts
    assert len(list((output_dir / "train" / "images").glob("*.jpg"))) == 6
    assert len(list((output_dir / "train" / "labels").glob("*.txt"))) == 6
    assert len(list((output_dir / "val" / "images").glob("*.jpg"))) == 4
    assert len(list((output_dir / "val" / "labels").glob("*.txt"))) == 4

    # Check data.yaml
    with open(output_dir / "data.yaml") as f:
        data = yaml.safe_load(f)
        assert data["path"] == str(output_dir.resolve())
        assert data["train"] == "train"
        assert data["val"] == "val"
        assert data["names"] == {0: "cat", 1: "dog"}


def test_move_entries(tmp_path: Path):
    # Create source directory with dummy files
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    train_img = source_dir / "train.jpg"
    train_txt = source_dir / "train.txt"
    val_img = source_dir / "val.jpg"
    val_txt = source_dir / "val.txt"
    train_img.touch()
    train_txt.touch()
    val_img.touch()
    val_txt.touch()

    output_dir = tmp_path / "output"
    _move_entries({train_img}, {val_img}, output_dir)

    # Check that files were moved
    assert (output_dir / "train" / "images" / "train.jpg").exists()
    assert (output_dir / "train" / "labels" / "train.txt").exists()
    assert (output_dir / "val" / "images" / "val.jpg").exists()
    assert (output_dir / "val" / "labels" / "val.txt").exists()

    # Check that original files are gone
    assert not train_img.exists()
    assert not train_txt.exists()
    assert not val_img.exists()
    assert not val_txt.exists()


def test_group_new_images(tmp_path: Path):
    # Create dummy image and json files
    img_paths = []
    for i in range(10):
        img_path = tmp_path / f"img{i}.jpg"
        img_path.touch()
        img_paths.append(img_path)
        json_path = tmp_path / f"img{i}.json"
        class_name = "cat" if i < 5 else "dog"
        json_path.write_text(json.dumps({class_name: []}))

    random.seed(42)
    train_paths, val_paths = _group_new_images(img_paths, val_split=0.4)

    assert len(train_paths) == 6
    assert len(val_paths) == 4
    assert len(train_paths.intersection(val_paths)) == 0
    assert len(train_paths.union(val_paths)) == 10


def test_process_metadata_files(tmp_path: Path):
    # Create a dummy image file
    img_path = tmp_path / "img1.jpg"
    img = Image.new("RGB", (100, 200), color="red")
    img.save(img_path)

    # Create a dummy json file
    json_path = tmp_path / "img1.json"
    json_path.write_text(
        json.dumps(
            {
                "cat": [{"x1": 10, "y1": 20, "x2": 60, "y2": 120}],
            }
        )
    )

    class_to_idx = {"cat": 0, "dog": 1}
    labeled_image_paths = _process_metadata_files([img_path], class_to_idx)

    assert labeled_image_paths == [img_path]
    txt_path = tmp_path / "img1.txt"
    assert txt_path.exists()
    with open(txt_path) as f:
        content = f.read().strip()
        parts = content.split()
        assert len(parts) == 5
        assert int(parts[0]) == 0
        assert float(parts[1]) == pytest.approx(0.35)
        assert float(parts[2]) == pytest.approx(0.35)
        assert float(parts[3]) == pytest.approx(0.5)
        assert float(parts[4]) == pytest.approx(0.5)


def test_discover_images(tmp_path: Path):
    # Create some dummy files
    (tmp_path / "img1.jpg").touch()
    (tmp_path / "img1.json").write_text(json.dumps({"cat": [], "dog": []}))
    (tmp_path / "img2.jpg").touch()
    (tmp_path / "img2.json").write_text(json.dumps({"dog": [], "bird": []}))
    (tmp_path / "img3.txt").touch()  # Should be ignored

    all_image_paths, class_names = _discover_images([str(tmp_path)])

    assert all_image_paths == {tmp_path / "img1.jpg", tmp_path / "img2.jpg"}
    assert class_names == ["bird", "cat", "dog"]


def test_convert_bbox_to_yolo():
    img_width = 1920
    img_height = 1080
    bbox = {"x1": 480, "y1": 270, "x2": 1440, "y2": 810}
    x_center_norm, y_center_norm, width_norm, height_norm = convert_bbox_to_yolo(img_width, img_height, bbox)
    assert x_center_norm == pytest.approx(0.5)
    assert y_center_norm == pytest.approx(0.5)
    assert width_norm == pytest.approx(0.5)
    assert height_norm == pytest.approx(0.5)


def test_main_update(tmp_path: Path, monkeypatch):
    # Create an existing dataset directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create an initial data.yaml
    initial_yaml_path = output_dir / "data.yaml"
    initial_classes = {0: "cat", 1: "dog"}
    with open(initial_yaml_path, "w") as f:
        yaml.dump(
            {
                "path": str(output_dir.resolve()),
                "train": "train",
                "val": "val",
                "names": initial_classes,
            },
            f,
        )

    # Create source directory with new images and a new class
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for i in range(5):
        img_path = source_dir / f"img{i}.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)
        json_path = source_dir / f"img{i}.json"
        class_name = "bird"
        json_path.write_text(json.dumps({class_name: [{"x1": 10, "y1": 10, "x2": 20, "y2": 20}]}))

    # Patch sys.argv and run main
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import_dataset.py",
            str(source_dir),
            "--dataset-dir",
            str(output_dir),
        ],
    )

    main()

    # Check that the new class was added and old classes are preserved.
    with open(output_dir / "data.yaml") as f:
        data = yaml.safe_load(f)
        assert data["names"] == {0: "cat", 1: "dog", 2: "bird"}
