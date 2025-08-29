from __future__ import annotations

# ruff: noqa: T201 `print` found
# ruff: noqa: PLR2004 Magic value used in comparison,
# ruff: noqa: S311 Standard pseudo-random generators are not suitable for cryptographic purposes
import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import device, nn, optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import tv_tensors
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from trailcam_classifier.util import (
    MODEL_SAVE_FILENAME,
    find_images,
    get_best_device,
    get_object_detection_transforms,
)

logger = logging.getLogger(__name__)


class SchedulerMode(Enum):
    COSINE_ANNEALING = auto()
    PLATEAU = auto()
    ONECYCLE = auto()

    @classmethod
    def from_string(cls, val: str):
        if val.lower() == "cosine_annealing":
            return cls.COSINE_ANNEALING
        if val.lower() == "plateau":
            return cls.PLATEAU
        if val.lower() == "onecycle":
            return cls.ONECYCLE

        msg = f"Invalid enum value {val}"
        raise ValueError(msg)


class ImageDataset(Dataset):
    @dataclass
    class SampleInfo:
        image_path: Path
        short_path: Path

    def __init__(self, root_dir: str | os.PathLike):
        self.root_dir = Path(root_dir)
        self.raw_images = find_images([self.root_dir])
        relative_images = [image.relative_to(self.root_dir) for image in self.raw_images]

        self.samples = []
        for raw_path, rel_path in zip(self.raw_images, relative_images):
            self.samples.append(ImageDataset.SampleInfo(raw_path, rel_path))

        # Discover classes from JSON files
        self.class_names = self._discover_classes()
        self.class_to_idx = {name: i + 1 for i, name in enumerate(self.class_names)}  # 0 is background

    def _discover_classes(self) -> list[str]:
        """Scans all JSON files to find the set of all class names."""
        class_names = set()
        for sample in tqdm(self.samples, desc="Discovering classes"):
            json_path = sample.image_path.with_suffix(".json")
            if json_path.exists():
                with json_path.open() as f:
                    data = json.load(f)
                    class_names.update(data.keys())
        return sorted(list(class_names))

    @property
    def classes(self) -> list[str]:
        return self.class_names

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[Any, dict[str, Any]]:
        sample: ImageDataset.SampleInfo = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        json_path = sample.image_path.with_suffix(".json")

        boxes = []
        labels = []
        if json_path.exists():
            with json_path.open() as f:
                annotations = json.load(f)
            for class_name, regions in annotations.items():
                for region in regions:
                    boxes.append([region["x1"], region["y1"], region["x2"], region["y2"]])
                    labels.append(self.class_to_idx[class_name])

        image = tv_tensors.Image(image)

        target = {}
        if boxes:
            target["boxes"] = tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=image.size, dtype=torch.float32
            )
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["image_id"] = torch.tensor([idx])
            target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                target["boxes"][:, 2] - target["boxes"][:, 0]
            )
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            # Handle images with no objects
            target["boxes"] = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4)), format="XYXY", canvas_size=image.size, dtype=torch.float32
            )
            target["labels"] = torch.zeros(0, dtype=torch.int64)
            target["image_id"] = torch.tensor([idx])
            target["area"] = torch.zeros(0, dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        return image, target

    def get_deterministic_split(self, val_split_ratio: float, artifact_path: str = "split_artifact.json"):
        """
        Creates a deterministic train/validation split that is stable across runs.
        """
        if os.path.exists(artifact_path):
            logger.info("Loading train/val split from artifact: %s", artifact_path)
            try:
                with open(artifact_path, encoding="utf-8") as infile:
                    split_data = json.load(infile)
                train_paths = {Path(p) for p in split_data["train"]}
                val_paths = {Path(p) for p in split_data["val"]}
            except (json.JSONDecodeError, KeyError):
                logger.warning("Could not read split artifact file. Creating a new split.")
                os.remove(artifact_path)
                train_paths, val_paths = set(), set()
        else:
            train_paths, val_paths = set(), set()

        if not train_paths and not val_paths:
            logger.info("Creating new train/val split and saving to artifact: %s", artifact_path)
            num_samples = len(self)
            val_size = int(val_split_ratio * num_samples)
            train_size = num_samples - val_size

            generator = torch.Generator().manual_seed(42)
            initial_train_subset, initial_val_subset = random_split(self, [train_size, val_size], generator=generator)

            train_paths = {self.samples[i].short_path for i in initial_train_subset.indices}
            val_paths = {self.samples[i].short_path for i in initial_val_subset.indices}

        all_current_paths = {s.short_path for s in self.samples}
        known_paths = train_paths.union(val_paths)
        new_paths = all_current_paths - known_paths

        if new_paths:
            logger.info("Found %d new files. Splitting and adding to existing sets.", len(new_paths))
            new_indices = [i for i, s in enumerate(self.samples) if s.short_path in new_paths]
            num_new = len(new_indices)
            val_size_new = int(val_split_ratio * num_new)

            rng = random.Random(42)
            rng.shuffle(new_indices)

            new_val_indices = new_indices[:val_size_new]
            new_train_indices = new_indices[val_size_new:]

            train_paths.update(self.samples[i].short_path for i in new_train_indices)
            val_paths.update(self.samples[i].short_path for i in new_val_indices)

        with open(artifact_path, "w") as f:
            json.dump(
                {"train": sorted([str(p) for p in train_paths]), "val": sorted([str(p) for p in val_paths])},
                f,
                indent=2,
            )

        final_train_indices = [i for i, s in enumerate(self.samples) if s.short_path in train_paths]
        final_val_indices = [i for i, s in enumerate(self.samples) if s.short_path in val_paths]

        train_subset = Subset(self, final_train_indices)
        val_subset = Subset(self, final_val_indices)

        return train_subset, val_subset


class DatasetTransformer(Dataset):
    """Applies a transform to a dataset."""

    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.dataset[index]
        if self.transform:
            img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        return len(self.dataset)


def _create_model(num_classes: int) -> tuple[device, nn.Module]:
    # load a model pre-trained on COCO
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = get_best_device()
    model.to(device)

    return device, model


def _load_checkpoint(
    model_save_path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    *,
    restart_scheduler: bool = False,
) -> tuple[int, float]:
    start_epoch = 0
    best_val_loss = float("inf")
    if os.path.exists(model_save_path):
        logger.info("Loading checkpoint from %s", model_save_path)
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        if not restart_scheduler:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        logger.info("Resuming training from epoch %d with best val loss %.4f", start_epoch, best_val_loss)

    return start_epoch, best_val_loss


def _run_training(
    dev: device,
    model: nn.Module,
    optimizer,
    scheduler,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    max_epochs: int,
    patience: int,
    start_epoch: int,
    best_val_loss: float,
    scheduler_mode: SchedulerMode,
) -> tuple[list, list, dict | None]:
    epochs_without_improvement = 0
    best_checkpoint_data = None

    for epoch in range(start_epoch, max_epochs):
        logger.info("Epoch %d/%d begin...", epoch + 1, max_epochs)

        model.train()
        running_train_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            images = list(image.to(dev) for image in images)
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            running_train_loss += losses.item() * len(images)

            if scheduler_mode == SchedulerMode.ONECYCLE:
                scheduler.step()

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # Validation
        model.train()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(validation_loader, desc=f"Epoch {epoch + 1} Validation"):
                images = list(image.to(dev) for image in images)
                targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                running_val_loss += losses.item() * len(images)
        model.eval()

        epoch_val_loss = running_val_loss / len(validation_loader.dataset)

        if scheduler_mode == SchedulerMode.PLATEAU:
            scheduler.step(epoch_val_loss)
        elif scheduler_mode == SchedulerMode.COSINE_ANNEALING:
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %d/%d -> Train Loss: %.4f, Val Loss: %.4f, LR: %.6f",
            epoch + 1,
            max_epochs,
            epoch_train_loss,
            epoch_val_loss,
            current_lr,
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            best_checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict().copy(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }
            logger.info("✨ Validation loss improved to %.4f. Saving checkpoint.", best_val_loss)
        else:
            epochs_without_improvement += 1

        if patience and epochs_without_improvement >= patience:
            logger.info("\nEarly stopping triggered after %d epochs with no improvement.", patience)
            break

    return [], [], best_checkpoint_data


def collate_fn(batch):
    return tuple(zip(*batch))


def train_model(
    data_dir: str,
    output_dir: str | None = None,
    num_epochs: int = 1000,
    learning_rate: float = 0.0015,
    weight_decay: float = 0.01,
    patience: int = 8,
    loader_workers: int = 8,
    batch_size: int = 128,
    scheduler_mode: SchedulerMode = SchedulerMode.COSINE_ANNEALING,
    augmentation_strength: float = 1.0,
    *,
    restart_scheduler: bool = False,
):
    """Loads data, fine-tunes a pretrained model, and trains with early stopping."""
    base_dataset = ImageDataset(data_dir)

    if not output_dir:
        output_dir = "."
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    class_names = base_dataset.classes
    num_classes = len(class_names) + 1  # Add 1 for background
    logger.info("Found %d images in %d classes: %s", len(base_dataset), len(class_names), sorted(class_names))

    dev, model = _create_model(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    logger.info("Using device: %s", dev)

    train_subset, val_subset = base_dataset.get_deterministic_split(
        val_split_ratio=0.2, artifact_path=os.path.join(output_dir, "training_split.json")
    )
    if not val_subset:
        logger.error("Validation set is empty. Check your data distribution or split ratio. Aborting.")
        return

    train_dataset = DatasetTransformer(train_subset, get_object_detection_transforms(train=True))
    val_dataset = DatasetTransformer(val_subset, get_object_detection_transforms(train=False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    logger.info("Training on %d images, validating on %d images.", len(train_dataset), len(val_dataset))

    if scheduler_mode == SchedulerMode.COSINE_ANNEALING:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif scheduler_mode == SchedulerMode.PLATEAU:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=patience,
            min_lr=1e-6,
        )
    elif scheduler_mode == SchedulerMode.ONECYCLE:
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, steps_per_epoch=steps_per_epoch, epochs=num_epochs
        )
    else:
        msg = f"Invalid scheduler mode {scheduler_mode}"
        raise ValueError(msg)

    model_save_path = os.path.join(output_dir, MODEL_SAVE_FILENAME)
    start_epoch, best_val_loss = _load_checkpoint(
        model_save_path, model, optimizer, scheduler, restart_scheduler=restart_scheduler
    )

    _, _, best_checkpoint = _run_training(
        dev,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        num_epochs,
        patience,
        start_epoch,
        best_val_loss,
        scheduler_mode,
    )

    print("\n--- Final Validation Report ---")
    if best_checkpoint:
        print("Loading best model weights for final report and saving.")
        model.load_state_dict(best_checkpoint["model_state_dict"])

        torch.save(best_checkpoint, model_save_path)
        print(f"\nBest model checkpoint saved to {model_save_path}")
    elif os.path.exists(model_save_path):
        print("No new best model found. Retaining existing checkpoint file.")
    else:
        print("Training did not produce a valid model. Nothing to save.")
        return

    class_names_path = os.path.join(output_dir, "class_names.txt")
    with open(class_names_path, "w") as f:
        f.write("\n".join(class_names))
    print(f"Class names saved to {class_names_path}")


def main():
    parser = argparse.ArgumentParser(description="Train an image classifier.")
    parser.add_argument("data_dir", type=str, help="Directory containing the classified image folders.")
    parser.add_argument("--learning-rate", "-r", default=1.0e-3, type=float, help="Initial learning rate")
    parser.add_argument("--weight-decay", "-w", default=0.01, type=float, help="Weight decay for the AdamW optimizer.")
    parser.add_argument("--output", "-o", help="Directory into which trained model outputs will be written.")
    parser.add_argument(
        "--patience", type=int, default=8, help="Maximum number of epochs without improvement before early exit."
    )
    parser.add_argument("--batch-size", "-b", default=2, type=int, help="Batch size for training.")
    parser.add_argument("--epochs", "-e", default=100, type=int, help="Maximum number of epochs to train.")
    parser.add_argument(
        "--scheduler-mode",
        choices=["plateau", "onecycle", "cosine_annealing"],
        default="cosine_annealing",
        help="Scheduler type for optimizer.",
    )
    parser.add_argument(
        "--restart-scheduler",
        "-R",
        action="store_true",
        help="Do not reload the scheduler/optimizer from the checkpoint.",
    )
    parser.add_argument(
        "--augmentation-strength",
        "-A",
        default=1.0,
        type=float,
        help="Strength of the image augmentation (0.0 to disable).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))

    train_model(
        data_dir=data_dir,
        output_dir=args.output,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
        scheduler_mode=SchedulerMode.from_string(args.scheduler_mode),
        restart_scheduler=args.restart_scheduler,
        augmentation_strength=args.augmentation_strength,
    )


if __name__ == "__main__":
    main()
