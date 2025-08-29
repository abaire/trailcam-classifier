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
from typing import Any, Counter

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from torch import device, nn, optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch_lr_finder import LRFinder
from torchvision import transforms
from torchvision.models import EfficientNet, EfficientNet_V2_S_Weights, efficientnet_v2_s
from tqdm import tqdm

from trailcam_classifier.util import (
    MODEL_SAVE_FILENAME,
    NORMALIZATION,
    CropInfoBar,
    find_images,
    get_best_device,
    get_classification_transforms,
    slice_with_feature,
)

logger = logging.getLogger(__name__)

weights = EfficientNet_V2_S_Weights.DEFAULT


def get_transforms(
    augmentation_strength: float = 1.0,
) -> tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """
    Returns a tuple of transforms for preprocessing, training, and validation.

    Augmentation strength parameter scales the intensity of the augmentations.
    """
    image_size = 384  # From EfficientNetV2-S spec

    # Deterministic transforms for caching.
    # We resize to a square, which may distort aspect ratio, but guarantees
    # that the entire image is visible to the model, which is critical for
    # images where the subject may be small and not in the center.
    preprocess_transform = transforms.Compose(
        [
            CropInfoBar(),
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        ]
    )

    # Augmentations applied to the cached images during training
    train_augment_transform = transforms.Compose(
        [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2 * augmentation_strength,
                contrast=0.2 * augmentation_strength,
                saturation=0.2 * augmentation_strength,
            ),
            transforms.ToTensor(),
            NORMALIZATION,
        ]
    )

    # Full set of transforms for validation, to be cached as tensors
    val_transform = get_classification_transforms()

    return preprocess_transform, train_augment_transform, val_transform


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
        class_index: int

    def __init__(self, root_dir: str | os.PathLike):
        self.root_dir = Path(root_dir)
        self.raw_images = find_images([self.root_dir])
        relative_images = [image.relative_to(self.root_dir) for image in self.raw_images]

        self.class_names = sorted({image_file.parts[0] for image_file in relative_images})
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.samples = []
        for raw_path, rel_path in zip(self.raw_images, relative_images):
            class_name = rel_path.parts[0]
            class_idx = self.class_to_idx[class_name]
            self.samples.append(ImageDataset.SampleInfo(raw_path, rel_path, class_idx))

    @property
    def classes(self) -> list[str]:
        return self.class_names

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[Any, int]:
        sample: ImageDataset.SampleInfo = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")

        json_path = sample.image_path.with_suffix(".json")
        if json_path.exists():
            with json_path.open() as f:
                feature_coords = json.load(f)
            image = slice_with_feature(image, feature_coords)

        return image, sample.class_index

    def get_deterministic_split(self, val_split_ratio: float, artifact_path: str = "split_artifact.json"):
        """
        Creates a deterministic train/validation split that is stable across runs.

        If an artifact file exists, it loads the split from there. Any new files
        not in the artifact are split and added to the existing sets, and the
        artifact is updated.

        If no artifact exists, a new split is created and saved.
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


class CachingPilDataset(Dataset):
    """
    A dataset that preprocesses and caches images as PIL Image objects.

    This dataset wraps a Subset, applies a series of transforms to each image,
    and saves the result to a cache directory. On subsequent accesses, it loads
    the preprocessed image from the cache, avoiding repeated transformations.
    This is useful for caching the deterministic part of a transformation pipeline.
    """

    def __init__(self, subset: Subset, cache_dir: str | os.PathLike, transform: transforms.Compose):
        self.subset = subset
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._precache_images()

    def _precache_images(self):
        logger.info("Precaching PIL images to %s...", self.cache_dir)
        for i in tqdm(range(len(self.subset)), desc=f"Caching to {self.cache_dir.name}"):
            self._cache_item(i)

    def _get_cache_path(self, original_index: int) -> Path:
        sample_info: ImageDataset.SampleInfo = self.subset.dataset.samples[original_index]
        relative_path = sample_info.short_path
        return (self.cache_dir / relative_path).with_suffix(".png")

    def _cache_item(self, index: int):
        original_index = self.subset.indices[index]
        cache_path = self._get_cache_path(original_index)

        if not cache_path.exists():
            image, _ = self.subset.dataset[original_index]  # Load original image
            image = self.transform(image)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(cache_path, format="PNG")

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        original_index = self.subset.indices[index]
        cache_path = self._get_cache_path(original_index)

        image = Image.open(cache_path).convert("RGB")
        label = self.subset.dataset.samples[original_index].class_index
        return image, label


class CachingTensorDataset(Dataset):
    """
    A dataset that preprocesses and caches images as PyTorch Tensors.

    This dataset wraps a Subset, applies a series of transforms to each image,
    and saves the resulting tensor to a cache directory. On subsequent accesses,
    it loads the tensor from the cache, avoiding repeated transformations. This
    is useful for caching a fully deterministic transformation pipeline.
    """

    def __init__(self, subset: Subset, cache_dir: str | os.PathLike, transform: transforms.Compose):
        self.subset = subset
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._precache_tensors()

    def _precache_tensors(self):
        logger.info("Precaching tensors to %s...", self.cache_dir)
        for i in tqdm(range(len(self.subset)), desc=f"Caching to {self.cache_dir.name}"):
            self._cache_item(i)

    def _get_cache_path(self, original_index: int) -> Path:
        sample_info: ImageDataset.SampleInfo = self.subset.dataset.samples[original_index]
        relative_path = sample_info.short_path
        return (self.cache_dir / relative_path).with_suffix(".pt")

    def _cache_item(self, index: int):
        original_index = self.subset.indices[index]
        cache_path = self._get_cache_path(original_index)

        if not cache_path.exists():
            image, _ = self.subset.dataset[original_index]  # Load original image
            tensor = self.transform(image)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensor, cache_path)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        original_index = self.subset.indices[index]
        cache_path = self._get_cache_path(original_index)

        tensor = torch.load(cache_path)
        label = self.subset.dataset.samples[original_index].class_index
        return tensor, label


class DatasetTransformer(Dataset):
    """Applies a transform to a dataset."""

    def __init__(self, dataset: Dataset, transform: transforms.Compose):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)


def _calculate_class_weights(dataset: Subset, num_classes: int) -> torch.Tensor:
    """Calculates class weights based on inverse frequency for a subset of data."""
    labels = [dataset.dataset.samples[i].class_index for i in dataset.indices]
    class_counts = Counter(labels)

    counts = [class_counts.get(i, 0) for i in range(num_classes)]

    total_samples = sum(counts)
    weights = []
    for count in counts:
        if count == 0:
            weights.append(0)
        else:
            weight = total_samples / (num_classes * count)
            weights.append(weight)

    return torch.tensor(weights, dtype=torch.float)


def _run_lr_finder(
    model: EfficientNet,
    optimizer,
    criterion,
    device: device,
    train_loader: DataLoader,
):
    """Runs the learning rate finder, plots the results, and suggests a rate."""
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
    lr_finder.plot(log_lr=True)

    lr_finder.reset()


def _set_trainable_layers(model: EfficientNet, trainable_layers: int):
    for param in model.features.parameters():
        param.requires_grad = False

    for block in list(model.features.children())[-trainable_layers:]:
        for param in block.parameters():
            param.requires_grad = True


def _create_model(num_classes: int, trainable_layers: int) -> tuple[device, EfficientNet]:
    model = efficientnet_v2_s(weights=weights)

    _set_trainable_layers(model, trainable_layers)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    device = get_best_device()
    model.to(device)

    return device, model


def _load_checkpoint(
    model_save_path: str,
    model: EfficientNet,
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
    model: EfficientNet,
    optimizer,
    scheduler,
    criterion,
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

    all_preds_epoch, all_labels_epoch = [], []

    class Timer:
        """A context manager to time a block of code and print the duration."""

        def __init__(self, description: str):
            self.description = description
            self.start_time = None

        def __enter__(self):
            """Starts the timer when entering the 'with' block."""
            self.start_time = time.perf_counter()

        def __exit__(self, exc_type, exc_value, traceback):
            """Stops the timer and prints the elapsed time when exiting the block."""
            end_time = time.perf_counter()
            elapsed_ms = (end_time - self.start_time) * 1000
            logger.debug("'%s' took %s.3f ms", self.description, elapsed_ms)

    for epoch in range(start_epoch, max_epochs):
        logger.info("Epoch %d/%d begin...", epoch + 1, max_epochs)

        with Timer("model.train()"):
            model.train()
        running_train_loss = 0.0
        for inputs_raw, labels_raw in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            with Timer("inputs, labels = inputs_raw.to(dev), labels_raw.to(dev)"):
                inputs, labels = inputs_raw.to(dev), labels_raw.to(dev)

            with Timer("optimizer.zero_grad()"):
                optimizer.zero_grad()
            with Timer("outputs = model(inputs)"):
                outputs = model(inputs)
            with Timer("loss = criterion(outputs, labels)"):
                loss = criterion(outputs, labels)
            with Timer("loss.backward()"):
                loss.backward()
            with Timer("optimizer.step()"):
                optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

            if scheduler_mode == SchedulerMode.ONECYCLE:
                scheduler.step()

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        with Timer("model.eval()"):
            model.eval()
        running_val_loss = 0.0
        all_preds_epoch, all_labels_epoch = [], []
        with torch.no_grad():
            for inputs_raw, labels_raw in tqdm(validation_loader, desc=f"Epoch {epoch + 1} Validation"):
                inputs, labels = inputs_raw.to(dev), labels_raw.to(dev)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds_epoch.extend(preds.cpu().numpy())
                all_labels_epoch.extend(labels.cpu().numpy())

        epoch_val_loss = running_val_loss / len(validation_loader.dataset)
        accuracy = accuracy_score(all_labels_epoch, all_preds_epoch)

        if scheduler_mode == SchedulerMode.PLATEAU:
            scheduler.step(epoch_val_loss)
        elif scheduler_mode == SchedulerMode.COSINE_ANNEALING:
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %d/%d -> " "Train Loss: %.4f, " "Val Loss: %.4f, " "Val Accuracy: %.4f, " "LR: %.6f",
            epoch + 1,
            max_epochs,
            epoch_train_loss,
            epoch_val_loss,
            accuracy,
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

    return all_labels_epoch, all_preds_epoch, best_checkpoint_data


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
    trainable_layers: int = 3,
    augmentation_strength: float = 1.0,
    *,
    find_lr: bool = False,
    restart_scheduler: bool = False,
):
    """Loads data, fine-tunes a pretrained model, and trains with early stopping."""
    base_dataset = ImageDataset(data_dir)

    if not output_dir:
        output_dir = "."
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = Path(output_dir) / "preprocessed_cache"

    class_names = base_dataset.classes
    num_classes = len(class_names)
    logger.info("Found %d images in %d classes: %s", len(base_dataset), num_classes, sorted(class_names))

    dev, model = _create_model(num_classes, trainable_layers)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    logger.info("Using device: %s", dev)

    train_subset, val_subset = base_dataset.get_deterministic_split(
        val_split_ratio=0.2, artifact_path=os.path.join(output_dir, "training_split.json")
    )
    if not val_subset:
        logger.error("Validation set is empty. Check your data distribution or split ratio. Aborting.")
        return

    class_weights = _calculate_class_weights(train_subset, num_classes).to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    preprocess_transform, train_augment_transform, val_transform = get_transforms(augmentation_strength)

    train_pil_dataset = CachingPilDataset(train_subset, cache_dir / "train", preprocess_transform)
    train_dataset = DatasetTransformer(train_pil_dataset, train_augment_transform)

    val_dataset = CachingTensorDataset(val_subset, cache_dir / "val", val_transform)

    if find_lr:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=loader_workers, pin_memory=True
        )
        _run_lr_finder(model, optimizer, criterion, dev, train_loader)
        return

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=loader_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers, pin_memory=True
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

    if start_epoch:
        _set_trainable_layers(model, trainable_layers)

    all_labels, all_preds, best_checkpoint = _run_training(
        dev,
        model,
        optimizer,
        scheduler,
        criterion,
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

    active_labels = sorted(set(all_labels))
    active_class_names = [class_names[i] for i in active_labels]
    report = classification_report(
        all_labels, all_preds, labels=active_labels, target_names=active_class_names, zero_division=0
    )
    print(report)

    class_names_path = os.path.join(output_dir, "class_names.txt")
    with open(class_names_path, "w") as f:
        f.write("\n".join(class_names))
    print(f"Class names saved to {class_names_path}")


def main():
    parser = argparse.ArgumentParser(description="Train an image classifier.")
    parser.add_argument("data_dir", type=str, help="Directory containing the classified image folders.")
    parser.add_argument("--learning-rate", "-r", default=1.0e-3, type=float, help="Initial learning rate")
    parser.add_argument("--weight-decay", "-w", default=0.01, type=float, help="Weight decay for the AdamW optimizer.")
    parser.add_argument(
        "--find-lr", action="store_true", help="Run the learning rate finder instead of a full training run."
    )
    parser.add_argument("--output", "-o", help="Directory into which trained model outputs will be written.")
    parser.add_argument(
        "--patience", type=int, default=8, help="Maximum number of epochs without improvement before early exit."
    )
    parser.add_argument("--batch-size", "-b", default=128, type=int, help="Batch size for training.")
    parser.add_argument("--epochs", "-e", default=1000, type=int, help="Maximum number of epochs to train.")
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
        "--trainable-layers", "-L", default=3, type=int, help="Number of layers to unfreeze for training."
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
        find_lr=args.find_lr,
        patience=args.patience,
        scheduler_mode=SchedulerMode.from_string(args.scheduler_mode),
        restart_scheduler=args.restart_scheduler,
        trainable_layers=args.trainable_layers,
        augmentation_strength=args.augmentation_strength,
    )


if __name__ == "__main__":
    main()
