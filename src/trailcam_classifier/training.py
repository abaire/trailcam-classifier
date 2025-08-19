from __future__ import annotations

# ruff: noqa: T201 `print` found
import argparse
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from torch import device, nn, optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch_lr_finder import LRFinder
from torchvision.models import EfficientNet, EfficientNet_V2_S_Weights, efficientnet_v2_s
from tqdm import tqdm

from trailcam_classifier.util import MODEL_SAVE_FILENAME, find_images, get_best_device

logger = logging.getLogger(__name__)

weights = EfficientNet_V2_S_Weights.DEFAULT
data_transform = weights.transforms()


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


class PreprocessedDataset(Dataset):
    @dataclass
    class SampleInfo:
        image_path: Path
        short_path: Path
        tensor_path: Path
        class_index: int

    def __init__(self, root_dir: str | os.PathLike, data_transform: Any):
        self.raw_images = find_images([root_dir])
        relative_images = [image.relative_to(root_dir) for image in self.raw_images]

        self.class_names = sorted({image_file.parts[0] for image_file in relative_images})
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.samples = []
        for raw_path, rel_path in zip(self.raw_images, relative_images):
            class_name = rel_path.parts[0]
            class_idx = self.class_to_idx[class_name]
            target_path = raw_path.with_suffix(".pt")
            self.samples.append(PreprocessedDataset.SampleInfo(raw_path, rel_path, target_path, class_idx))

        for sample in tqdm(self.samples, desc="Preprocessing Images"):
            if sample.tensor_path.is_file():
                continue

            sample.tensor_path.parent.mkdir(parents=True, exist_ok=True)

            image = Image.open(sample.image_path).convert("RGB")
            tensor = data_transform(image)
            torch.save(tensor, sample.tensor_path)

    @property
    def classes(self) -> list[str]:
        return self.class_names

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[Any, int]:
        sample: PreprocessedDataset.SampleInfo = self.samples[idx]
        tensor = torch.load(sample.tensor_path)
        return tensor, sample.class_index

    def get_deterministic_split(self, val_split_ratio: float, artifact_path: str = "validation_set.txt"):
        """
        Creates a deterministic train/validation split.

        Loads the validation set from an artifact file if it exists.
        Otherwise, it creates a new split and saves the validation file paths to the artifact.
        """

        if os.path.exists(artifact_path):
            # TODO: This is incorrect, it'll never grow the validation set with newly added files.
            logger.info("Loading validation set from artifact: %s", artifact_path)
            with open(artifact_path, encoding="utf-8") as infile:
                val_files = {Path(line.strip()) for line in infile}
            logger.debug("Loaded %d entries", len(val_files))

            train_indices, val_indices = [], []
            for i, sample in enumerate(self.samples):
                if sample.short_path in val_files:
                    val_indices.append(i)
                else:
                    train_indices.append(i)

            train_subset = Subset(self, train_indices)
            val_subset = Subset(self, val_indices)

        else:
            logger.info("Creating new validation set and saving to artifact: %s", artifact_path)
            num_samples = len(self)
            val_size = int(val_split_ratio * num_samples)
            train_size = num_samples - val_size

            generator = torch.Generator().manual_seed(42)
            train_subset, val_subset = random_split(self, [train_size, val_size], generator=generator)

            with open(artifact_path, "w") as f:
                for idx in val_subset.indices:
                    filepath = self.samples[idx].short_path
                    f.write(f"{filepath}\n")

        return train_subset, val_subset


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


def _create_model(num_classes: int) -> tuple[device, EfficientNet]:
    model = efficientnet_v2_s(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    for block in list(model.features.children())[-2:]:
        for param in block.parameters():
            param.requires_grad = True

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
) -> tuple[int, float]:
    start_epoch = 0
    best_val_loss = float("inf")
    if os.path.exists(model_save_path):
        logger.info("Loading checkpoint from %s", model_save_path)
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint["model_state_dict"])
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
        for inputs_raw, labels_raw in train_loader:
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
            for inputs_raw, labels_raw in validation_loader:
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
            # Instead of just weights, we save the entire training state
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
    patience: int = 8,
    loader_workers: int = 8,
    batch_size: int = 128,
    scheduler_mode: SchedulerMode = SchedulerMode.COSINE_ANNEALING,
    *,
    find_lr: bool = False,
):
    """Loads data, fine-tunes a pretrained model, and trains with early stopping."""
    dataset = PreprocessedDataset(data_dir, data_transform)

    if not output_dir:
        output_dir = "."
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    class_names = dataset.classes
    num_classes = len(class_names)
    logger.info("Found %d images in %d classes: %s", len(dataset), num_classes, sorted(class_names))

    dev, model = _create_model(num_classes)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    logger.info("Using device: %s", dev)

    if find_lr:
        train_dataset = dataset
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=loader_workers, pin_memory=True
        )
        _run_lr_finder(model, optimizer, criterion, dev, train_loader)
        return

    train_dataset, val_dataset = dataset.get_deterministic_split(val_split_ratio=0.2)
    if not val_dataset:
        logger.error("Validation set is empty. Check your data distribution or split ratio. Aborting.")
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
        steps_per_epoch = len(train_loader) // batch_size
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, steps_per_epoch=steps_per_epoch, epochs=num_epochs
        )
    else:
        msg = f"Invalid scheduler mode {scheduler_mode}"
        raise ValueError(msg)

    model_save_path = os.path.join(output_dir, MODEL_SAVE_FILENAME)
    start_epoch, best_val_loss = _load_checkpoint(model_save_path, model, optimizer, scheduler)

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

    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))
    print("Class names saved to class_names.txt")


def main():
    parser = argparse.ArgumentParser(description="Train an image classifier.")
    parser.add_argument("data_dir", type=str, help="Directory containing the classified image folders.")
    parser.add_argument("--learning-rate", "-L", default=1.0e-3, type=float, help="Initial learning rate")
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
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))

    train_model(
        data_dir=data_dir,
        output_dir=args.output,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        find_lr=args.find_lr,
        patience=args.patience,
        scheduler_mode=SchedulerMode.from_string(args.scheduler_mode),
    )


if __name__ == "__main__":
    main()
