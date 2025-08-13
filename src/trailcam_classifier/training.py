from __future__ import annotations

# ruff: noqa: T201 `print` found
import argparse
import os

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import device, nn, optim
from torch.utils.data import DataLoader, Subset, random_split
from torch_lr_finder import LRFinder
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet, EfficientNet_V2_S_Weights, efficientnet_v2_s

weights = EfficientNet_V2_S_Weights.DEFAULT
data_transform = weights.transforms()

MODEL_SAVE_FILENAME = "trailcam_classifier_model.pth"


def get_deterministic_split(dataset: ImageFolder, val_split_ratio: float, artifact_path: str = "validation_set.txt"):
    """
    Creates a deterministic train/validation split.

    Loads the validation set from an artifact file if it exists.
    Otherwise, it creates a new split and saves the validation file paths to the artifact.
    """
    if os.path.exists(artifact_path):
        print(f"Loading validation set from artifact: {artifact_path}")
        with open(artifact_path) as f:
            val_files = {line.strip() for line in f}

        train_indices, val_indices = [], []
        for i, (path, _) in enumerate(dataset.samples):
            if path in val_files:
                val_indices.append(i)
            else:
                train_indices.append(i)

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

    else:
        print(f"Creating new validation set and saving to artifact: {artifact_path}")
        num_samples = len(dataset)
        val_size = int(val_split_ratio * num_samples)
        train_size = num_samples - val_size

        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)

        with open(artifact_path, "w") as f:
            for idx in val_subset.indices:
                filepath = dataset.samples[idx][0]
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

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
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
        print(f"Loading checkpoint from {model_save_path}")
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resuming training from epoch {start_epoch} with best val loss {best_val_loss:.4f}")

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
) -> tuple[list, list, dict | None]:
    epochs_without_improvement = 0
    best_checkpoint_data = None

    all_preds_epoch, all_labels_epoch = [], []

    for epoch in range(start_epoch, max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs} begin...")

        model.train()
        running_train_loss = 0.0
        for inputs_raw, labels_raw in train_loader:
            inputs, labels = inputs_raw.to(dev), labels_raw.to(dev)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

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

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{max_epochs} -> "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}, "
            f"Val Accuracy: {accuracy:.4f}, "
            f"LR: {current_lr:.6f}"
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
            print(f"✨ Validation loss improved to {best_val_loss:.4f}. Saving checkpoint.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break

    return all_labels_epoch, all_preds_epoch, best_checkpoint_data


def train_model(
    data_dir: str,
    output_dir: str | None = None,
    num_epochs: int = 1000,
    learning_rate: float = 0.0015,
    patience: int = 8,
    loader_workers: int = 8,
    batch_size: int = 64,
    *,
    find_lr: bool = False,
):
    """Loads data, fine-tunes a pretrained model, and trains with early stopping."""
    dataset = ImageFolder(root=data_dir, transform=data_transform, allow_empty=True)
    if not dataset:
        print("No images found in any of the subdirectories. Exiting.")
        return

    if not output_dir:
        output_dir = "."
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    class_names = dataset.classes
    num_classes = len(class_names)
    print(f"Found {len(dataset)} images in {num_classes} classes: {class_names}")

    dev, model = _create_model(num_classes)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    print(f"Using device: {dev}")

    if find_lr:
        train_dataset = dataset
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=loader_workers, pin_memory=True
        )
        _run_lr_finder(model, optimizer, criterion, dev, train_loader)
        return

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    train_dataset, val_dataset = get_deterministic_split(dataset, val_split_ratio=0.2)
    if not val_dataset:
        print("Validation set is empty. Check your data distribution or split ratio. Aborting.")
        return

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=loader_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers, pin_memory=True
    )

    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

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
    parser.add_argument("--learning-rate", "-L", default=4.0e-3, type=float, help="Initial learning rate")
    parser.add_argument(
        "--find-lr", action="store_true", help="Run the learning rate finder instead of a full training run."
    )
    parser.add_argument("--output", "-o", help="Directory into which trained model outputs will be written.")
    args = parser.parse_args()

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))

    train_model(data_dir=data_dir, output_dir=args.output, learning_rate=args.learning_rate, find_lr=args.find_lr)


if __name__ == "__main__":
    main()
