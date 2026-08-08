"""
Train the Kigali e-waste classifier using an already-split dataset.

Expected dataset structure:

data/raw/modified-dataset/
├── train/
│   ├── Battery/
│   ├── Laptop/
│   └── ...
├── val/
│   ├── Battery/
│   ├── Laptop/
│   └── ...
└── test/
    ├── Battery/
    ├── Laptop/
    └── ...

Run from the project root:

python train_pre_split.py --data-dir data/raw/modified-dataset
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int, pretrained: bool = True):
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        num_classes,
    )
    return model


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder(
        data_dir / "train",
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        data_dir / "val",
        transform=eval_transform,
    )
    test_dataset = datasets.ImageFolder(
        data_dir / "test",
        transform=eval_transform,
    )

    if val_dataset.classes != train_dataset.classes:
        raise ValueError(
            "Validation classes do not match training classes.\n"
            f"Train: {train_dataset.classes}\n"
            f"Val:   {val_dataset.classes}"
        )

    if test_dataset.classes != train_dataset.classes:
        raise ValueError(
            "Test classes do not match training classes.\n"
            f"Train: {train_dataset.classes}\n"
            f"Test:  {test_dataset.classes}"
        )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes,
    )


def run_epoch(
    model,
    loader,
    loss_fn,
    device,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    targets = []
    predictions = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_fn(logits, labels)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().numpy())
        targets.extend(labels.detach().cpu().numpy())

    average_loss = total_loss / len(loader.dataset)
    accuracy = float(np.mean(np.asarray(predictions) == np.asarray(targets)))
    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    return average_loss, accuracy, macro_f1, targets, predictions


def save_confusion_matrix(
    targets,
    predictions,
    class_names,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(len(class_names))),
    )

    figure_size = max(8, len(class_names) * 0.65)
    plt.figure(figsize=(figure_size, figure_size))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=90,
    )
    plt.yticks(
        range(len(class_names)),
        class_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("E-waste classifier confusion matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Folder containing train, val, and test directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train without ImageNet pretrained weights.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_splits = ["train", "val", "test"]
    missing_splits = [
        split
        for split in required_splits
        if not (data_dir / split).is_dir()
    ]
    if missing_splits:
        raise FileNotFoundError(
            f"Missing dataset directories: {missing_splits}\n"
            f"Expected train, val, and test inside: {data_dir}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)
    print("Dataset:", data_dir.resolve())

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        class_names,
    ) = build_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Classes:", class_names)
    print("Train images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))

    class_counts = np.bincount(
        train_dataset.targets,
        minlength=len(class_names),
    )
    class_weights = len(train_dataset) / (
        len(class_names) * np.maximum(class_counts, 1)
    )
    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    print("Training images per class:")
    for class_name, count in zip(class_names, class_counts):
        print(f"  {class_name}: {count}")

    model = build_model(
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.3,
        patience=2,
    )

    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer,
        )
        val_loss, val_acc, val_f1, _, _ = run_epoch(
            model,
            val_loader,
            loss_fn,
            device,
        )

        scheduler.step(val_f1)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_loss:.4f}, "
            f"train acc={train_acc:.3f}, "
            f"train F1={train_f1:.3f} | "
            f"val loss={val_loss:.4f}, "
            f"val acc={val_acc:.3f}, "
            f"val F1={val_f1:.3f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "image_size": IMAGE_SIZE,
                    "model_name": "mobilenet_v3_small",
                },
                output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping.")
                break

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2)
    )

    checkpoint = torch.load(
        output_dir / "best.pt",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state"])

    _, test_accuracy, test_f1, targets, predictions = run_epoch(
        model,
        test_loader,
        loss_fn,
        device,
    )

    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report["overall"] = {
        "accuracy": test_accuracy,
        "macro_f1": test_f1,
    }

    (output_dir / "class_report.json").write_text(
        json.dumps(report, indent=2)
    )
    (output_dir / "class_counts.json").write_text(
        json.dumps(
            dict(zip(class_names, class_counts.tolist())),
            indent=2,
        )
    )
    save_confusion_matrix(
        targets,
        predictions,
        class_names,
        output_dir / "confusion_matrix.png",
    )

    print()
    print(f"Test accuracy: {test_accuracy:.3f}")
    print(f"Test macro F1: {test_f1:.3f}")
    print("Saved outputs to:", output_dir.resolve())


if __name__ == "__main__":
    main()

