import os
import time
import json
import csv

import torch
from torch.utils.data import DataLoader

from config import (
    DATASETS,
    TRAIN_SITES,
    TEST_SITES,
    MODEL_OUTPUT_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    ENCODER_CHANNELS,
    DECODER_CHANNELS,
    BOTTLENECK_CHANNELS,
    OUTPUT_CHANNELS,
    DEVICE,
    NUM_WORKERS,
    AUGMENT_TRAIN
)
from model import UNet
from dataset import list_tiles, train_val_split, estimate_positive_ratio, LandslideDataset
from losses import BCEDiceLoss
from metrics import binary_metrics, raw_confusion_counts
from plotting import plot_training_curves, plot_confusion_matrix

MAX_POS_WEIGHT = 50.0


def build_dataloaders(dataset_name):
    pairs = list_tiles(dataset_name, TRAIN_SITES)

    if not pairs:
        raise RuntimeError(f"No tiles found for dataset: {dataset_name}")

    train_pairs, val_pairs = train_val_split(pairs)

    train_dataset = LandslideDataset(train_pairs, augment=AUGMENT_TRAIN)
    val_dataset = LandslideDataset(val_pairs, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda")
    )

    print(f"  Tiles -> train: {len(train_pairs)}, val: {len(val_pairs)}")

    return train_loader, val_loader, train_pairs


def run_epoch(model, loader, criterion, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
    num_batches = 0

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            if is_training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_metrics = binary_metrics(logits, masks)

            total_loss += loss.item()
            for key, value in batch_metrics.items():
                total_metrics[key] += value

            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}

    return avg_loss, avg_metrics


def evaluate_confusion(model, loader):
    """Aggregates raw TP/FP/FN/TN pixel counts across an entire loader,
    used for the final confusion matrix plot."""

    model.eval()
    total_tp = total_fp = total_fn = total_tn = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            tp, fp, fn, tn = raw_confusion_counts(logits, masks)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    return total_tp, total_fp, total_fn, total_tn


def save_history_csv(history, path):
    if not history:
        return

    fieldnames = list(history[0].keys())

    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def train_on_dataset(dataset_name, in_channels):
    print(f"\n=== Training on dataset: {dataset_name} ({in_channels} input channels) ===")

    train_loader, val_loader, train_pairs = build_dataloaders(dataset_name)

    positive_ratio = estimate_positive_ratio(train_pairs)

    if positive_ratio > 0:
        pos_weight = min((1.0 - positive_ratio) / positive_ratio, MAX_POS_WEIGHT)
    else:
        pos_weight = 1.0

    print(f"  Positive pixel ratio: {positive_ratio:.5f} -> pos_weight: {pos_weight:.2f}")

    model = UNet(
        in_channels=in_channels,
        out_channels=OUTPUT_CHANNELS,
        encoder_channels=ENCODER_CHANNELS,
        decoder_channels=DECODER_CHANNELS,
        bottleneck_channels=BOTTLENECK_CHANNELS
    ).to(DEVICE)

    criterion = BCEDiceLoss(pos_weight=pos_weight).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    output_dir = os.path.join(MODEL_OUTPUT_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    best_val_iou = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion)

        elapsed = time.time() - start_time

        print(
            f"[{dataset_name}] Epoch {epoch}/{EPOCHS} ({elapsed:.1f}s) | "
            f"train_loss={train_loss:.4f} train_iou={train_metrics['iou']:.4f} | "
            f"val_loss={val_loss:.4f} val_iou={val_metrics['iou']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()}
        })

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"  -> new best val IoU: {best_val_iou:.4f}, checkpoint saved")

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))

    with open(os.path.join(output_dir, "history.json"), "w") as history_file:
        json.dump(history, history_file, indent=2)

    save_history_csv(history, os.path.join(output_dir, "history.csv"))
    plot_training_curves(history, output_dir)

    # Confusion matrix uses the best checkpoint, not the last epoch's weights
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth"), map_location=DEVICE))
    tp, fp, fn, tn = evaluate_confusion(model, val_loader)
    plot_confusion_matrix(tp, fp, fn, tn, output_dir)

    with open(os.path.join(output_dir, "confusion_matrix.json"), "w") as cm_file:
        json.dump({"tp": tp, "fp": fp, "fn": fn, "tn": tn}, cm_file, indent=2)

    print(f"  Saved: history.json/csv, training_curves.png, confusion_matrix.png/json -> {output_dir}")

    return history


def main():
    print("Training configuration loaded")
    print("Device:", DEVICE)
    print("Train sites:", TRAIN_SITES)
    print("Test sites (held out, not used here):", TEST_SITES)

    for dataset_name, in_channels in DATASETS.items():
        train_on_dataset(dataset_name, in_channels)

    print("\nFinished training all dataset variants")


if __name__ == "__main__":
    main()
