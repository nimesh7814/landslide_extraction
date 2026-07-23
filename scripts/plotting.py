import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(history, output_dir):
    """Saves a 2x3 figure with train/val loss, IoU, Dice, accuracy,
    precision and recall curves for one dataset variant."""

    epochs = [row["epoch"] for row in history]

    def series(key):
        return [row[key] for row in history]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    axes[0, 0].plot(epochs, series("train_loss"), label="train")
    axes[0, 0].plot(epochs, series("val_loss"), label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, series("train_iou"), label="train")
    axes[0, 1].plot(epochs, series("val_iou"), label="val")
    axes[0, 1].set_title("IoU")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()

    axes[0, 2].plot(epochs, series("train_dice"), label="train")
    axes[0, 2].plot(epochs, series("val_dice"), label="val")
    axes[0, 2].set_title("Dice / F1")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].legend()

    axes[1, 0].plot(epochs, series("train_accuracy"), label="train")
    axes[1, 0].plot(epochs, series("val_accuracy"), label="val")
    axes[1, 0].set_title("Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, series("train_precision"), label="train")
    axes[1, 1].plot(epochs, series("val_precision"), label="val")
    axes[1, 1].set_title("Precision")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()

    axes[1, 2].plot(epochs, series("train_recall"), label="train")
    axes[1, 2].plot(epochs, series("val_recall"), label="val")
    axes[1, 2].set_title("Recall")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def plot_confusion_matrix(tp, fp, fn, tn, output_dir):
    """Saves a 2x2 confusion matrix heatmap (pixel counts) for the
    landslide / non-landslide classes."""

    matrix = [[tn, fp], [fn, tp]]
    labels = ["Non-landslide", "Landslide"]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Validation confusion matrix (pixels)")

    max_val = max(matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1])

    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i][j] > max_val / 2 else "black"
            ax.text(j, i, f"{matrix[i][j]:,.0f}", ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
