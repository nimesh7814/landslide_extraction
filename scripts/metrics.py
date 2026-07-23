import torch


@torch.no_grad()
def binary_metrics(logits, targets, threshold=0.5, eps=1e-7):
    """Returns IoU, Dice/F1, precision and recall for a batch of
    binary segmentation predictions."""

    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return {
        "iou": iou.item(),
        "dice": dice.item(),
        "precision": precision.item(),
        "recall": recall.item()
    }


@torch.no_grad()
def raw_confusion_counts(logits, targets, threshold=0.5):
    """Returns raw pixel counts (tp, fp, fn, tn) for a batch, used to
    build an aggregated confusion matrix over a full dataset split."""

    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()

    return tp, fp, fn, tn
