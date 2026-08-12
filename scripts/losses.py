import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    # Combines BCEWithLogits with a soft Dice loss; both the Dice term and pos_weight (>1 penalizes
    # missed landslide pixels more) counteract the severe class imbalance in landslide tiles.

    def __init__(self, bce_weight=0.5, smooth=1.0, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth

        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight
        )

        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs * targets_flat).sum(dim=1)
        union = probs.sum(dim=1) + targets_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss
