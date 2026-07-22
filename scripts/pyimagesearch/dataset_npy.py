import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SegmentationDatasetNPY(Dataset):
    def __init__(self, imagePaths, maskPaths, targetSize=None, augment=False):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.targetSize = targetSize
        self.augment = augment

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # images are saved as (C, H, W) float32, masks as (H, W) uint8
        image = np.load(self.imagePaths[idx]).astype("float32")
        mask = np.load(self.maskPaths[idx]).astype("float32")

        image = np.nan_to_num(image, nan=0.0)

        image_t = torch.from_numpy(image)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                image_t = torch.flip(image_t, dims=[2])
                mask_t = torch.flip(mask_t, dims=[2])
            if torch.rand(1).item() < 0.5:
                image_t = torch.flip(image_t, dims=[1])
                mask_t = torch.flip(mask_t, dims=[1])

        if self.targetSize is not None:
            image_t = F.interpolate(
                image_t.unsqueeze(0), size=self.targetSize,
                mode="bilinear", align_corners=False
            ).squeeze(0)
            mask_t = F.interpolate(
                mask_t.unsqueeze(0), size=self.targetSize, mode="nearest"
            ).squeeze(0)

        return image_t, mask_t
