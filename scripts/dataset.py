import os
import glob
import random

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from config import (
    DATASET_DIR,
    RANDOM_SAMPLE_PERCENTAGE,
    RANDOM_SEED,
    TRAIN_PERCENTAGE,
    VALIDATION_PERCENTAGE,
    AUGMENT_FLIP_PROBABILITY,
    AUGMENT_BRIGHTNESS_PROBABILITY,
    AUGMENT_BRIGHTNESS_RANGE,
    AUGMENT_BLUR_PROBABILITY,
    AUGMENT_BLUR_SIGMA_RANGE
)

assert abs(TRAIN_PERCENTAGE + VALIDATION_PERCENTAGE - 1.0) < 1e-6, (
    "TRAIN_PERCENTAGE and VALIDATION_PERCENTAGE must sum to 1.0"
)


def _site_prefixes(sites):
    return {f"site_{site:02d}" for site in sites}


def list_tiles(dataset_name, sites):
    """Return a sorted list of (image_path, mask_path) tuples for a given
    dataset variant, restricted to the given site numbers."""

    img_dir = os.path.join(DATASET_DIR, dataset_name, "images")
    mask_dir = os.path.join(DATASET_DIR, dataset_name, "masks")

    prefixes = _site_prefixes(sites)

    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.npy")))

    pairs = []
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        site_prefix = "_".join(filename.split("_")[:2])

        if site_prefix not in prefixes:
            continue

        mask_name = filename.replace(".npy", "_m.npy")
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Missing mask for tile: {mask_path}")

        pairs.append((image_path, mask_path))

    return pairs


def train_val_split(pairs):
    """Shuffle deterministically, optionally subsample, then split into
    train/val pairs using the percentages defined in config.py."""

    rng = random.Random(RANDOM_SEED)

    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    if RANDOM_SAMPLE_PERCENTAGE < 1.0:
        keep = max(1, int(len(shuffled) * RANDOM_SAMPLE_PERCENTAGE))
        shuffled = shuffled[:keep]

    split_index = int(len(shuffled) * TRAIN_PERCENTAGE)

    train_pairs = shuffled[:split_index]
    val_pairs = shuffled[split_index:]

    return train_pairs, val_pairs


def estimate_positive_ratio(pairs, max_samples=200, seed=RANDOM_SEED):
    """Estimates the fraction of positive (landslide) pixels by scanning
    a sample of masks, used to weight the loss against class imbalance."""

    rng = random.Random(seed)

    sample = pairs if len(pairs) <= max_samples else rng.sample(pairs, max_samples)

    positive_pixels = 0
    total_pixels = 0

    for _, mask_path in sample:
        mask = np.load(mask_path)
        positive_pixels += int((mask > 0).sum())
        total_pixels += mask.size

    if total_pixels == 0:
        return 0.0

    return positive_pixels / total_pixels


def _random_brightness(image):
    """Scales pixel intensities by a random factor to simulate darker or
    brighter conditions. Image is expected to be normalized to [0, 1]."""

    factor = random.uniform(*AUGMENT_BRIGHTNESS_RANGE)

    image = image * factor

    return np.clip(image, 0.0, 1.0)


def _random_blur(image):
    """Applies a light Gaussian blur to each channel independently to
    simulate slightly out-of-focus or lower-resolution imagery."""

    sigma = random.uniform(*AUGMENT_BLUR_SIGMA_RANGE)

    blurred = np.empty_like(image)

    for channel_index in range(image.shape[2]):
        blurred[:, :, channel_index] = cv2.GaussianBlur(
            image[:, :, channel_index],
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma
        )

    return blurred


class LandslideDataset(Dataset):
    """Loads pre-tiled (image, mask) .npy pairs produced by
    1_create_train_dataset.py."""

    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]

        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        mask = mask / 255.0

        if self.augment:
            if random.random() < AUGMENT_FLIP_PROBABILITY:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()

            if random.random() < AUGMENT_FLIP_PROBABILITY:
                image = np.flip(image, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()

            if random.random() < AUGMENT_BRIGHTNESS_PROBABILITY:
                image = _random_brightness(image)

            if random.random() < AUGMENT_BLUR_PROBABILITY:
                image = _random_blur(image)

        image = np.transpose(image, (2, 0, 1))

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor
