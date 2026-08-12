import os
import glob
import random
import time

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from config import (
    DATASET_DIR,
    RANDOM_SAMPLE_PERCENTAGE,
    RANDOM_SEED,
    AUGMENT_FLIP_PROBABILITY,
    AUGMENT_BRIGHTNESS_PROBABILITY,
    AUGMENT_BRIGHTNESS_RANGE,
    AUGMENT_BLUR_PROBABILITY,
    AUGMENT_BLUR_SIGMA_RANGE
)


def _site_prefixes(sites):
    return {f"site_{site:02d}" for site in sites}


def list_tiles(dataset_name, sites):
    # Returns a sorted list of (image_path, mask_path) tuples, restricted to the given site numbers.
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


def subsample_pairs(pairs, percentage=RANDOM_SAMPLE_PERCENTAGE, seed=RANDOM_SEED):
    # Deterministically shrinks (image, mask) pairs to `percentage`, for quick experiments;
    # leave at 1.0 to use every pair. Train/validation splitting is separate (config.py FIT/VAL_SITES).
    if percentage >= 1.0:
        return pairs

    rng = random.Random(seed)

    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    keep = max(1, int(len(shuffled) * percentage))

    return shuffled[:keep]


def estimate_positive_ratio(pairs, max_samples=200, seed=RANDOM_SEED):
    # Estimates the fraction of positive (landslide) pixels from a mask sample, to weight the loss.
    rng = random.Random(seed)

    sample = pairs if len(pairs) <= max_samples else rng.sample(pairs, max_samples)

    positive_pixels = 0
    total_pixels = 0

    for _, mask_path in sample:
        mask = _load_npy_with_retry(mask_path)
        positive_pixels += int((mask > 0).sum())
        total_pixels += mask.size

    if total_pixels == 0:
        return 0.0

    return positive_pixels / total_pixels


def _random_brightness(image):
    # Scales pixel intensities (expected normalized to [0, 1]) by a random factor.
    factor = random.uniform(*AUGMENT_BRIGHTNESS_RANGE)

    image = image * factor

    return np.clip(image, 0.0, 1.0)


def _random_blur(image):
    # Applies a light Gaussian blur to each channel independently.
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


def _load_npy_with_retry(path, max_attempts=4, base_delay=0.5):
    # Loads a .npy file, retrying with backoff on transient OS read failures (e.g. a cloud-sync
    # client like OneDrive evicting a placeholder file); re-raises if every attempt fails.
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return np.load(path)
        except (OSError, PermissionError) as error:
            last_error = error

            if attempt == max_attempts:
                break

            delay = base_delay * (2 ** (attempt - 1))
            print(f"  Warning: transient read failure on {path} (attempt {attempt}/{max_attempts}): "
                  f"{error} -- retrying in {delay:.1f}s")
            time.sleep(delay)

    raise last_error


class LandslideDataset(Dataset):
    # Loads pre-tiled (image, mask) .npy pairs produced by 1_create_train_dataset.py.

    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]

        image = _load_npy_with_retry(image_path).astype(np.float32)
        mask = _load_npy_with_retry(mask_path).astype(np.float32)

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
