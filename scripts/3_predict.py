import os
import random
import argparse

import numpy as np
import torch
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import shapefile
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATASETS,
    TEST_SITES,
    TRAIN_SITES,
    DATASET_DIR,
    MODEL_OUTPUT_DIR,
    PREDICT_OUTPUT_DIR,
    ENCODER_CHANNELS,
    DECODER_CHANNELS,
    BOTTLENECK_CHANNELS,
    OUTPUT_CHANNELS,
    DEVICE,
    RANDOM_SEED
)
from model import UNet
from dataset import list_tiles

_tile_geometry_cache = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize U-Net predictions and export predicted masks as GeoTIFFs."
    )
    parser.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Which dataset variant to predict with (1-4). If omitted, all variants are used."
    )
    parser.add_argument(
        "--num",
        type=int,
        default=6,
        help="Number of sample tiles to visualize and export (default: 6)."
    )
    parser.add_argument(
        "--sites",
        choices=["test", "train"],
        default="test",
        help="Which site set to draw samples from (default: test, i.e. the held-out sites)."
    )
    return parser.parse_args()


def select_datasets(model_number):
    if model_number is None:
        return DATASETS

    dataset_names = list(DATASETS.keys())
    selected_name = dataset_names[model_number - 1]

    return {selected_name: DATASETS[selected_name]}


def parse_tile_filename(image_path):
    """Recovers the site name and tile number from a tile filename such
    as 'site_01_23.npy' -> ('site_01', 23)."""

    filename = os.path.basename(image_path)
    stem = filename[:-len(".npy")]
    parts = stem.split("_")

    site_name = "_".join(parts[:2])
    tile_no = int(parts[2])

    return site_name, tile_no


def load_tile_geometry(dataset_name, site_name):
    """Loads the per-site tile shapefile (written by
    1_create_train_dataset.py into that dataset variant's own 'tiles'
    folder) and returns a dict mapping tile number to its
    (left, bottom, right, top) bounds, plus the CRS."""

    cache_key = (dataset_name, site_name)

    if cache_key in _tile_geometry_cache:
        return _tile_geometry_cache[cache_key]

    tiles_dir = os.path.join(DATASET_DIR, dataset_name, "tiles")
    shp_path = os.path.join(tiles_dir, f"{site_name}_tiles.shp")
    prj_path = os.path.join(tiles_dir, f"{site_name}_tiles.prj")

    if not os.path.isfile(shp_path):
        raise FileNotFoundError(
            f"Tile shapefile not found for {dataset_name}/{site_name}: {shp_path}. "
            "Run 1_create_train_dataset.py first to generate it."
        )

    bounds_by_tile = {}

    with shapefile.Reader(shp_path) as reader:
        for shape_record in reader.shapeRecords():
            tile_no = shape_record.record["tile_no"]
            xmin, ymin, xmax, ymax = shape_record.shape.bbox
            bounds_by_tile[tile_no] = (xmin, ymin, xmax, ymax)

    crs = None
    if os.path.isfile(prj_path):
        with open(prj_path) as prj_file:
            crs = CRS.from_wkt(prj_file.read())

    result = (bounds_by_tile, crs)
    _tile_geometry_cache[cache_key] = result

    return result


def save_predicted_mask_tif(pred_mask, dataset_name, site_name, tile_no, output_dir):
    """Writes the predicted mask as a georeferenced GeoTIFF, using the
    tile's real-world bounds recovered from that dataset variant's
    tile shapefile."""

    bounds_by_tile, crs = load_tile_geometry(dataset_name, site_name)

    if tile_no not in bounds_by_tile:
        print(f"    Warning: no tile geometry for {site_name} tile {tile_no}, skipping GeoTIFF export")
        return None

    left, bottom, right, top = bounds_by_tile[tile_no]
    height, width = pred_mask.shape

    transform = from_bounds(left, bottom, right, top, width, height)

    tif_path = os.path.join(output_dir, f"{site_name}_{tile_no}_pred_mask.tif")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "transform": transform,
        "compress": "lzw"
    }

    if crs is not None:
        profile["crs"] = crs

    with rasterio.open(tif_path, "w", **profile) as destination:
        destination.write((pred_mask * 255).astype("uint8"), 1)

    return tif_path


def create_mask(logits, threshold=0.5):
    """Converts raw model logits into a binary single-channel mask.
    This plays the same role as argmax-based mask creation for
    multi-class models, but for binary segmentation a sigmoid +
    threshold is the correct equivalent."""

    probs = torch.sigmoid(logits)
    mask = (probs > threshold).float()

    return mask


def plot_prediction_grid(rows, dataset_name, site_name, output_dir):
    """Plots a grid of (input image, true mask, predicted mask) rows,
    one row per sampled tile, and saves it as a single overview figure
    for the site. Column headers are only shown on the top row."""

    num_rows = len(rows)

    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))

    if num_rows == 1:
        axes = axes[np.newaxis, :]

    for row_index, (image_chw, true_mask, pred_mask, tile_no) in enumerate(rows):
        display_image = np.transpose(image_chw[:3], (1, 2, 0))
        display_image = np.clip(display_image, 0.0, 1.0)

        axes[row_index, 0].imshow(display_image)
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(true_mask, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].axis("off")

        axes[row_index, 2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].axis("off")

        axes[row_index, 0].set_ylabel(f"tile {tile_no}", rotation=90, fontsize=8)

    axes[0, 0].set_title("Input Image")
    axes[0, 1].set_title("True Mask")
    axes[0, 2].set_title("Predicted Mask")

    fig.suptitle(f"{dataset_name} | {site_name}")
    fig.tight_layout()

    save_path = os.path.join(output_dir, f"{site_name}_overview.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


def load_model(dataset_name, in_channels):
    model = UNet(
        in_channels=in_channels,
        out_channels=OUTPUT_CHANNELS,
        encoder_channels=ENCODER_CHANNELS,
        decoder_channels=DECODER_CHANNELS,
        bottleneck_channels=BOTTLENECK_CHANNELS
    ).to(DEVICE)

    checkpoint_path = os.path.join(MODEL_OUTPUT_DIR, dataset_name, "best_model.pth")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"No trained checkpoint found for {dataset_name}: {checkpoint_path}. "
            "Run 2_train_model.py first."
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    return model


def show_predictions(dataset_name, in_channels, num_samples, sites):
    print(f"\n=== Predicting on dataset: {dataset_name} ({in_channels} input channels) ===")

    model = load_model(dataset_name, in_channels)

    output_dir = os.path.join(PREDICT_OUTPUT_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    rng = random.Random(RANDOM_SEED)

    for site in sites:
        site_name = f"site_{site:02d}"

        pairs = list_tiles(dataset_name, [site])

        if not pairs:
            print(f"  No tiles found for {site_name}, skipping.")
            continue

        sample_pairs = pairs if len(pairs) <= num_samples else rng.sample(pairs, num_samples)

        print(f"  {site_name}: found {len(pairs)} tile(s), using {len(sample_pairs)} for prediction")

        rows = []

        with torch.no_grad():
            for image_path, mask_path in tqdm(sample_pairs, desc=f"Predicting {site_name}", unit="tile", leave=False):
                _, tile_no = parse_tile_filename(image_path)

                image = np.load(image_path).astype(np.float32)
                mask = np.load(mask_path).astype(np.float32) / 255.0

                image_chw = np.transpose(image, (2, 0, 1))
                image_tensor = torch.from_numpy(image_chw).unsqueeze(0).to(DEVICE)

                logits = model(image_tensor)
                pred_mask = create_mask(logits)[0, 0].cpu().numpy()

                rows.append((image_chw, mask, pred_mask, tile_no))

                save_predicted_mask_tif(pred_mask, dataset_name, site_name, tile_no, output_dir)

        grid_path = plot_prediction_grid(rows, dataset_name, site_name, output_dir)

        print(f"  {site_name}: {len(rows)} sample(s) -> {os.path.basename(grid_path)} + GeoTIFFs")

    print(f"  Saved overviews and GeoTIFFs -> {output_dir}")


def main():
    args = parse_args()

    sites = TEST_SITES if args.sites == "test" else TRAIN_SITES

    print("Prediction configuration loaded")
    print("Device:", DEVICE)
    print(f"Site set used ({args.sites}):", sites)
    print("Samples per site:", args.num)

    datasets_to_run = select_datasets(args.model)
    print("Datasets:", list(datasets_to_run.keys()))

    for dataset_name, in_channels in datasets_to_run.items():
        show_predictions(dataset_name, in_channels, args.num, sites)

    print("\nFinished generating predictions")


if __name__ == "__main__":
    main()
