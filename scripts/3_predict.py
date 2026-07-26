import os

# See 1_create_train_dataset.py for why: some systems (e.g. with
# PostgreSQL/PostGIS installed) set PROJ_LIB / PROJ_DATA globally to
# an incompatible proj.db, which breaks rasterio's CRS lookups.
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import random
import argparse
import csv
import shutil
import tempfile
from contextlib import ExitStack

import numpy as np
import cv2
import torch
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.merge import merge as rasterio_merge
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
    RANDOM_SEED,
    USE_SKYSENSE_FOR_ORTHO,
    SKYSENSE_CHECKPOINT_PATH
)
from model import UNet
from skysense_model import SkySenseUNet
from dataset import list_tiles
from metrics import raw_confusion_counts, metrics_from_confusion
from plotting import plot_confusion_matrix

_tile_geometry_cache = {}

# Project's known CRS (SLD99 / Sri Lanka Grid 1999), matching
# FALLBACK_CRS_EPSG in 1_create_train_dataset.py. Applied to every
# GeoTIFF this script writes, regardless of what (if anything) the
# source orthomosaic itself reports.
FULL_SITE_CRS_EPSG = 5235


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
        help="Which site set to draw samples from (default: test, i.e. the held-out sites). "
             "Ignored if --site is given."
    )
    parser.add_argument(
        "--site",
        type=int,
        nargs="+",
        default=None,
        help="Predict specific site number(s) (1-12) instead of a whole site set, e.g. "
             "--site 3 or --site 3 7. Overrides --sites."
    )
    parser.add_argument(
        "--full-site",
        dest="full_site",
        action="store_true",
        default=True,
        help="Run inference on every tile of each site (not just --num samples) and "
             "stitch the predicted tile masks into one seamless GeoTIFF covering the "
             "whole site, plus per-site metrics CSV and confusion matrix. On by default."
    )
    parser.add_argument(
        "--no-full-site",
        dest="full_site",
        action="store_false",
        help="Disable full-site mosaic/metrics generation; only produce the --num "
             "sample grid and per-tile GeoTIFFs."
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


def write_georeferenced_tile(array, dataset_name, site_name, tile_no, output_dir, suffix):
    """Writes a single tile -- 2D single-band, or 3D band-first
    multi-band -- as a georeferenced GeoTIFF using that tile's
    real-world bounds recovered from that dataset variant's tile
    shapefile. Always tagged with EPSG:{FULL_SITE_CRS_EPSG} (this
    project's known CRS), regardless of what the source orthomosaic
    itself reports."""

    bounds_by_tile, _ = load_tile_geometry(dataset_name, site_name)

    if tile_no not in bounds_by_tile:
        return None

    left, bottom, right, top = bounds_by_tile[tile_no]

    if array.ndim == 2:
        band_count = 1
        height, width = array.shape
    else:
        band_count, height, width = array.shape

    transform = from_bounds(left, bottom, right, top, width, height)

    tif_path = os.path.join(output_dir, f"{site_name}_{tile_no}_{suffix}.tif")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": band_count,
        "dtype": "uint8",
        "transform": transform,
        "crs": CRS.from_epsg(FULL_SITE_CRS_EPSG),
        "compress": "lzw"
    }

    with rasterio.open(tif_path, "w", **profile) as destination:
        if band_count == 1:
            destination.write(array.astype("uint8"), 1)
        else:
            destination.write(array.astype("uint8"))

    return tif_path


def save_predicted_mask_tif(pred_mask, dataset_name, site_name, tile_no, output_dir):
    """Writes the predicted mask as a georeferenced GeoTIFF (EPSG:5235),
    using the tile's real-world bounds recovered from that dataset
    variant's tile shapefile."""

    tif_path = write_georeferenced_tile(
        (pred_mask * 255).astype("uint8"),
        dataset_name, site_name, tile_no, output_dir, "pred_mask"
    )

    if tif_path is None:
        print(f"    Warning: no tile geometry for {site_name} tile {tile_no}, skipping GeoTIFF export")

    return tif_path


def mosaic_tiles(tif_paths):
    """Merges a list of individually georeferenced tile GeoTIFFs into one
    seamless array + rasterio profile covering their combined extent."""

    with ExitStack() as open_files:
        sources = [open_files.enter_context(rasterio.open(path)) for path in tif_paths]

        mosaic_array, mosaic_transform = rasterio_merge(sources, method="first")

        profile = sources[0].profile.copy()
        profile.update(
            height=mosaic_array.shape[1],
            width=mosaic_array.shape[2],
            transform=mosaic_transform
        )

    return mosaic_array, profile


def save_mosaic_tif(mosaic_array, profile, path):
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(mosaic_array)


def _downsample_for_display(array_2d, max_dim=2000, is_mask=False):
    """Shrinks a full-site array before plotting, so the overview PNG
    stays a reasonable size even when the stitched site is huge. Uses
    nearest-neighbor for masks (keeps them binary) and area averaging
    for the input image."""

    height, width = array_2d.shape[:2]
    scale = min(1.0, max_dim / max(height, width))

    if scale >= 1.0:
        return array_2d

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA

    return cv2.resize(array_2d, (new_width, new_height), interpolation=interpolation)


def plot_full_site_overview(input_mosaic, true_mask_mosaic, pred_mosaic, dataset_name, site_name, output_dir):
    """Saves a single-row (input orthomosaic / true mask / predicted
    landslide mask) overview figure for the whole, stitched site -- the
    full-site counterpart to plot_prediction_grid's per-tile samples."""

    display_image = np.transpose(input_mosaic, (1, 2, 0)).astype(np.float32) / 255.0
    display_image = _downsample_for_display(display_image)
    display_image = np.clip(display_image, 0.0, 1.0)

    display_true_mask = _downsample_for_display(true_mask_mosaic[0], is_mask=True)
    display_pred_mask = _downsample_for_display(pred_mosaic[0], is_mask=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(display_image)
    axes[0].set_title("Input Image (Orthomosaic)")
    axes[0].axis("off")

    axes[1].imshow(display_true_mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("True Mask (Landslide Area)")
    axes[1].axis("off")

    axes[2].imshow(display_pred_mask, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Predicted Mask (Landslide Area)")
    axes[2].axis("off")

    fig.suptitle(f"{dataset_name} | {site_name} (full site)")
    fig.tight_layout()

    save_path = os.path.join(output_dir, f"{site_name}_full_overview.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


def export_full_site_mask(model, dataset_name, site_name, pairs, output_dir):
    """Runs inference on every tile in `pairs` (the full set of tiles for
    this site, not just the handful used for the sample grid). From that:
      - stitches the predicted tiles into one seamless, EPSG:5235
        landslide mask GeoTIFF covering the whole site
      - aggregates pixel-wise TP/FP/FN/TN across every tile, giving
        dataset-wide accuracy/precision/recall/F1(Dice)/IoU plus a
        confusion matrix plot for this site
      - stitches the input image and true mask too, purely to build a
        full-site overview PNG (input orthomosaic / true mask /
        predicted mask)

    Returns a metrics dict for this site, or None if no tiles could be
    georeferenced."""

    tiles_dir = os.path.join(output_dir, f"{site_name}_tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix=f"{site_name}_mosaic_")

    pred_tif_paths = []
    input_tif_paths = []
    true_mask_tif_paths = []

    total_tp = total_fp = total_fn = total_tn = 0.0

    try:
        with torch.no_grad():
            for image_path, mask_path in tqdm(pairs, desc=f"Predicting {site_name} (full site)", unit="tile", leave=False):
                _, tile_no = parse_tile_filename(image_path)

                image = np.load(image_path).astype(np.float32)
                mask = np.load(mask_path).astype(np.float32)

                image_chw = np.transpose(image, (2, 0, 1))
                image_tensor = torch.from_numpy(image_chw).unsqueeze(0).to(DEVICE)
                mask_tensor = torch.from_numpy(mask / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)

                logits = model(image_tensor)
                pred_mask = create_mask(logits)[0, 0].cpu().numpy()

                tp, fp, fn, tn = raw_confusion_counts(logits, mask_tensor)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

                pred_tif_path = write_georeferenced_tile(
                    (pred_mask * 255).astype("uint8"), dataset_name, site_name, tile_no, tiles_dir, "pred_mask"
                )
                if pred_tif_path is not None:
                    pred_tif_paths.append(pred_tif_path)

                rgb_uint8 = (np.clip(image[:, :, :3], 0.0, 1.0) * 255.0)
                rgb_chw = np.transpose(rgb_uint8, (2, 0, 1))
                input_tif_path = write_georeferenced_tile(
                    rgb_chw, dataset_name, site_name, tile_no, temp_dir, "input"
                )
                if input_tif_path is not None:
                    input_tif_paths.append(input_tif_path)

                true_mask_tif_path = write_georeferenced_tile(
                    mask.astype("uint8"), dataset_name, site_name, tile_no, temp_dir, "true_mask"
                )
                if true_mask_tif_path is not None:
                    true_mask_tif_paths.append(true_mask_tif_path)

        if not pred_tif_paths:
            print(f"    Warning: no georeferenced tiles produced for {site_name}, skipping full-site mosaic")
            return None

        pred_mosaic_array, pred_profile = mosaic_tiles(pred_tif_paths)
        mosaic_path = os.path.join(output_dir, f"{site_name}_full_predicted_mask.tif")
        save_mosaic_tif(pred_mosaic_array, pred_profile, mosaic_path)

        input_mosaic_array, _ = mosaic_tiles(input_tif_paths)
        true_mask_mosaic_array, _ = mosaic_tiles(true_mask_tif_paths)

        overview_path = plot_full_site_overview(
            input_mosaic_array, true_mask_mosaic_array, pred_mosaic_array,
            dataset_name, site_name, output_dir
        )

        confusion_matrix_filename = f"{site_name}_confusion_matrix.png"
        plot_confusion_matrix(
            total_tp, total_fp, total_fn, total_tn, output_dir,
            filename=confusion_matrix_filename
        )

        site_metrics = metrics_from_confusion(total_tp, total_fp, total_fn, total_tn)

        print(f"  {site_name}: {len(pairs)} tile(s) stitched -> {os.path.basename(mosaic_path)}")
        print(f"  {site_name}: overview -> {os.path.basename(overview_path)}, "
              f"confusion matrix -> {confusion_matrix_filename}")

        return {
            "dataset": dataset_name,
            "site": site_name,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
            **site_metrics
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


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


def save_metrics_csv(rows, path):
    """Writes the per-site accuracy/precision/recall/F1(Dice)/IoU (and
    raw TP/FP/FN/TN) rows out to a CSV file."""

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value, fmt):
    if value is None:
        return "-"
    if fmt is None:
        return str(value)
    return format(value, fmt)


def print_metrics_table(title, rows):
    """Prints a simple aligned ASCII table of per-site prediction
    metrics to the console."""

    columns = [
        ("site", "Site", None),
        ("accuracy", "Accuracy", ".4f"),
        ("precision", "Precision", ".4f"),
        ("recall", "Recall", ".4f"),
        ("dice", "F1 / Dice", ".4f"),
        ("iou", "IoU", ".4f")
    ]

    print(f"\n{title}")

    widths = []
    for key, header, fmt in columns:
        cell_values = [_format_value(row.get(key), fmt) for row in rows]
        widths.append(max([len(header)] + [len(v) for v in cell_values]))

    def format_row(values):
        return " | ".join(value.ljust(width) for value, width in zip(values, widths))

    print(format_row([header for _, header, _ in columns]))
    print("-+-".join("-" * width for width in widths))

    for row in rows:
        values = [_format_value(row.get(key), fmt) for key, _, fmt in columns]
        print(format_row(values))


def load_model(dataset_name, in_channels):
    if dataset_name == "01_ortho_dataset" and USE_SKYSENSE_FOR_ORTHO:
        # freeze_backbone doesn't matter here (eval mode, no
        # optimizer) -- SkySenseUNet still needs a checkpoint_path to
        # construct its decoder (channel dims are discovered from a
        # dummy forward pass through the backbone), but every weight,
        # frozen or not, gets overwritten by the fine-tuned
        # best_model.pth loaded just below anyway.
        model = SkySenseUNet(
            checkpoint_path=SKYSENSE_CHECKPOINT_PATH,
            out_channels=OUTPUT_CHANNELS,
            freeze_backbone=False
        ).to(DEVICE)
    else:
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


def show_predictions(dataset_name, in_channels, num_samples, sites, full_site=False):
    print(f"\n=== Predicting on dataset: {dataset_name} ({in_channels} input channels) ===")

    model = load_model(dataset_name, in_channels)

    output_dir = os.path.join(PREDICT_OUTPUT_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    rng = random.Random(RANDOM_SEED)

    full_site_metrics = []

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

        if full_site:
            site_metrics = export_full_site_mask(model, dataset_name, site_name, pairs, output_dir)

            if site_metrics is not None:
                full_site_metrics.append(site_metrics)

    if full_site and full_site_metrics:
        csv_path = os.path.join(output_dir, "prediction_metrics.csv")
        save_metrics_csv(full_site_metrics, csv_path)

        print_metrics_table(f"Full-site prediction metrics ({dataset_name})", full_site_metrics)

        print(f"\n  Saved metrics -> {os.path.basename(csv_path)}")

    print(f"  Saved overviews and GeoTIFFs -> {output_dir}")


def main():
    args = parse_args()

    if args.site is not None:
        sites = args.site
        print("Prediction configuration loaded")
        print("Device:", DEVICE)
        print("Site(s) used:", sites)
    else:
        sites = TEST_SITES if args.sites == "test" else TRAIN_SITES
        print("Prediction configuration loaded")
        print("Device:", DEVICE)
        print(f"Site set used ({args.sites}):", sites)

    print("Samples per site:", args.num)
    print("Full-site mosaic:", "on" if args.full_site else "off")

    datasets_to_run = select_datasets(args.model)
    print("Datasets:", list(datasets_to_run.keys()))

    for dataset_name, in_channels in datasets_to_run.items():
        show_predictions(dataset_name, in_channels, args.num, sites, full_site=args.full_site)

    print("\nFinished generating predictions")


if __name__ == "__main__":
    main()
