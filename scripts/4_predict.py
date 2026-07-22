import csv
import os

import geopandas as gpd
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds, bounds as window_bounds
from rasterio.windows import transform as window_transform
from rasterio.enums import Resampling
from tqdm import tqdm

from pyimagesearch import config
from pyimagesearch.model import UNet


# Must match 2_training_data.py and 3_train_model.py
SCENARIOS = {

    "RGB": 3,
    "RGB_DTM": 4,
    "RGB_DTM_Hillshade": 5,
    "RGB_DTM_Hillshade_Slope": 6

}

tile_size = 256
nodata_threshold = 0.30

input_folder = os.path.join(config.base_dir, "data", "0_input")


def compute_metrics(tp, fp, fn, tn):

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# Same normalization function used in 2_training_data.py
def normalize(array):

    array = array.astype("float32")

    min_val = np.nanmin(array)
    max_val = np.nanmax(array)

    return (array - min_val) / (max_val - min_val + 0.00001)


def load_model(scenario_name):

    model_path = os.path.join(
        config.TRAINING_RESULTS_PATH,
        scenario_name,
        f"unet_{scenario_name}.pth"
    )

    if not os.path.exists(model_path):
        print(f"[WARNING] no trained model found for {scenario_name} at {model_path}, skipping")
        return None

    model = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    model = model.to(config.DEVICE)
    model.eval()

    return model


def predict_site(site_no, models):

    site = f"site_{site_no:02d}"

    print(f"\n================ Predicting {site} ================")

    ortho_file = os.path.join(input_folder, f"{site}_orthomosaic.tif")
    dtm_file = os.path.join(input_folder, f"{site}_dtm.tif")
    hill_file = os.path.join(input_folder, f"{site}_hillshade.tif")
    slope_file = os.path.join(input_folder, f"{site}_slope.tif")
    shp_file = os.path.join(input_folder, f"{site}_landslide_annotation.shp")

    required_files = [ortho_file, dtm_file, hill_file, slope_file, shp_file]
    missing = [f for f in required_files if not os.path.exists(f)]

    if missing:
        print("Missing files:")
        for f in missing:
            print(f)
        print("Skipping", site)
        return []

    # Ground-truth polygons, used to score predictions against the
    # actual landslide extent (same "value" convention as 2_training_data.py)
    polygons = gpd.read_file(shp_file)

    if "value" not in polygons.columns:
        raise ValueError(
            f"'value' field not found in {shp_file}. "
            f"Available columns: {list(polygons.columns)}"
        )

    with rasterio.open(ortho_file) as ortho, \
         rasterio.open(dtm_file) as dtm, \
         rasterio.open(hill_file) as hill, \
         rasterio.open(slope_file) as slope:

        # Same malformed-CRS fallback as 2_training_data.py: some
        # orthomosaics carry a bare LOCAL_CS with no EPSG code, so only
        # reproject when the raster CRS is actually transformable.
        if ortho.crs is not None and ortho.crs.to_epsg() is not None and polygons.crs != ortho.crs:
            polygons = polygons.to_crs(ortho.crs)

        width = ortho.width
        height = ortho.height
        transform = ortho.transform
        crs = ortho.crs

        shapes = [
            (geom, val)
            for geom, val in zip(polygons.geometry, polygons["value"])
        ]

        # one output array per scenario, filled with nodata (255)
        # until a tile actually gets predicted
        prob_outputs = {
            name: np.full((height, width), np.nan, dtype="float32")
            for name in models
        }

        # confusion-matrix counts per scenario, accumulated over every
        # valid (non-nodata) pixel across every predicted tile
        stats = {
            name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            for name in models
        }

        for y in tqdm(range(0, height, tile_size), desc=site):
            for x in range(0, width, tile_size):

                if x + tile_size > width or y + tile_size > height:
                    continue

                win = Window(x, y, tile_size, tile_size)

                rgb = ortho.read((1, 2, 3), window=win)

                tile_bounds = window_bounds(win, transform)

                dtm_data = dtm.read(
                    1,
                    window=from_bounds(*tile_bounds, transform=dtm.transform),
                    out_shape=(tile_size, tile_size),
                    resampling=Resampling.bilinear,
                    boundless=True,
                    masked=True
                ).astype("float32").filled(np.nan)

                hill_data = hill.read(
                    1,
                    window=from_bounds(*tile_bounds, transform=hill.transform),
                    out_shape=(tile_size, tile_size),
                    resampling=Resampling.bilinear,
                    boundless=True,
                    masked=True
                ).astype("float32").filled(np.nan)

                slope_data = slope.read(
                    1,
                    window=from_bounds(*tile_bounds, transform=slope.transform),
                    out_shape=(tile_size, tile_size),
                    resampling=Resampling.bilinear,
                    boundless=True,
                    masked=True
                ).astype("float32").filled(np.nan)

                empty = (
                    np.all(rgb == 0, axis=0)
                    | np.isnan(dtm_data)
                    | np.isnan(hill_data)
                    | np.isnan(slope_data)
                )

                empty_ratio = np.sum(empty) / (tile_size * tile_size)

                if empty_ratio > nodata_threshold:
                    continue

                # Ground-truth mask for this tile, rasterized the same
                # way as 2_training_data.py (native tile_size resolution)
                transform_window = window_transform(win, transform)

                gt_mask = rasterize(
                    shapes,
                    out_shape=(tile_size, tile_size),
                    transform=transform_window,
                    fill=0,
                    dtype="uint8"
                )

                valid = ~empty

                rgb_norm = rgb.astype("float32") / 255.0
                dtm_norm = normalize(dtm_data)
                hill_norm = normalize(hill_data)
                slope_norm = normalize(slope_data)

                full_image = np.concatenate(
                    [
                        rgb_norm,
                        dtm_norm[np.newaxis, :, :],
                        hill_norm[np.newaxis, :, :],
                        slope_norm[np.newaxis, :, :]
                    ],
                    axis=0
                )

                images = {

                    "RGB": full_image[0:3],
                    "RGB_DTM": full_image[0:4],
                    "RGB_DTM_Hillshade": full_image[0:5],
                    "RGB_DTM_Hillshade_Slope": full_image[0:6]

                }

                for name, model in models.items():

                    tile = images[name]

                    tile_tensor = torch.from_numpy(tile.astype("float32")).unsqueeze(0)
                    tile_tensor = tile_tensor.to(config.DEVICE)

                    # Models were trained on tiles downsampled to
                    # (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH), so the
                    # input has to be resized the same way here before
                    # the output is resized back up to tile_size to
                    # align with the raster grid.
                    model_size = (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)
                    tile_tensor = F.interpolate(
                        tile_tensor, size=model_size,
                        mode="bilinear", align_corners=False
                    )

                    with torch.no_grad():
                        logits = model(tile_tensor)
                        probs = torch.sigmoid(logits)

                    probs = F.interpolate(
                        probs, size=(tile_size, tile_size),
                        mode="bilinear", align_corners=False
                    ).squeeze().cpu().numpy()

                    prob_outputs[name][y:y + tile_size, x:x + tile_size] = probs

                    pred = probs[valid] >= 0.5
                    truth = gt_mask[valid].astype(bool)

                    stats[name]["tp"] += int(np.sum(pred & truth))
                    stats[name]["fp"] += int(np.sum(pred & ~truth))
                    stats[name]["fn"] += int(np.sum(~pred & truth))
                    stats[name]["tn"] += int(np.sum(~pred & ~truth))

        # write one georeferenced prediction raster per scenario
        for name, prob_array in prob_outputs.items():

            out_dir = os.path.join(config.TEST_PREDICTIONS_PATH, name)
            os.makedirs(out_dir, exist_ok=True)

            prob_path = os.path.join(out_dir, f"{site}_prob.tif")
            mask_path = os.path.join(out_dir, f"{site}_mask.tif")

            out_meta = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": "float32",
                "crs": crs,
                "transform": transform,
                "nodata": np.nan
            }

            with rasterio.open(prob_path, "w", **out_meta) as dst:
                dst.write(prob_array, 1)

            binary_mask = np.where(np.isnan(prob_array), 255, (prob_array >= 0.5).astype("uint8"))

            bin_meta = out_meta.copy()
            bin_meta["dtype"] = "uint8"
            bin_meta["nodata"] = 255

            with rasterio.open(mask_path, "w", **bin_meta) as dst:
                dst.write(binary_mask, 1)

            print(f"[INFO] saved {name} prediction for {site} -> {prob_path}, {mask_path}")

        # per-site, per-scenario Accuracy/Precision/Recall/F1 against
        # the site's own ground-truth landslide annotations
        rows = []
        for name, counts in stats.items():

            metrics = compute_metrics(**counts)

            print(
                f"[INFO] {site} / {name} -> "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, "
                f"f1={metrics['f1']:.4f}"
            )

            rows.append({
                "site": site,
                "scenario": name,
                **counts,
                **metrics
            })

        return rows


if __name__ == "__main__":

    print("[INFO] loading trained models...")
    models = {}
    for scenario_name in SCENARIOS:
        model = load_model(scenario_name)
        if model is not None:
            models[scenario_name] = model

    if not models:
        raise RuntimeError("No trained models found under config.TRAINING_RESULTS_PATH. "
                            "Run 3_train_model.py first.")

    for site_no in config.TEST_SITES:
        predict_site(site_no, models)

    print("\n--------------------------------")
    print("TEST SITE PREDICTION COMPLETE")
    print("--------------------------------\n")
