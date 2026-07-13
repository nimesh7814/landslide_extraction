"""Create one outer boundary shapefile for each orthomosaic GeoTIFF."""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from rasterio import features
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.errors import NodataShadowWarning
from scipy.ndimage import binary_fill_holes
from shapely import coverage_union_all
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union


DEFAULT_RAW_DIR = Path("data")
DEFAULT_TARGET_EPSG = 5235
SHAPEFILE_EXTENSIONS = (
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
    ".qix",
    ".sbn",
    ".sbx",
    ".fix",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one EPSG:5235 outer boundary polygon per orthomosaic."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Folder containing orthomosaic files. Default: {DEFAULT_RAW_DIR}",
    )
    parser.add_argument(
        "--target-epsg",
        type=int,
        default=DEFAULT_TARGET_EPSG,
        help=f"Output EPSG code. Default: {DEFAULT_TARGET_EPSG}",
    )
    parser.add_argument(
        "--mode",
        choices=("footprint", "extent", "auto"),
        default="footprint",
        help=(
            "Boundary mode. 'footprint' uses the valid-pixel outer boundary. "
            "'extent' writes the raster rectangle. 'auto' uses footprint only "
            "when a NoData/alpha/mask exists. Default: footprint."
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=4096,
        help="Pixel tile size for footprint polygonization. Default: 4096",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not overwrite an existing footprint shapefile.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing files.",
    )
    return parser.parse_args()


def find_orthomosaics(raw_dir: Path) -> list[Path]:
    tif_paths = list(raw_dir.rglob("*.tif")) + list(raw_dir.rglob("*.tiff"))
    return sorted(path for path in tif_paths if "orthomosaic" in path.stem.lower())


def output_path_for_raster(raster_path: Path) -> Path:
    site_match = re.match(r"(?i)^(site_\d+)_orthomosaic$", raster_path.stem)
    if site_match:
        return raster_path.with_name(f"{site_match.group(1)}_footprint.shp")

    site_match = re.match(
        r"(?i)^(site_\d+)_orthomosaic_(.+)$",
        raster_path.stem,
    )
    if site_match:
        site_id, location = site_match.groups()
        return raster_path.with_name(f"{site_id}_footprint_{location}.shp")

    # Retain support for older names such as Alawathugoda_Orthomosaic.tif.
    match = re.match(r"(?i)^(.*?)(?:[_\-\s]*orthomosaic.*)$", raster_path.stem)
    prefix = match.group(1).rstrip("_- ") if match else raster_path.stem
    if not prefix:
        prefix = raster_path.stem
    return raster_path.with_name(f"{prefix}_footprint.shp")


def raster_extent_polygon(dataset: rasterio.io.DatasetReader) -> Polygon:
    transform = dataset.transform
    width = dataset.width
    height = dataset.height
    return Polygon(
        [
            transform * (0, 0),
            transform * (width, 0),
            transform * (width, height),
            transform * (0, height),
            transform * (0, 0),
        ]
    )


def has_alpha_band(dataset: rasterio.io.DatasetReader) -> bool:
    return (
        dataset.count >= 4
        and len(dataset.colorinterp) >= 4
        and dataset.colorinterp[3] == ColorInterp.alpha
    )


def has_valid_data_mask(dataset: rasterio.io.DatasetReader) -> bool:
    if has_alpha_band(dataset):
        return True

    return any(
        MaskFlags.all_valid not in band_flags
        for band_flags in dataset.mask_flag_enums
    )


def iter_windows(width: int, height: int, block_size: int):
    for row_offset in range(0, height, block_size):
        window_height = min(block_size, height - row_offset)
        for col_offset in range(0, width, block_size):
            window_width = min(block_size, width - col_offset)
            yield rasterio.windows.Window(
                col_offset,
                row_offset,
                window_width,
                window_height,
            )


def valid_pixel_mask(dataset: rasterio.io.DatasetReader, window) -> np.ndarray:
    """Return a filled valid-data mask for one raster window.

    If band 4 is an alpha band, it is used directly. This is the Rasterio
    equivalent of `dataset.GetRasterBand(4).ReadAsArray() > 0`, but it reads
    only one tile at a time instead of loading the full orthomosaic into memory.
    """

    if has_alpha_band(dataset):
        mask = dataset.read(4, window=window) > 0
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NodataShadowWarning)
            mask = dataset.dataset_mask(window=window) > 0

    return binary_fill_holes(mask)


def geometry_polygon_parts(geometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]

    if hasattr(geometry, "geoms"):
        polygons = []
        for part in geometry.geoms:
            polygons.extend(geometry_polygon_parts(part))
        return polygons

    return []


def single_outer_shell(polygons: list[Polygon]) -> Polygon:
    try:
        footprint = coverage_union_all(polygons)
    except Exception:
        footprint = unary_union(polygons)

    footprint = footprint.buffer(0)
    polygon_parts = geometry_polygon_parts(footprint)

    if not polygon_parts:
        raise ValueError("Could not create a polygon footprint.")

    outer_polygon = max(polygon_parts, key=lambda polygon: polygon.area)
    return Polygon(outer_polygon.exterior)


def raster_footprint_polygon(
    dataset: rasterio.io.DatasetReader,
    block_size: int,
) -> Polygon:
    if block_size <= 0:
        raise ValueError("--block-size must be greater than 0")

    polygons = []

    for window in iter_windows(dataset.width, dataset.height, block_size):
        mask_filled = valid_pixel_mask(dataset, window)

        if not mask_filled.any():
            continue

        window_transform = dataset.window_transform(window)

        if mask_filled.all():
            polygons.append(
                Polygon(
                    [
                        window_transform * (0, 0),
                        window_transform * (window.width, 0),
                        window_transform * (window.width, window.height),
                        window_transform * (0, window.height),
                        window_transform * (0, 0),
                    ]
                )
            )
            continue

        polygons.extend(
            shape(geometry)
            for geometry, value in features.shapes(
                mask_filled.astype(np.uint8),
                mask=mask_filled,
                transform=window_transform,
            )
            if value == 1
        )

    if not polygons:
        raise ValueError(f"No valid pixels found in {dataset.name}")

    return single_outer_shell(polygons)


def boundary_geometry(
    dataset: rasterio.io.DatasetReader,
    mode: str,
    block_size: int,
):
    if mode == "extent":
        return raster_extent_polygon(dataset)

    if mode == "footprint":
        return raster_footprint_polygon(dataset, block_size)

    if has_valid_data_mask(dataset):
        return raster_footprint_polygon(dataset, block_size)

    return raster_extent_polygon(dataset)


def remove_existing_shapefile(output_path: Path) -> None:
    for extension in SHAPEFILE_EXTENSIONS:
        sidecar = output_path.with_suffix(extension)
        if sidecar.exists():
            try:
                sidecar.unlink()
            except PermissionError as error:
                raise PermissionError(
                    "Cannot overwrite existing shapefile because it is open in "
                    f"another program: {sidecar}. Close the layer in ArcGIS/QGIS "
                    "and run the script again."
                ) from error


def looks_like_sri_lanka_grid_1999(crs: CRS) -> bool:
    crs_text = f"{crs.name} {crs.to_wkt()}".lower()
    return "sld99" in crs_text and "sri lanka grid 1999" in crs_text


def normalized_source_crs(source_crs, target_crs: CRS) -> CRS:
    if source_crs is None:
        print("WARNING: source raster has no CRS; assigning output CRS directly.")
        return target_crs

    crs = CRS.from_user_input(source_crs)

    if crs.to_epsg() == target_crs.to_epsg():
        return target_crs

    if looks_like_sri_lanka_grid_1999(crs):
        return target_crs

    return crs


def write_boundary(
    geometry,
    source_crs,
    output_path: Path,
    target_crs: CRS,
) -> None:
    crs = normalized_source_crs(source_crs, target_crs)
    boundary = gpd.GeoDataFrame({"value": [0]}, geometry=[geometry], crs=crs)

    if boundary.crs != target_crs:
        boundary = boundary.to_crs(target_crs)

    remove_existing_shapefile(output_path)
    boundary.to_file(output_path, driver="ESRI Shapefile", engine="pyogrio")


def create_boundary(
    raster_path: Path,
    output_path: Path,
    target_crs: CRS,
    mode: str,
    block_size: int,
) -> None:
    with rasterio.open(raster_path) as dataset:
        geometry = boundary_geometry(dataset, mode, block_size)
        write_boundary(geometry, dataset.crs, output_path, target_crs)


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    target_crs = CRS.from_epsg(args.target_epsg)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw folder does not exist: {raw_dir}")

    orthomosaics = find_orthomosaics(raw_dir)
    if not orthomosaics:
        raise FileNotFoundError(f"No orthomosaic .tif files found under {raw_dir}")

    for raster_path in orthomosaics:
        output_path = output_path_for_raster(raster_path)

        if args.skip_existing and output_path.exists():
            print(f"SKIP existing: {output_path}")
            continue

        if args.dry_run:
            print(f"WOULD CREATE: {output_path} <- {raster_path}")
            continue

        create_boundary(
            raster_path,
            output_path,
            target_crs,
            args.mode,
            args.block_size,
        )
        print(f"CREATED: {output_path}")


if __name__ == "__main__":
    main()
