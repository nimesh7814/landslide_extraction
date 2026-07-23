import os
import argparse
from contextlib import ExitStack

import cv2
import numpy as np
import rasterio
import shapefile
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import (
    Window,
    bounds as window_bounds,
    from_bounds
)
from tqdm import tqdm


# PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "..",
    "output",
    "1_training_datasets"
)


# SETTINGS
TARGET_SIZE = 512

SITE_START = 1
SITE_END = 12


DATASETS = {

    "01_ortho_dataset":
    [
        "orthomosaic"
    ],

    "02_ortho_dtm_dataset":
    [
        "orthomosaic",
        "dtm"
    ],

    "03_ortho_dtm_hillshade_dataset":
    [
        "orthomosaic",
        "dtm",
        "hillshade"
    ],

    "04_ortho_dtm_hillshade_slope_dataset":
    [
        "orthomosaic",
        "dtm",
        "hillshade",
        "slope"
    ]

}


# FUNCTIONS
def get_files(site):

    return {

        "orthomosaic":
        os.path.join(
            DATA_DIR,
            f"site_{site:02d}_orthomosaic.tif"
        ),

        "dtm":
        os.path.join(
            DATA_DIR,
            f"site_{site:02d}_dtm.tif"
        ),

        "hillshade":
        os.path.join(
            DATA_DIR,
            f"site_{site:02d}_hillshade.tif"
        ),

        "slope":
        os.path.join(
            DATA_DIR,
            f"site_{site:02d}_slope.tif"
        ),

        "mask":
        os.path.join(
            DATA_DIR,
            f"site_{site:02d}_mask.tif"
        ),

        "annotation":
        os.path.join(
            DATA_DIR,
            f"site_{site:02d}_landslide_annotation.shp"
        ),

        "footprint":
        os.path.join(
            DATA_DIR,
            f"site_{site:02d}_footprint.shp"
        )

    }


def read_geometries(path):

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Geometry file is missing: {path}"
        )

    with shapefile.Reader(path) as source:
        geometries = [
            shape.__geo_interface__
            for shape in source.iterShapes()
            if shape.points
        ]

    if not geometries:
        raise ValueError(
            f"No geometries found in {path}"
        )

    return geometries


def create_mask_if_missing(files):

    mask_path = files["mask"]

    if os.path.isfile(mask_path):
        return

    annotation_path = files["annotation"]
    orthomosaic_path = files["orthomosaic"]

    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(
            "Cannot create mask because the annotation "
            f"file is missing: {annotation_path}"
        )

    if not os.path.isfile(orthomosaic_path):
        raise FileNotFoundError(
            "Cannot create mask because the orthomosaic "
            f"file is missing: {orthomosaic_path}"
        )

    print(
        "Mask not found; rasterizing:",
        annotation_path
    )

    temporary_mask_path = f"{mask_path}.tmp"

    geometries = read_geometries(annotation_path)

    with rasterio.open(orthomosaic_path) as reference:
        mask = rasterize(
            (
                (geometry, 255)
                for geometry in geometries
            ),
            out_shape=(
                reference.height,
                reference.width
            ),
            transform=reference.transform,
            fill=0,
            dtype=np.uint8
        )

        profile = reference.profile.copy()
        profile.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=0,
            compress="lzw",
            photometric="minisblack",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER"
        )

        try:
            with rasterio.open(
                temporary_mask_path,
                "w",
                **profile
            ) as destination:
                destination.write(mask, 1)

            os.replace(
                temporary_mask_path,
                mask_path
            )

        finally:
            if os.path.exists(temporary_mask_path):
                os.remove(temporary_mask_path)

    print("Created mask:", mask_path)


def find_best_tile_positions(files):

    footprint_path = files["footprint"]
    orthomosaic_path = files["orthomosaic"]

    geometries = read_geometries(footprint_path)

    print(
        "Finding best 512 x 512 grid inside:",
        footprint_path
    )

    with rasterio.open(orthomosaic_path) as reference:
        footprint = rasterize(
            (
                (geometry, 1)
                for geometry in geometries
            ),
            out_shape=(
                reference.height,
                reference.width
            ),
            transform=reference.transform,
            fill=0,
            dtype=np.uint8
        )

    valid_height = footprint.shape[0] - TARGET_SIZE + 1
    valid_width = footprint.shape[1] - TARGET_SIZE + 1

    if valid_height <= 0 or valid_width <= 0:
        raise ValueError(
            "The footprint is smaller than one "
            f"{TARGET_SIZE} x {TARGET_SIZE} tile"
        )

    # With the anchor at the top-left, erosion marks every
    # possible tile origin whose full square is in the footprint.

    kernel = np.ones(
        (TARGET_SIZE, TARGET_SIZE),
        dtype=np.uint8
    )

    valid_origins = cv2.erode(
        footprint,
        kernel,
        anchor=(0, 0),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    valid_origins = valid_origins[
        :valid_height,
        :valid_width
    ]

    del footprint
    del kernel

    best_count = 0
    best_x_offset = 0
    best_y_offset = 0

    full_x_blocks = valid_width // TARGET_SIZE
    full_x_width = full_x_blocks * TARGET_SIZE
    remainder = valid_width - full_x_width

    for y_offset in tqdm(
            range(min(TARGET_SIZE, valid_height)),
            desc="Optimizing tile grid",
            unit="offset",
            leave=False
    ):

        sampled_rows = valid_origins[
            y_offset::TARGET_SIZE,
            :
        ]

        counts = np.zeros(
            TARGET_SIZE,
            dtype=np.int64
        )

        if full_x_blocks:
            counts += np.count_nonzero(
                sampled_rows[
                    :,
                    :full_x_width
                ].reshape(
                    sampled_rows.shape[0],
                    full_x_blocks,
                    TARGET_SIZE
                ),
                axis=(0, 1)
            )

        if remainder:
            counts[:remainder] += np.count_nonzero(
                sampled_rows[
                    :,
                    full_x_width:
                ],
                axis=0
            )

        x_offset = int(np.argmax(counts))
        count = int(counts[x_offset])

        if count > best_count:
            best_count = count
            best_x_offset = x_offset
            best_y_offset = y_offset

    if best_count == 0:
        raise ValueError(
            "No complete "
            f"{TARGET_SIZE} x {TARGET_SIZE} "
            "tile fits inside the footprint"
        )

    positions = [
        (x, y)
        for y in range(
            best_y_offset,
            valid_height,
            TARGET_SIZE
        )
        for x in range(
            best_x_offset,
            valid_width,
            TARGET_SIZE
        )
        if valid_origins[y, x]
    ]

    del valid_origins

    print(
        "Selected",
        len(positions),
        "fully contained tiles with grid offset",
        f"(x={best_x_offset}, y={best_y_offset})"
    )

    return positions


def write_tile_index_shapefile(files, dataset_name, site_name, positions):
    """Writes one shapefile per site with a polygon for every tile in
    the grid, storing the tile number and center coordinates. Written
    into the dataset variant's own 'tiles' folder so it survives
    alongside that variant even if 'images'/'masks' are later deleted
    to free disk space."""

    orthomosaic_path = files["orthomosaic"]

    tiles_dir = os.path.join(OUTPUT_DIR, dataset_name, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    shp_path = os.path.join(tiles_dir, f"{site_name}_tiles.shp")

    with rasterio.open(orthomosaic_path) as reference:
        transform = reference.transform
        crs = reference.crs

        with shapefile.Writer(shp_path, shapeType=shapefile.POLYGON) as writer:
            writer.field("tile_no", "N")
            writer.field("site", "C")
            writer.field("center_x", "F", decimal=3)
            writer.field("center_y", "F", decimal=3)

            for tile_no, (x, y) in enumerate(positions):
                left, bottom, right, top = window_bounds(
                    Window(x, y, TARGET_SIZE, TARGET_SIZE),
                    transform
                )

                polygon = [
                    (left, top),
                    (right, top),
                    (right, bottom),
                    (left, bottom),
                    (left, top)
                ]

                writer.poly([polygon])
                writer.record(
                    tile_no,
                    site_name,
                    (left + right) / 2.0,
                    (top + bottom) / 2.0
                )

    if crs is not None:
        prj_path = os.path.join(tiles_dir, f"{site_name}_tiles.prj")
        with open(prj_path, "w") as prj_file:
            prj_file.write(crs.to_wkt())

    print("Saved tile index shapefile:", shp_path)



def create_output_folder(dataset):

    img_folder = os.path.join(
        OUTPUT_DIR,
        dataset,
        "images"
    )

    mask_folder = os.path.join(
        OUTPUT_DIR,
        dataset,
        "masks"
    )


    os.makedirs(
        img_folder,
        exist_ok=True
    )

    os.makedirs(
        mask_folder,
        exist_ok=True
    )

    return img_folder, mask_folder


def get_normalization_range(
        source,
        band_index,
        cache,
        layer_name
):

    key = (
        os.path.abspath(source.name),
        band_index
    )

    if key in cache:
        return cache[key]

    dtype = np.dtype(
        source.dtypes[band_index - 1]
    )

    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        min_val = float(limits.min)
        max_val = float(limits.max)

    else:
        tags = source.tags(band_index)

        try:
            min_val = float(
                tags["STATISTICS_MINIMUM"]
            )
            max_val = float(
                tags["STATISTICS_MAXIMUM"]
            )

        except (
            KeyError,
            TypeError,
            ValueError
        ):
            min_val = np.inf
            max_val = -np.inf

            blocks = list(
                source.block_windows(band_index)
            )

            for _, window in tqdm(
                    blocks,
                    desc=f"Scanning {layer_name} range",
                    unit="block",
                    leave=False
            ):
                block = source.read(
                    band_index,
                    window=window,
                    masked=True
                )

                values = block.compressed()
                values = values[
                    np.isfinite(values)
                ]

                if values.size:
                    min_val = min(
                        min_val,
                        float(values.min())
                    )
                    max_val = max(
                        max_val,
                        float(values.max())
                    )

            if (
                not np.isfinite(min_val)
                or not np.isfinite(max_val)
            ):
                raise ValueError(
                    "No valid pixels found in "
                    f"{source.name}, band {band_index}"
                )

    cache[key] = (
        min_val,
        max_val
    )

    return cache[key]


def normalize_tile(
        tile,
        min_val,
        max_val
):

    invalid = np.ma.getmaskarray(tile)

    normalized = np.asarray(
        tile.filled(min_val),
        dtype=np.float32
    )

    value_range = max_val - min_val

    if value_range > 0:
        normalized -= min_val
        normalized /= value_range

    else:
        normalized.fill(0)

    np.clip(
        normalized,
        0,
        1,
        out=normalized
    )

    normalized[
        invalid
        | ~np.isfinite(normalized)
    ] = 0

    return normalized


def read_aligned_tile(
        source,
        band_indexes,
        reference,
        reference_window
):

    same_grid = (
        source.width == reference.width
        and source.height == reference.height
        and source.transform.almost_equals(
            reference.transform
        )
    )

    if same_grid:
        source_window = reference_window

    else:
        left, bottom, right, top = window_bounds(
            reference_window,
            reference.transform
        )

        source_window = from_bounds(
            left,
            bottom,
            right,
            top,
            transform=source.transform
        )

    return source.read(
        band_indexes,
        window=source_window,
        out_shape=(
            len(band_indexes),
            TARGET_SIZE,
            TARGET_SIZE
        ),
        resampling=Resampling.bilinear,
        boundless=True,
        masked=True
    )



def create_tiles(
        files,
        bands,
        dataset,
        site_name,
        positions,
        normalization_cache
):

    img_folder, mask_folder = create_output_folder(dataset)

    with ExitStack() as open_files:

        sources = {
            band:
            open_files.enter_context(
                rasterio.open(files[band])
            )
            for band in bands
        }

        reference = sources["orthomosaic"]

        mask_source = open_files.enter_context(
            rasterio.open(files["mask"])
        )

        if (
            mask_source.width != reference.width
            or mask_source.height != reference.height
            or not mask_source.transform.almost_equals(
                reference.transform
            )
        ):
            raise ValueError(
                "Mask grid does not match orthomosaic: "
                f"{files['mask']}"
            )

        layer_details = []

        for band in bands:

            source = sources[band]

            if band == "orthomosaic":
                if source.count < 3:
                    raise ValueError(
                        "Orthomosaic must contain at least "
                        f"three bands: {source.name}"
                    )

                band_indexes = [1, 2, 3]

            else:
                band_indexes = [1]

            ranges = [
                get_normalization_range(
                    source,
                    band_index,
                    normalization_cache,
                    band
                )
                for band_index in band_indexes
            ]

            layer_details.append(
                (
                    source,
                    band_indexes,
                    ranges
                )
            )


        for tile_no, (x, y) in enumerate(
                tqdm(
                    positions,
                    desc=f"Writing {dataset}",
                    unit="tile",
                    leave=False
                )
        ):

            reference_window = Window(
                x,
                y,
                TARGET_SIZE,
                TARGET_SIZE
            )

            channels = []

            for (
                    source,
                    band_indexes,
                    ranges
            ) in layer_details:

                data = read_aligned_tile(
                    source,
                    band_indexes,
                    reference,
                    reference_window
                )

                for channel, value_range in zip(
                        data,
                        ranges
                ):
                    channels.append(
                        normalize_tile(
                            channel,
                            *value_range
                        )
                    )

            img_tile = np.stack(
                channels,
                axis=2
            )

            mask_tile = mask_source.read(
                1,
                window=reference_window
            )

            img_name = (
                f"{site_name}_{tile_no}.npy"
            )

            mask_name = (
                f"{site_name}_{tile_no}_m.npy"
            )

            np.save(
                os.path.join(
                    img_folder,
                    img_name
                ),
                img_tile
            )

            np.save(
                os.path.join(
                    mask_folder,
                    mask_name
                ),
                mask_tile
            )

    return len(positions)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Create tiled training datasets for one or all dataset variants."
    )
    parser.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Which dataset variant to create (1-4). If omitted, all variants are created."
    )
    return parser.parse_args()


def select_datasets(model_number):
    if model_number is None:
        return DATASETS

    dataset_names = list(DATASETS.keys())
    selected_name = dataset_names[model_number - 1]

    return {selected_name: DATASETS[selected_name]}


def main():
    args = parse_args()

    datasets_to_create = select_datasets(args.model)

    print("Dataset creation configuration loaded")
    print("Tile size:", f"{TARGET_SIZE} x {TARGET_SIZE}")
    print("Sites to process:", list(range(SITE_START, SITE_END + 1)))
    print("Datasets to create:", list(datasets_to_create.keys()))

    total = 0
    normalization_cache = {}

    for site in tqdm(
            range(
                SITE_START,
                SITE_END + 1
            ),
            desc="Sites",
            unit="site"
    ):

        site_name = f"site_{site:02d}"

        print("\nProcessing", site_name)

        files = get_files(site)

        # Create the mask from the annotation shapefile when needed

        create_mask_if_missing(files)

        positions = find_best_tile_positions(files)

        for dataset, bands in tqdm(
                datasets_to_create.items(),
                desc=f"{site_name} datasets",
                unit="dataset",
                leave=False
        ):

            print(
                "Creating:",
                dataset
            )

            tiles = create_tiles(
                files,
                bands,
                dataset,
                site_name,
                positions,
                normalization_cache
            )

            write_tile_index_shapefile(files, dataset, site_name, positions)

            print(
                "Tiles:",
                tiles
            )

            total += tiles

    print("\n====================")
    print("Finished")
    print(
        "Total tiles:",
        total
    )
    print("====================")


if __name__ == "__main__":
    main()
