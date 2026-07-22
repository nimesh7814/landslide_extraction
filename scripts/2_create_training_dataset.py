import os
import rasterio
import geopandas as gpd
import numpy as np

from rasterio.windows import Window, from_bounds, bounds as window_bounds
from rasterio.features import rasterize
from rasterio.windows import transform as window_transform
from rasterio.enums import Resampling

from tqdm import tqdm

# Script location
script_dir = os.path.dirname(
    os.path.abspath(__file__)
)

# Go one directory up
base_dir = os.path.abspath(
    os.path.join(script_dir, "..")
)

input_folder = os.path.join(
    base_dir,
    "data",
    "0_input"
)

output_folder = os.path.join(
    base_dir,
    "data",
    "1_training_dataset"
)



datasets = {

    "RGB": 3,
    "RGB_DTM": 4,
    "RGB_DTM_Hillshade": 5,
    "RGB_DTM_Hillshade_Slope": 6

}


for dataset in datasets:

    os.makedirs(
        os.path.join(
            output_folder,
            dataset,
            "images"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            output_folder,
            dataset,
            "masks"
        ),
        exist_ok=True
    )



# Tile size and nodata threshold
tile_size = 256
nodata_threshold = 0.30


# Training sites (site_01 to site_10)
sites = range(1,11)


# Normalization function for Raster
def normalize(array):

    array = array.astype(
        "float32"
    )

    min_val = np.nanmin(array)
    max_val = np.nanmax(array)

    return (
        array - min_val
    ) / (
        max_val - min_val + 0.00001
    )


# Process each site
for site_no in sites:

    site = f"site_{site_no:02d}"

    print("Processing:", site)

    # Input files
    ortho_file = os.path.join(
        input_folder,
        f"{site}_orthomosaic.tif"
    )

    dtm_file = os.path.join(
        input_folder,
        f"{site}_dtm.tif"
    )

    hill_file = os.path.join(
        input_folder,
        f"{site}_hillshade.tif"
    )

    slope_file = os.path.join(
        input_folder,
        f"{site}_slope.tif"
    )

    shp_file = os.path.join(
        input_folder,
        f"{site}_landslide_annotation.shp"
    )


    required_files = [

        ortho_file,
        dtm_file,
        hill_file,
        slope_file,
        shp_file

    ]


    missing = [

        f for f in required_files
        if not os.path.exists(f)

    ]


    if missing:

        print("Missing files:")
        for f in missing:
            print(f)

        print("Skipping", site)

        continue


    # Read shapefile
    polygons = gpd.read_file(
        shp_file
    )

    # Make sure the "value" field exists (1 = landslide, 0 = no landslide)
    if "value" not in polygons.columns:

        raise ValueError(
            f"'value' field not found in {shp_file}. "
            f"Available columns: {list(polygons.columns)}"
        )

    # Open Raster files
    with rasterio.open(ortho_file) as ortho, \
         rasterio.open(dtm_file) as dtm, \
         rasterio.open(hill_file) as hill, \
         rasterio.open(slope_file) as slope:

        # Some orthomosaic exports embed a bare LOCAL_CS (no datum/projection
        # params), so PROJ can't build a transformer to it even though the
        # coordinates are already in the same system as the annotations.
        # Only reproject when the raster CRS is a real, transformable CRS.
        if ortho.crs is not None and ortho.crs.to_epsg() is not None and polygons.crs != ortho.crs:

            polygons = polygons.to_crs(
                ortho.crs
            )

        width = ortho.width

        height = ortho.height

        transform = ortho.transform

        tile_id = 0

        total_tiles = (

            (width // tile_size)

            *

            (height // tile_size)

        )


        print(
            "Expected tiles:",
            total_tiles
        )



        # Tile loop
        for y in tqdm(
            range(
                0,
                height,
                tile_size
            ),
            desc=site
        ):


            for x in range(
                0,
                width,
                tile_size
            ):



                # Skip tiles that exceed image boundaries
                if (
                    x + tile_size > width

                    or

                    y + tile_size > height
                ):

                    continue



                win = Window(

                    x,

                    y,

                    tile_size,

                    tile_size

                )


                # Read Data (RGB only - band 4 in these orthomosaics is an
                # alpha channel, not a usable spectral band)
                rgb = ortho.read(
                    (1, 2, 3),
                    window=win
                )

                # dtm/hillshade/slope have a coarser resolution - and in some
                # sites a smaller real-world footprint - than the
                # orthomosaic, so the tile must be read by its real-world
                # bounds (not the ortho pixel window), resampled to
                # tile_size, and read boundless so off-coverage areas come
                # back masked instead of raising an out-of-range error.
                tile_bounds = window_bounds(
                    win,
                    transform
                )

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


                # Check for NoData - ortho nodata is 0 (uint8), and a
                # pixel outside dtm/hillshade/slope coverage comes back
                # as NaN from the boundless reads above
                empty = (
                    np.all(rgb == 0, axis=0)
                    | np.isnan(dtm_data)
                    | np.isnan(hill_data)
                    | np.isnan(slope_data)
                )


                empty_ratio = (

                    np.sum(empty)

                    /

                    (tile_size * tile_size)

                )



                if empty_ratio > nodata_threshold:

                    continue


                # Create mask from shapefile
                transform_window = window_transform(

                    win,

                    transform

                )


                # Use the actual "value" attribute per polygon
                # (1 = landslide, 0 = no landslide) instead of
                # hardcoding every polygon to 1
                shapes = [

                    (geom, val)

                    for geom, val in zip(
                        polygons.geometry,
                        polygons["value"]
                    )

                ]



                mask = rasterize(

                    shapes,

                    out_shape=(

                        tile_size,

                        tile_size

                    ),

                    transform=transform_window,

                    fill=0,

                    dtype="uint8"

                )


                # Normalize rasters
                rgb = rgb.astype(
                    "float32"
                ) / 255.0

                dtm_data = normalize(
                    dtm_data
                )
                
                hill_data = normalize(
                    hill_data
                )

                slope_data = normalize(
                    slope_data
                )


                # Create 6-channel image
                full_image = np.concatenate(

                    [

                        rgb,

                        dtm_data[np.newaxis,:,:],

                        hill_data[np.newaxis,:,:],

                        slope_data[np.newaxis,:,:]

                    ],

                    axis=0

                )


                # Save images and masks for each dataset
                images = {


                    "RGB":

                    full_image[0:3],


                    "RGB_DTM":

                    full_image[0:4],


                    "RGB_DTM_Hillshade":

                    full_image[0:5],


                    "RGB_DTM_Hillshade_Slope":

                    full_image[0:6]


                }



                for name, image in images.items():


                    image_path = os.path.join(

                        output_folder,

                        name,

                        "images",

                        f"{site}_{tile_id}.npy"

                    )


                    mask_path = os.path.join(

                        output_folder,

                        name,

                        "masks",

                        f"{site}_{tile_id}_mask.npy"

                    )



                    np.save(

                        image_path,

                        image

                    )


                    np.save(

                        mask_path,

                        mask

                    )



                tile_id += 1



        print(

            site,

            "saved tiles:",

            tile_id

        )



print("\n--------------------------------")
print("TRAINING DATASET CREATION COMPLETE")
print("--------------------------------\n")