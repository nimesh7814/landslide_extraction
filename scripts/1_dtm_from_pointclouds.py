import os
import re
import numpy as np
import laspy
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from scipy.interpolate import griddata

data_dir = r"..\data"

GROUND_CLASS = 2
RESOLUTION = 0.1  # meters

las_files = [f for f in os.listdir(data_dir) if f.endswith(".las")]

for las_name in sorted(las_files):
    match = re.search(r"site_(\d+)", las_name)
    if not match:
        continue
    site_no = match.group(1)

    las_path = os.path.join(data_dir, las_name)
    shp_path = os.path.join(data_dir, f"site_{site_no}_footprint.shp")
    out_path = os.path.join(data_dir, f"site_{site_no}_dtm.tif")

    if not os.path.exists(shp_path):
        print(f"Skipping site_{site_no}: footprint shapefile not found")
        continue

    print(f"\nProcessing site {site_no}")

    footprint = gpd.read_file(shp_path)
    boundary_geom = footprint.geometry.union_all()
    minx, miny, maxx, maxy = footprint.total_bounds
    crs = footprint.crs

    xs, ys, zs = [], [], []
    with laspy.open(las_path) as f:
        for chunk in f.chunk_iterator(2_000_000):
            mask = (
                (chunk.classification == GROUND_CLASS)
                & (chunk.x >= minx) & (chunk.x <= maxx)
                & (chunk.y >= miny) & (chunk.y <= maxy)
            )
            if np.any(mask):
                xs.append(chunk.x[mask])
                ys.append(chunk.y[mask])
                zs.append(chunk.z[mask])

    if not xs:
        print(f"  No ground points found within footprint bounds for site {site_no}")
        continue

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)

    pts_gdf = gpd.GeoDataFrame(
        {"z": z}, geometry=gpd.points_from_xy(x, y), crs=crs
    )
    clipped = pts_gdf[pts_gdf.within(boundary_geom)]

    print(f"  Ground points after clip: {len(clipped)}")
    if len(clipped) == 0:
        print(f"  Skipping site {site_no}: no points inside polygon")
        continue

    x = clipped.geometry.x.values
    y = clipped.geometry.y.values
    z = clipped["z"].values

    ncols = int(np.ceil((maxx - minx) / RESOLUTION))
    nrows = int(np.ceil((maxy - miny) / RESOLUTION))

    col_idx = ((x - minx) / RESOLUTION).astype(int)
    row_idx = ((maxy - y) / RESOLUTION).astype(int)
    col_idx = np.clip(col_idx, 0, ncols - 1)
    row_idx = np.clip(row_idx, 0, nrows - 1)

    sum_grid = np.zeros((nrows, ncols), dtype=np.float64)
    count_grid = np.zeros((nrows, ncols), dtype=np.int32)
    np.add.at(sum_grid, (row_idx, col_idx), z)
    np.add.at(count_grid, (row_idx, col_idx), 1)

    dtm = np.full((nrows, ncols), np.nan, dtype=np.float32)
    filled = count_grid > 0
    dtm[filled] = (sum_grid[filled] / count_grid[filled]).astype(np.float32)

    print(f"  Cells with direct point data: {filled.sum()} / {filled.size} "
          f"({100 * filled.sum() / filled.size:.1f}%)")

    if np.any(~filled):
        known_rows, known_cols = np.where(filled)
        known_vals = dtm[filled]
        missing_rows, missing_cols = np.where(~filled)

        # Linear interpolation first (smooth blending between known points)
        linear_vals = griddata(
            (known_rows, known_cols), known_vals,
            (missing_rows, missing_cols), method="linear"
        )

        # Nearest-neighbor fallback only for cells linear couldn't reach
        # (outside the convex hull of known points)
        still_missing = np.isnan(linear_vals)
        if np.any(still_missing):
            nearest_vals = griddata(
                (known_rows, known_cols), known_vals,
                (missing_rows[still_missing], missing_cols[still_missing]),
                method="nearest"
            )
            linear_vals[still_missing] = nearest_vals

        dtm[missing_rows, missing_cols] = linear_vals

    transform = from_origin(minx, maxy, RESOLUTION, RESOLUTION)
    outside_mask = geometry_mask(
        [boundary_geom], out_shape=(nrows, ncols), transform=transform, invert=False
    )
    dtm[outside_mask] = np.nan

    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress="lzw",
    ) as dst:
        dst.write(dtm, 1)

    print(f"  Saved: {out_path}")