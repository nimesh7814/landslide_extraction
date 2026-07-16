import os
import re
import numpy as np
import laspy
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
from scipy.spatial import cKDTree

IDW_NEIGHBORS = 12  # number of nearest known cells averaged per gap cell
IDW_POWER = 2  # inverse-distance weighting exponent

data_dir = r"..\data"

GROUND_CLASS = 2
# Ground-point density varies a lot between sites (observed ~0.2-0.8 m natural spacing
# across this dataset), so the grid resolution is derived per site from the actual point
# density rather than fixed. A fixed fine resolution (e.g. 0.1 m) on a sparser site leaves
# ~95%+ of cells with no direct measurement, which scipy.griddata then bridges with large
# flat Delaunay-triangle planes -- visible as sharp straight-line "edge" artifacts in the
# DTM. Matching resolution to point spacing keeps most cells directly measured.
MIN_RESOLUTION = 0.1  # meters, never grid finer than this
MAX_RESOLUTION = 1.0  # meters, never grid coarser than this

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

    point_density = len(clipped) / boundary_geom.area  # points per m^2
    natural_spacing = 1.0 / np.sqrt(point_density) if point_density > 0 else MAX_RESOLUTION
    RESOLUTION = float(np.clip(round(natural_spacing, 2), MIN_RESOLUTION, MAX_RESOLUTION))
    print(f"  Ground point density: {point_density:.2f} pts/m^2 (natural spacing "
          f"~{natural_spacing:.2f} m) -> grid resolution {RESOLUTION} m")

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

        # Inverse-distance-weighted fill over the k nearest directly-measured cells.
        # scipy.griddata's "linear" method triangulates the known points (Delaunay) and
        # interpolates each triangle as a flat plane -- this produces visible straight-edge
        # facets wherever a triangle spans a locally sparser patch, even after matching the
        # grid resolution to the site's average point density. IDW blends several nearby
        # points smoothly instead of stitching flat planes, so it has no facet edges, and
        # naturally covers cells beyond the convex hull of known points too (no separate
        # nearest-neighbor fallback needed).
        known_coords = np.column_stack([known_rows, known_cols])
        missing_coords = np.column_stack([missing_rows, missing_cols])
        tree = cKDTree(known_coords)
        k = min(IDW_NEIGHBORS, len(known_vals))
        dist, idx = tree.query(missing_coords, k=k)
        if k == 1:
            dist = dist[:, np.newaxis]
            idx = idx[:, np.newaxis]
        weights = 1.0 / np.maximum(dist, 1e-6) ** IDW_POWER
        weights /= weights.sum(axis=1, keepdims=True)
        dtm[missing_rows, missing_cols] = np.sum(known_vals[idx] * weights, axis=1)

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