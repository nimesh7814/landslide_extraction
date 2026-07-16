import os
import re
import warnings
import numpy as np
import rasterio
from rasterio.crs import CRS

data_dir = r"..\data"

TARGET_EPSG = 5235


def read_dtm(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        transform = src.transform
        src_crs = src.crs
        nodata = src.nodata
    if nodata is not None and not np.isnan(nodata):
        arr[arr == nodata] = np.nan
    cellsize_x = transform.a
    cellsize_y = -transform.e  # transform.e is negative (north-up raster)

    target_crs = CRS.from_epsg(TARGET_EPSG)
    if src_crs is None:
        warnings.warn(f"{path}: source file has no CRS defined; tagging output as EPSG:{TARGET_EPSG}")
    elif src_crs.to_epsg() != TARGET_EPSG:
        warnings.warn(
            f"{path}: source CRS reports as {src_crs.to_epsg() or src_crs.to_wkt()[:60]}, "
            f"not EPSG:{TARGET_EPSG}. Forcing EPSG:{TARGET_EPSG} on output -- verify this is "
            f"actually correct and the source DTM isn't in a different CRS."
        )
    return arr, transform, target_crs, cellsize_x, cellsize_y


def neighbor_stack(arr):

    padded = np.pad(arr, 1, mode="edge")

    z1 = padded[0:-2, 0:-2]
    z2 = padded[0:-2, 1:-1]
    z3 = padded[0:-2, 2:]
    z4 = padded[1:-1, 0:-2]
    z5 = padded[1:-1, 1:-1]  # center
    z6 = padded[1:-1, 2:]
    z7 = padded[2:, 0:-2]
    z8 = padded[2:, 1:-1]
    z9 = padded[2:, 2:]

    window = np.stack([z1, z2, z3, z4, z5, z6, z7, z8, z9], axis=0)
    invalid = np.any(np.isnan(window), axis=0)

    return (z1, z2, z3, z4, z5, z6, z7, z8, z9), invalid


def compute_slope_aspect(arr, cellsize_x, cellsize_y):
    (z1, z2, z3, z4, _z5, z6, z7, z8, z9), invalid = neighbor_stack(arr)

    # Horn (1981) weighted finite-difference kernel
    dz_dx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * cellsize_x)
    dz_dy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8 * cellsize_y)

    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    aspect_deg = 90.0 - np.degrees(aspect_rad)
    aspect_deg = np.mod(aspect_deg, 360.0)
    # flat cells (no dx/dy) have undefined aspect -> conventional -1 sentinel
    flat = (dz_dx == 0) & (dz_dy == 0)
    aspect_deg[flat] = -1.0

    slope_deg[invalid] = np.nan
    aspect_deg[invalid] = np.nan
    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)


def compute_curvature(arr, cellsize_x, cellsize_y):
    (z1, z2, z3, z4, z5, z6, z7, z8, z9), invalid = neighbor_stack(arr)
    cs = (cellsize_x + cellsize_y) / 2.0  # Zevenbergen & Thorne assumes square cells

    # Zevenbergen & Thorne (1987) second-order partial derivatives
    D = ((z4 + z6) / 2.0 - z5) / (cs ** 2)
    E = ((z2 + z8) / 2.0 - z5) / (cs ** 2)
    Fx = (z6 - z4) / (2.0 * cs)
    Fy = (z2 - z8) / (2.0 * cs)

    p = Fx ** 2 + Fy ** 2
    q = p + 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        profile_curv = -2.0 * (D * Fx ** 2 + E * Fy ** 2 + 0 * (Fx * Fy)) / np.where(p == 0, np.nan, p) / np.power(q, 1.5)
        plan_curv = -2.0 * (D * Fy ** 2 + E * Fx ** 2 - 0 * (Fx * Fy)) / np.where(p == 0, np.nan, p) / np.sqrt(q)
        general_curv = -2.0 * (D + E)

    # flat/level cells: profile & plan curvature are conventionally 0 there
    flat = p == 0
    profile_curv[flat] = 0.0
    plan_curv[flat] = 0.0

    for grid in (profile_curv, plan_curv, general_curv):
        grid[invalid] = np.nan

    return (
        profile_curv.astype(np.float32),
        plan_curv.astype(np.float32),
        general_curv.astype(np.float32),
    )


def compute_tri(arr):
    (z1, z2, z3, z4, z5, z6, z7, z8, z9), invalid = neighbor_stack(arr)
    neighbors = np.stack([z1, z2, z3, z4, z6, z7, z8, z9], axis=0)  # excludes center
    tri = np.sqrt(np.sum((neighbors - z5) ** 2, axis=0))
    tri[invalid] = np.nan
    return tri.astype(np.float32)


def write_single_band(path, arr, transform, crs):
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress="lzw",
    ) as dst:
        dst.write(arr, 1)


def write_multi_band(path, arrays, band_names, transform, crs):
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=arrays[0].shape[0],
        width=arrays[0].shape[1],
        count=len(arrays),
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress="lzw",
    ) as dst:
        for i, (arr, name) in enumerate(zip(arrays, band_names), start=1):
            dst.write(arr, i)
            dst.set_band_description(i, name)


if __name__ == "__main__":
    dtm_files = [f for f in os.listdir(data_dir) if re.match(r"site_\d+_dtm\.tif$", f)]

    for dtm_name in sorted(dtm_files):
        match = re.search(r"site_(\d+)", dtm_name)
        site_no = match.group(1)
        dtm_path = os.path.join(data_dir, dtm_name)

        print(f"\nProcessing site {site_no}")
        arr, transform, crs, cellsize_x, cellsize_y = read_dtm(dtm_path)
        print(f"  Grid: {arr.shape}, cell size: {cellsize_x:.2f} x {cellsize_y:.2f} m")

        slope, aspect = compute_slope_aspect(arr, cellsize_x, cellsize_y)
        profile_curv, plan_curv, general_curv = compute_curvature(arr, cellsize_x, cellsize_y)
        tri = compute_tri(arr)

        write_single_band(os.path.join(data_dir, f"site_{site_no}_slope.tif"), slope, transform, crs)
        write_single_band(os.path.join(data_dir, f"site_{site_no}_aspect.tif"), aspect, transform, crs)
        write_multi_band(
            os.path.join(data_dir, f"site_{site_no}_curvature.tif"),
            [profile_curv, plan_curv, general_curv],
            ["profile_curvature", "plan_curvature", "general_curvature"],
            transform, crs,
        )
        write_single_band(os.path.join(data_dir, f"site_{site_no}_tri.tif"), tri, transform, crs)

        print(f"  Slope: {np.nanmin(slope):.1f}-{np.nanmax(slope):.1f} deg, "
              f"TRI: {np.nanmin(tri):.3f}-{np.nanmax(tri):.3f}")
        print(f"  Saved: slope, aspect, curvature (3-band), tri")
