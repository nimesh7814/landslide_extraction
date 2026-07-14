"""Convert shapefiles to same-named GeoPackages and archive their source files."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import geopandas as gpd
import pyogrio


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_DIR / "data"
DEFAULT_ARCHIVE_DIR = DEFAULT_INPUT_DIR / "old"
TARGET_CRS = "EPSG:5235"

# A shapefile is a bundle of files. These are the component/metadata suffixes
# that should be archived after its replacement GeoPackage has been verified.
SHAPEFILE_SUFFIXES = {
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
    ".qix",
    ".sbn",
    ".sbx",
    ".qmd",
    ".fix",
    ".ain",
    ".aih",
    ".atx",
    ".ixs",
    ".mxs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert each shapefile below an input directory to a same-named "
            f"GeoPackage in {TARGET_CRS}, then move its source bundle to an archive."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory to search recursively (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help=f"Destination for shapefile bundles (default: {DEFAULT_ARCHIVE_DIR})",
    )
    return parser.parse_args()


def find_shapefiles(input_dir: Path, archive_dir: Path) -> list[Path]:
    """Find inputs recursively without reprocessing files already archived."""
    return sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file()
            and path.suffix.casefold() == ".shp"
            and not path.is_relative_to(archive_dir)
        ),
        key=lambda path: str(path.relative_to(input_dir)).casefold(),
    )


def shapefile_components(shapefile: Path) -> list[Path]:
    """Return all known files belonging to a shapefile dataset."""
    components = []
    prefix = f"{shapefile.stem}."
    for path in shapefile.parent.iterdir():
        lower_name = path.name.casefold()
        if not path.is_file() or not lower_name.startswith(prefix.casefold()):
            continue
        if path.suffix.casefold() in SHAPEFILE_SUFFIXES or lower_name.endswith(".shp.xml"):
            components.append(path)
    return sorted(components, key=lambda path: path.name.casefold())


def convert_one(shapefile: Path) -> Path:
    """Convert and verify one dataset, replacing an existing output atomically."""
    output = shapefile.with_suffix(".gpkg")
    temporary_output = output.with_name(f".{output.stem}.tmp.gpkg")
    temporary_output.unlink(missing_ok=True)

    frame = gpd.read_file(shapefile, engine="pyogrio")
    if frame.crs is None:
        raise ValueError(f"Cannot convert a dataset with no CRS: {shapefile}")
    if frame.crs.to_epsg() != 5235:
        print(f"  Reprojecting {frame.crs} -> {TARGET_CRS}")
        frame = frame.to_crs(TARGET_CRS)

    try:
        frame.to_file(
            temporary_output,
            layer=shapefile.stem,
            driver="GPKG",
            engine="pyogrio",
            index=False,
        )

        info = pyogrio.read_info(temporary_output, layer=shapefile.stem)
        if info["features"] != len(frame):
            raise RuntimeError(
                f"Feature verification failed for {shapefile.name}: expected "
                f"{len(frame)}, found {info['features']}"
            )
        if info["crs"] != TARGET_CRS:
            raise RuntimeError(
                f"CRS verification failed for {shapefile.name}: "
                f"expected {TARGET_CRS}, found {info['crs']}"
            )

        os.replace(temporary_output, output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    return output


def convert(input_dir: Path, archive_dir: Path) -> None:
    input_dir = input_dir.resolve()
    archive_dir = archive_dir.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if archive_dir == input_dir or not archive_dir.is_relative_to(input_dir):
        raise ValueError("The archive directory must be located inside the input directory")

    shapefiles = find_shapefiles(input_dir, archive_dir)
    if not shapefiles:
        existing_outputs = sorted(input_dir.rglob("*.gpkg"))
        archived_inputs = (
            sorted(archive_dir.rglob("*.shp")) if archive_dir.is_dir() else []
        )
        print(f"No unarchived shapefiles found below: {input_dir}")
        if existing_outputs and archived_inputs:
            print(
                f"Nothing to do: found {len(existing_outputs)} GeoPackages and "
                f"{len(archived_inputs)} archived shapefiles."
            )
        else:
            print("Nothing to convert.")
        return

    # Convert and verify every dataset before moving any original source files.
    outputs = []
    for index, shapefile in enumerate(shapefiles, start=1):
        relative_path = shapefile.relative_to(input_dir)
        print(f"[{index}/{len(shapefiles)}] {relative_path}")
        output = convert_one(shapefile)
        outputs.append(output)
        print(f"  Created {output.name} ({TARGET_CRS})")

    moves: list[tuple[Path, Path]] = []
    for shapefile in shapefiles:
        relative_parent = shapefile.parent.relative_to(input_dir)
        for component in shapefile_components(shapefile):
            destination = archive_dir / relative_parent / component.name
            if destination.exists():
                raise FileExistsError(
                    f"Archive destination already exists; sources were not moved: {destination}"
                )
            moves.append((component, destination))

    for source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, destination)

    print(f"\nCreated and verified {len(outputs)} GeoPackages in {TARGET_CRS}.")
    print(f"Archived {len(moves)} shapefile component files in: {archive_dir}")


def main() -> None:
    args = parse_args()
    convert(args.input_dir, args.archive_dir)


if __name__ == "__main__":
    main()
