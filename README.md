# Landslide Extraction

## Introduction

Landslides have caused significant loss of life and damage in Sri Lanka. However, mapping landslides remains a largely manual task, making it time-consuming and labor-intensive—especially when numerous landslides occur simultaneously in close proximity. To address this problem, this study integrates geospatial techniques to accurately identify landslide areas from drone imagery.

The resulting model can be used to build landslide inventory datasets that document landslide events, as well as to create training datasets for prediction models that rely on post-disaster data. This makes it a valuable tool for landslide studies and research.

## Datasets

This study uses LiDAR drone datasets from 12 post-landslide locations. The input data includes LiDAR point clouds and very high-resolution orthomosaics for each site.

## Training Dataset

The model's training dataset is prepared through visual interpretation, since most landslides are partially or fully visible in the drone imagery at this resolution.

## Create Orthomosaic Footprint Polygons

Create one footprint shapefile for each orthomosaic in `data`:

```powershell
python scripts/create_orthomosaic_boundaries.py
```

The script writes each output shapefile beside its source orthomosaic using the
name pattern `site_<number>_footprint.shp`, for example:

```text
data/site_01_orthomosaic.tif
data/site_01_footprint.shp
```

Each output is written in `EPSG:5235` and contains a `value` attribute set to
`0`. By default, the script writes one outer boundary polygon for the valid
orthomosaic footprint, excluding NoData/transparent background and removing
internal holes or tile seams. To write simple rectangular raster extents
instead, run:

```powershell
python scripts/create_orthomosaic_boundaries.py --mode extent
```
