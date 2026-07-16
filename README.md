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
python scripts/0_boundaries_from_orthomosaic.py
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
python scripts/0_boundaries_from_orthomosaic.py --mode extent
```

## Build the DTM

For each site with a footprint and point cloud, build a 0.1 m digital terrain model from
ground-classified LiDAR points:

```powershell
cd scripts
python 1_dtm_from_pointclouds.py
```

## Foundation-Model Landslide Segmentation Pipeline

Beyond the raw geoprocessing above, this repo evaluates five remote-sensing foundation
models (SkySense, DOFA, Scale-MAE, Prithvi-EO-2.0, SatMAE) for pixel-wise landslide
segmentation, fusing orthomosaic RGB with DTM-derived terrain features. See
[`docs/model_rationale.md`](docs/model_rationale.md) for why each model was chosen and
[`docs/workflow.md`](docs/workflow.md) for full setup and pipeline details. Quick start:

```powershell
pip install -r requirements.txt
pip install torch==2.13.0+cu132 torchvision==0.28.0+cu132 --index-url https://download.pytorch.org/whl/cu132

python scripts/prepare_terrain_features.py
python scripts/build_patch_dataset.py
python scripts/train.py --config configs/dofa_frozen.yaml
python scripts/evaluate.py --config configs/dofa_frozen.yaml
python scripts/generate_report.py
```
