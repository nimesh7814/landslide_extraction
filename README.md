# Landslide Extraction

## Introduction

Landslides have caused significant loss of life and damage in Sri Lanka. However, mapping landslides remains a largely manual task, making it time-consuming and labor-intensive — especially when numerous landslides occur simultaneously in close proximity. This project trains a semantic segmentation model to identify landslide areas from high-resolution drone orthomosaics, optionally fused with terrain features (DTM, hillshade, slope).

The resulting model can be used to build landslide inventory datasets that document landslide events, as well as to create training datasets for prediction models that rely on post-disaster data.

## Model

A single architecture is used throughout this project: **U-Net with a ResNet50 encoder pretrained on ImageNet**, built via [`segmentation_models_pytorch`](https://github.com/qubvel-org/segmentation_models.pytorch) (`scripts/model.py`). Training starts with the encoder frozen (decoder-only warm-up), then unfreezes the encoder for fine-tuning — see `FREEZE_ENCODER_EPOCHS` / `ENCODER_LR` in `scripts/config.py`.

## Data

Each of the 12 project sites needs the following files in `data/`, named `site_<NN>_<file>` (e.g. `site_01_orthomosaic.tif`):

| File | Description |
|---|---|
| `orthomosaic.tif` | RGB drone orthomosaic (required) |
| `footprint.shp` | Polygon of the valid orthomosaic extent, used to fit the tile grid |
| `landslide_annotation.shp` | Landslide polygons (or a full-coverage 0/1 layer, see `ANNOTATION_FIELD_NAME` in `1_create_train_dataset.py`) |
| `mask.tif` | Rasterized landslide mask; auto-generated from the annotation shapefile on first run if missing |
| `dtm.tif` | Digital terrain model (only needed for the DTM dataset variant) |
| `hillshade.tif` | Hillshade derived from the DTM (only needed for the hillshade dataset variant) |
| `slope.tif` | Slope derived from the DTM (only needed for the slope dataset variant) |

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install PyTorch separately, matching your CUDA version — see [pytorch.org](https://pytorch.org/get-started/locally/):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Pipeline

The pipeline has three stages, run in order from the `scripts/` directory. Every stage accepts `--model 1|2|3|4` to target a single dataset variant instead of all four:

| # | Variant | Input channels |
|---|---|---|
| 1 | `01_ortho_dataset` | RGB orthomosaic only |
| 2 | `02_ortho_dtm_dataset` | RGB + DTM |
| 3 | `03_ortho_hillshade_dataset` | RGB + hillshade |
| 4 | `04_ortho_slope_dataset` | RGB + slope |

Each variant isolates one terrain feature against the RGB baseline, so any accuracy change can be attributed to that one added feature.

### 1. Create the tiled training dataset

Tiles each site's orthomosaic (and terrain feature, if any) into 512x512 `.npy` image/mask pairs, normalized using dataset-wide min/max ranges.

```powershell
python 1_create_train_dataset.py --model 1 --site 3 7
```

`--site` limits which sites are tiled; omit both flags to tile every site for every dataset variant. Output goes to `output/1_training_datasets/<dataset_name>/`.

### 2. Train the model

```powershell
python 2_train_model.py --model 1
```

Trains the U-Net (ResNet50 encoder) on the selected dataset variant(s), with early stopping on validation IoU. Output (checkpoints, training curves, confusion matrix, metrics) goes to `output/2_trained_model/<dataset_name>/`.

### 3. Predict and evaluate

```powershell
python 3_predict.py --model 1 --sites test
```

Runs inference with the trained checkpoint, exports predicted masks as georeferenced GeoTIFFs, and (by default, via `--full-site`) stitches a full-site mask plus per-site metrics. `--sites test|train` picks which site pool to predict on (see `TEST_SITES` in `config.py`); `--site 3 7` predicts specific sites instead. Output goes to `output/3_predict/<dataset_name>/`.

### Run the whole pipeline in one command

```powershell
python run_pipeline.py --model 1
```

Runs steps 1-3 back to back for one dataset variant, or all four in sequence if `--model` is omitted (in which case each variant's tiled dataset is deleted after prediction to save disk space — pass `--no-cleanup` to keep it). See `python run_pipeline.py --help` for the full set of options.

## Configuration

All paths, site splits, training hyperparameters, augmentation settings, and model settings live in `scripts/config.py`. Notable settings:

- `TEST_SITES` — sites held out entirely for independent testing.
- `VALIDATION_SITE_COUNT` — how many of the remaining sites are held out for validation (not used in training).
- `ENCODER` / `ENCODER_WEIGHTS` — the segmentation_models_pytorch encoder and pretrained weights (`resnet50` / `imagenet`).
- `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `EARLY_STOPPING_PATIENCE` — training loop settings.
