import os
import random
import torch



# PATHS
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATASET_DIR = os.path.join(
    BASE_DIR,
    "..",
    "output",
    "1_training_datasets"
)


MODEL_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "..",
    "output",
    "2_trained_model"
)


PREDICT_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "..",
    "output",
    "3_predict"
)



# SITE CONFIGURATION

# Sites reserved only for independent testing
# These will NOT be used during training
TEST_SITES = [
    11,
    12
]


# Sites available for model training

TRAIN_SITES = [
    i for i in range(1,13)
    if i not in TEST_SITES
]


# DATASET SAMPLING

# Percentage of available training data used

# 1.0 = use 100%
# 0.5 = use 50%
# 0.25 = use 25%

RANDOM_SAMPLE_PERCENTAGE = 1.0



# Random seed for reproducibility

RANDOM_SEED = 42



# TRAIN / VALIDATION SITE SPLIT

# Number of sites (out of TRAIN_SITES) held out entirely for
# validation -- no tiles from these sites are used in training at
# all. This keeps validation metrics honest about performance on
# genuinely unseen terrain. The previous approach (a random 70/30
# split at the tile level) let validation tiles sit right next to
# training tiles from the same site, leaking local context and making
# validation metrics an overly optimistic stand-in for held-out-site
# performance (like TEST_SITES).
VALIDATION_SITE_COUNT = 2

_val_site_rng = random.Random(RANDOM_SEED)
_shuffled_train_sites = TRAIN_SITES.copy()
_val_site_rng.shuffle(_shuffled_train_sites)

# Sites actually tiled/used for training
FIT_SITES = sorted(_shuffled_train_sites[VALIDATION_SITE_COUNT:])

# Sites held out from training entirely, used only for validation
VAL_SITES = sorted(_shuffled_train_sites[:VALIDATION_SITE_COUNT])


# DATASETS
DATASETS = {

    "01_ortho_dataset":3,

    "02_ortho_dtm_dataset":4,

    "03_ortho_dtm_hillshade_dataset":5,

    "04_ortho_dtm_hillshade_slope_dataset":6

}



# =====================================================
# TRAINING PARAMETERS
# =====================================================

IMAGE_SIZE = 512

BATCH_SIZE = 8

EPOCHS = 100

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

# Upper bound on the BCE positive-class weight (see losses.py). This
# is derived from the training set's positive pixel ratio, but capped
# here so it doesn't push the model to over-predict landslide pixels
# and tank precision. 50 was too aggressive in practice (val precision
# stuck ~0.40-0.50 across a full 100-epoch run while recall sat
# ~0.55-0.65); try a lower cap like this first, and tune further based
# on the resulting precision/recall balance.
MAX_POS_WEIGHT = 10.0



# =====================================================
# MODEL PARAMETERS
# =====================================================

ENCODER_CHANNELS = [
    16,
    32,
    64,
    128
]


DECODER_CHANNELS = [
    128,
    64,
    32,
    16
]


BOTTLENECK_CHANNELS = 256


OUTPUT_CHANNELS = 1


# DATALOADER

NUM_WORKERS = 4


# =====================================================
# AUGMENTATION PARAMETERS (applied only to the training split)
# =====================================================

AUGMENT_TRAIN = True

# Random horizontal / vertical flip
AUGMENT_FLIP_PROBABILITY = 0.5

# Random brightness scaling (factor < 1 darkens, > 1 brightens)
AUGMENT_BRIGHTNESS_PROBABILITY = 0.5
AUGMENT_BRIGHTNESS_RANGE = (0.8, 1.2)

# Random Gaussian blur
AUGMENT_BLUR_PROBABILITY = 0.3
AUGMENT_BLUR_SIGMA_RANGE = (0.3, 1.2)


# =====================================================
# SKYSENSE (pretrained backbone) SETTINGS
# =====================================================

# Whether 01_ortho_dataset should use skysense_model.SkySenseUNet
# (a SkySense Swin V2-Huge backbone + custom decoder) instead of
# model.UNet. Only applies to 01_ortho_dataset -- the DTM/hillshade/
# slope variants have no matching pretrained SkySense modality, so
# they always use the plain UNet regardless of this flag.
USE_SKYSENSE_FOR_ORTHO = False

# Path to the downloaded skysense_model_backbone_hr.pth checkpoint
# (see https://github.com/Jack-bo1220/SkySense's README for where to
# get it -- it's a manual download, not automatable).
SKYSENSE_CHECKPOINT_PATH = os.path.join(BASE_DIR, "..", "checkpoints", "skysense_model_backbone_hr.pth")

# Epochs to train with the backbone fully frozen (decoder-only)
# before unfreezing SKYSENSE_UNFREEZE_STAGES for the remaining epochs.
SKYSENSE_FREEZE_EPOCHS = 20

# How many of the backbone's 4 stages (counting from the deepest) to
# unfreeze after SKYSENSE_FREEZE_EPOCHS.
SKYSENSE_UNFREEZE_STAGES = 1

# Learning rate for any unfrozen backbone parameters -- deliberately
# much lower than LEARNING_RATE (used for the decoder throughout, and
# for the whole plain UNet) to avoid destroying the pretrained
# weights during fine-tuning.
SKYSENSE_BACKBONE_LR = LEARNING_RATE * 0.1


# DEVICE
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


os.makedirs(
    MODEL_OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PREDICT_OUTPUT_DIR,
    exist_ok=True
)