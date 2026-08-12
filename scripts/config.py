import os
import random
import torch

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "..", "output", "1_training_datasets")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output", "2_trained_model")
PREDICT_OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output", "3_predict")

# SITE CONFIGURATION

# Sites reserved only for independent testing; never used during training.
TEST_SITES = [11, 12]

# Sites available for model training.
TRAIN_SITES = [i for i in range(1, 13) if i not in TEST_SITES]

# DATASET SAMPLING

# Fraction of available training data to use (1.0 = 100%, 0.5 = 50%, ...).
RANDOM_SAMPLE_PERCENTAGE = 1.0

# Random seed for reproducibility.
RANDOM_SEED = 42

# TRAIN / VALIDATION SITE SPLIT

# Number of TRAIN_SITES held out entirely for validation (no tiles from
# these sites are used in training), so validation reflects performance
# on genuinely unseen terrain rather than tiles next to training tiles
# from the same site.
VALIDATION_SITE_COUNT = 2

_val_site_rng = random.Random(RANDOM_SEED)
_shuffled_train_sites = TRAIN_SITES.copy()
_val_site_rng.shuffle(_shuffled_train_sites)

# Sites actually tiled/used for training.
FIT_SITES = sorted(_shuffled_train_sites[VALIDATION_SITE_COUNT:])

# Sites held out from training entirely, used only for validation.
VAL_SITES = sorted(_shuffled_train_sites[:VALIDATION_SITE_COUNT])

# DATASETS

# Each variant isolates ONE terrain feature against the RGB orthomosaic
# baseline, so any accuracy change can be attributed to that one added
# feature (dtm, hillshade, or slope) instead of a stack of confounds.
DATASETS = {
    "01_ortho_dataset": 3,
    "02_ortho_dtm_dataset": 4,
    "03_ortho_hillshade_dataset": 4,
    "04_ortho_slope_dataset": 4
}

# TRAINING PARAMETERS

IMAGE_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Upper bound on the BCE positive-class weight (see losses.py), capped so
# the model doesn't over-predict landslide pixels and tank precision.
MAX_POS_WEIGHT = 10.0

# Stop training if val IoU hasn't improved for this many consecutive epochs.
EARLY_STOPPING_PATIENCE = 12

# DATALOADER

# NUM_WORKERS > 0 loads tiles in background OS processes. Set to 0 if
# training crashes with an OSError inside a DataLoader worker on Windows.
NUM_WORKERS = 4

# AUGMENTATION PARAMETERS (applied only to the training split)

AUGMENT_TRAIN = True

# Random horizontal / vertical flip.
AUGMENT_FLIP_PROBABILITY = 0.5

# Random brightness scaling (factor < 1 darkens, > 1 brightens).
AUGMENT_BRIGHTNESS_PROBABILITY = 0.5
AUGMENT_BRIGHTNESS_RANGE = (0.8, 1.2)

# Random Gaussian blur.
AUGMENT_BLUR_PROBABILITY = 0.3
AUGMENT_BLUR_SIGMA_RANGE = (0.3, 1.2)

# MODEL SETTINGS (UNet with a pretrained ResNet50 encoder, via segmentation_models_pytorch)

OUTPUT_CHANNELS = 1

# Any encoder supported by segmentation_models_pytorch, e.g. "resnet34"
# (lighter/faster) or "resnet50" (more capacity). ResNet50 is the
# current, chosen configuration for this project.
ENCODER = "resnet50"

# Pretrained weights to initialize the encoder from ("imagenet"), or
# None for from-scratch.
ENCODER_WEIGHTS = "imagenet"

# Epochs to train with the encoder fully frozen (decoder-only warm-up)
# before unfreezing it for the remaining epochs.
FREEZE_ENCODER_EPOCHS = 4

# Learning rate for encoder parameters once unfrozen, deliberately lower
# than LEARNING_RATE to avoid destroying the pretrained ImageNet weights.
ENCODER_LR = LEARNING_RATE * 0.1

# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICT_OUTPUT_DIR, exist_ok=True)
