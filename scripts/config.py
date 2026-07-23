import os
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



# TRAIN VALIDATION SPLIT

TRAIN_PERCENTAGE = 0.7
VALIDATION_PERCENTAGE = 0.3


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