import torch
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, ".."))


DATASET_PATH = os.path.join(base_dir, "data", "1_training_dataset")


IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "imgs")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")


TEST_SPLIT = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PIN_MEMORY = True if DEVICE == "cuda" else False

NUM_CHANNELS = 3
NUM_CLASSES = 1

INIT_LR = 0.0001
NUM_EPOCHS = 100
BATCH_SIZE = 32

INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128



BASE_OUTPUT = os.path.join(base_dir, "data", "4_output")


TRAINING_RESULTS_PATH = os.path.join(base_dir, "data", "2_training_results")


MODEL_PATH = os.path.join(BASE_OUTPUT, "1_training_output", "unet.pth")
PLOT_PATH = os.path.join(BASE_OUTPUT, "1_training_output", "plot.png")
TEST_PATHS = os.path.join(BASE_OUTPUT, "1_training_output", "test_paths.txt")
PRED_PATHS = os.path.join(base_dir, "data", "2_predictions")
PRED_PATHS_CONT = os.path.join(base_dir, "data", "3_continuous_predictions")


TEST_PREDICTIONS_PATH = os.path.join(base_dir, "data", "3_prediction")


TEST_SITES = [11, 12]