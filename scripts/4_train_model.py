# USAGE
# python train_all_scenarios.py
#
# Trains one U-Net per input scenario (RGB, RGB_DTM, RGB_DTM_Hillshade,
# RGB_DTM_Hillshade_Slope), matching the folder layout produced by
# 2_training_data.py:
#
#   DATASET_PATH/<scenario>/images/*.npy
#   DATASET_PATH/<scenario>/masks/*.npy

import glob
import os
import time

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from pyimagesearch import config
from pyimagesearch.dataset_npy import SegmentationDatasetNPY
from pyimagesearch.model import UNet


# -----------------------------------------------------------------
# Scenario definitions: name -> number of input channels
# Must match the folder names created by 2_training_data.py
# -----------------------------------------------------------------
SCENARIOS = {

    "RGB": 3,
    "RGB_DTM": 4,
    "RGB_DTM_Hillshade": 5,
    "RGB_DTM_Hillshade_Slope": 6

}

# Base folder that contains one subfolder per scenario
# (this is the output_folder from 2_training_data.py;
# config.DATASET_PATH now points here directly)
DATASET_PATH = config.DATASET_PATH


def train_one_scenario(scenario_name, n_channels):

    print(f"\n================ {scenario_name} ({n_channels} channels) ================")

    image_dir = os.path.join(DATASET_PATH, scenario_name, "images")
    mask_dir = os.path.join(DATASET_PATH, scenario_name, "masks")

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.npy")))

    # Match each image to its mask by tile id rather than assuming
    # sort order lines up, since filenames differ ("_mask" suffix)
    mask_paths = []
    for p in image_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        mask_path = os.path.join(mask_dir, f"{base}_mask.npy")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing mask for {p}: expected {mask_path}")
        mask_paths.append(mask_path)

    print(f"[INFO] found {len(image_paths)} tiles for {scenario_name}")

    if len(image_paths) == 0:
        print(f"[WARNING] no tiles found for {scenario_name}, skipping")
        return

    # split into train/test
    split = train_test_split(
        image_paths, mask_paths,
        test_size=config.TEST_SPLIT,
        random_state=42
    )
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    # per-scenario output paths so runs don't overwrite each other
    scenario_out = os.path.join(config.TRAINING_RESULTS_PATH, scenario_name)
    os.makedirs(scenario_out, exist_ok=True)

    model_path = os.path.join(scenario_out, f"unet_{scenario_name}.pth")
    plot_path = os.path.join(scenario_out, "plot.png")
    losses_path = os.path.join(scenario_out, "losses.csv")
    test_paths_file = os.path.join(scenario_out, "test_paths.txt")

    print("[INFO] saving testing image paths...")
    with open(test_paths_file, "w") as f:
        f.write("\n".join(testImages))

    target_size = (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)

    trainDS = SegmentationDatasetNPY(
        imagePaths=trainImages, maskPaths=trainMasks,
        targetSize=target_size, augment=True
    )
    testDS = SegmentationDatasetNPY(
        imagePaths=testImages, maskPaths=testMasks,
        targetSize=target_size, augment=False
    )

    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # num_workers=0 keeps data loading in the main process. On Windows,
    # each DataLoader worker is a spawned subprocess that reimports torch
    # from scratch (reloading its CUDA DLLs), so num_workers=os.cpu_count()
    # was loading the CUDA DLL stack once per core simultaneously and
    # exhausting the paging file (WinError 1455).
    trainLoader = DataLoader(
        trainDS, shuffle=True,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=0
    )
    testLoader = DataLoader(
        testDS, shuffle=False,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=0
    )

    # Encoder channel tuple: first value is input channels for this
    # scenario, remaining values are the hidden-layer widths already
    # used in model.py's default UNet().
    encChannels = (n_channels, 4, 8, 12)
    decChannels = (12, 8, 4)

    unet = UNet(encChannels=encChannels, decChannels=decChannels).to(config.DEVICE)

    # NOTE: pos_weight reflects landslide/no-landslide pixel imbalance.
    # The masks are identical across scenarios (same annotations), so
    # this value doesn't need to be recomputed per scenario, only per
    # dataset. Adjust if your class balance changes.
    pos_weight = torch.tensor([0.2]).to(config.DEVICE)

    lossFunc = BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = Adam(unet.parameters(), lr=config.INIT_LR)

    trainSteps = max(len(trainDS) // config.BATCH_SIZE, 1)
    testSteps = max(len(testDS) // config.BATCH_SIZE, 1)

    H = {"train_loss": [], "test_loss": []}

    print(f"[INFO] training the network for {scenario_name}...")
    startTime = time.time()

    for e in tqdm(range(config.NUM_EPOCHS)):

        unet.train()
        totalTrainLoss = 0
        totalTestLoss = 0

        for (x, y) in trainLoader:
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            pred = unet(x)
            loss = lossFunc(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            totalTrainLoss += loss

        with torch.no_grad():
            unet.eval()
            for (x, y) in testLoader:
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                pred = unet(x)
                totalTestLoss += lossFunc(pred, y)

        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps

        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))

    endTime = time.time()
    print(f"[INFO] total time taken to train {scenario_name}: {endTime - startTime:.2f}s")

    torch.save(unet, model_path)

    # write raw per-epoch train/test loss values, not just the plot,
    # so the numbers themselves are available for later comparison
    with open(losses_path, "w") as f:
        f.write("epoch,train_loss,test_loss\n")
        for i, (tr, te) in enumerate(zip(H["train_loss"], H["test_loss"]), start=1):
            f.write(f"{i},{float(tr):.6f},{float(te):.6f}\n")

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title(f"Training Loss - {scenario_name}")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] saved model to {model_path}")
    print(f"[INFO] saved losses to {losses_path}")
    print(f"[INFO] saved plot to {plot_path}")


if __name__ == "__main__":

    for scenario_name, n_channels in SCENARIOS.items():
        train_one_scenario(scenario_name, n_channels)

    print("\n--------------------------------")
    print("ALL SCENARIOS TRAINED")
    print("--------------------------------\n")
