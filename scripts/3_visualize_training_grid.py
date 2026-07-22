import os
import random
import re

import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, ".."))

training_dataset_dir = os.path.join(base_dir, "data", "1_training_dataset")

# RGB_DTM_Hillshade_Slope has all 6 channels (R,G,B,DTM,Hillshade,Slope)
# plus the mask, so it's the one scenario that can supply every layer.
dataset_dir = os.path.join(training_dataset_dir, "RGB_DTM_Hillshade_Slope")
images_dir = os.path.join(dataset_dir, "images")
masks_dir = os.path.join(dataset_dir, "masks")

output_path = os.path.join(training_dataset_dir, "training_grid_preview.png")

N_TILES = 7
LAYERS = ["orthomosaic", "dtm", "slope", "hillshade", "mask"]

TILE_NAME_RE = re.compile(r"^site_\d+_\d+\.npy$")


def main():

    tile_files = [
        fname for fname in os.listdir(images_dir)
        if TILE_NAME_RE.match(fname)
    ]

    if not tile_files:
        raise RuntimeError(
            f"No tiles found in {images_dir}. "
            "Run 2_training_data.py first."
        )

    n_tiles = min(N_TILES, len(tile_files))
    if n_tiles < N_TILES:
        print(
            f"[WARNING] only {len(tile_files)} tile(s) available; "
            f"grid will have {n_tiles} row(s) instead of {N_TILES}."
        )

    chosen = random.sample(tile_files, n_tiles)

    fig, axes = plt.subplots(
        nrows=n_tiles,
        ncols=len(LAYERS),
        figsize=(len(LAYERS) * 3, n_tiles * 3)
    )
    axes = np.atleast_2d(axes)

    for row, fname in enumerate(chosen):

        base = os.path.splitext(fname)[0]
        site = base.rsplit("_", 1)[0]

        image = np.load(os.path.join(images_dir, fname))
        mask = np.load(os.path.join(masks_dir, f"{base}_mask.npy"))

        layers = {
            "orthomosaic": np.transpose(image[0:3], (1, 2, 0)),
            "dtm": image[3],
            "hillshade": image[4],
            "slope": image[5],
            "mask": mask
        }

        for col, layer_name in enumerate(LAYERS):

            ax = axes[row, col]
            data = layers[layer_name]

            if layer_name == "orthomosaic":
                ax.imshow(np.clip(data, 0, 1))
            else:
                ax.imshow(data, cmap="gray", vmin=0, vmax=1)

            ax.set_title(f"{site}_{layer_name}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout(pad=2.0, w_pad=1.5, h_pad=1.5)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] saved grid preview -> {output_path}")


if __name__ == "__main__":
    main()
