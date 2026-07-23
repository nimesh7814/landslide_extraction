import os
import sys
import time
import shutil
import argparse
import subprocess

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Needed so "from config import ..." resolves regardless of the
# directory this script is launched from.
sys.path.insert(0, SCRIPTS_DIR)

from config import DATASETS, DATASET_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full pipeline (dataset creation -> training -> prediction) "
                    "for one or all dataset variants, in one command."
    )
    parser.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Which dataset variant to run end-to-end (1-4). If omitted, all 4 are run, "
             "one after another."
    )
    parser.add_argument(
        "--num",
        type=int,
        default=6,
        help="Number of prediction samples per site, passed to 3_predict.py (default: 6)."
    )
    parser.add_argument(
        "--sites",
        choices=["test", "train"],
        default="test",
        help="Which site set 3_predict.py should sample from (default: test)."
    )
    parser.add_argument(
        "--cleanup",
        dest="cleanup",
        action="store_true",
        default=None,
        help="Force-delete each model's tiled dataset (images/masks only) after prediction. "
             "Default: on automatically when running all 4 models, off when --model is given."
    )
    parser.add_argument(
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        help="Force-disable cleanup, even when running all 4 models."
    )
    return parser.parse_args()


def run_step(script_name, extra_args):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    command = [sys.executable, script_path] + extra_args

    print(f"\n{'=' * 70}")
    print(f"Running: {' '.join(command)}")
    print(f"{'=' * 70}")

    start_time = time.time()
    result = subprocess.run(command)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n!! {script_name} failed (exit code {result.returncode}) after {elapsed:.1f}s. Stopping pipeline.")
        sys.exit(result.returncode)

    print(f"\n{script_name} finished successfully in {elapsed:.1f}s")


def delete_dataset_tiles(model_number):
    """Permanently deletes only the tiled images/masks for this dataset
    variant (output/1_training_datasets/<dataset_name>/{images,masks}),
    using shutil.rmtree (a direct OS-level delete -- this never goes
    through the Recycle Bin/Trash, on any platform). This frees disk
    space between models without touching the raw input data or the
    dataset's own 'tiles/' folder (the per-site georeferencing
    shapefiles), which is left in place."""

    dataset_names = list(DATASETS.keys())
    dataset_name = dataset_names[model_number - 1]

    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    images_dir = os.path.join(dataset_dir, "images")
    masks_dir = os.path.join(dataset_dir, "masks")

    print(f"\nCleaning up tiled dataset for {dataset_name} (deleting images/masks, keeping tiles/)...")

    deleted_any = False

    for folder in (images_dir, masks_dir):
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            print(f"  Deleted: {folder}")
            deleted_any = True

    if not deleted_any:
        print(f"  Nothing to delete for {dataset_name} (no images/masks folders found)")
        return

    if os.path.isdir(dataset_dir) and not os.listdir(dataset_dir):
        os.rmdir(dataset_dir)


def run_for_model(model_number, num_samples, sites, cleanup):
    model_args = ["--model", str(model_number)] if model_number is not None else []

    run_step("1_create_train_dataset.py", model_args)
    run_step("2_train_model.py", model_args)
    run_step("3_predict.py", model_args + ["--num", str(num_samples), "--sites", sites])

    if cleanup:
        delete_dataset_tiles(model_number)


def main():
    args = parse_args()

    pipeline_start = time.time()

    if args.model is not None:
        cleanup = args.cleanup if args.cleanup is not None else False

        print(f"Running pipeline for model {args.model} only (cleanup: {'on' if cleanup else 'off'})")
        run_for_model(args.model, args.num, args.sites, cleanup=cleanup)

    else:
        cleanup = args.cleanup if args.cleanup is not None else True

        print(f"Running pipeline for all 4 models, one after another (cleanup: {'on' if cleanup else 'off'})")

        for model_number in [1, 2, 3, 4]:
            print(f"\n\n{'#' * 70}")
            print(f"#  MODEL {model_number}")
            print(f"{'#' * 70}")

            run_for_model(model_number, args.num, args.sites, cleanup=cleanup)

    total_elapsed = time.time() - pipeline_start
    print(f"\nFull pipeline finished in {total_elapsed / 60:.1f} minute(s)")


if __name__ == "__main__":
    main()
