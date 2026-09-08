#!/usr/bin/env python3
"""
predict.py

Test the trained U-Net on the 5 Wakashio-nearest tiles.

Project structure expected:

SIH project/
├── oil_spill_dataset/
├── preprocessing/
├── satellite data/
│   └── model_input/
│       ├── tiles/
│       ├── nearest_tile_statistics.csv
│       └── ...
├── SIH26143/
│   └── phase1_segmentation/
│       ├── checkpoints/
│       ├── model.py
│       └── ...
└── predict.py

This script:
    PNG tile -> /255 -> U-Net -> probability -> threshold 0.20

It processes only 5 tiles and uses a small batch size to minimize
GPU/RAM usage.

IMPORTANT:
- Edit MODEL_PATH if your checkpoint has a different name.
- If the checkpoint is a weights-only file, BUILD_MODEL_FROM_PY=True.
- If it is a complete .keras/.h5 model, set BUILD_MODEL_FROM_PY=False.
"""

from pathlib import Path
import sys
import csv

import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

# predict.py is at:
# /mnt/e/SIH project/predict.py
PROJECT_ROOT = Path(__file__).resolve().parent

SATELLITE_DIR = PROJECT_ROOT / "satellite data"
MODEL_INPUT_DIR = SATELLITE_DIR / "model_input_db30"

TILES_DIR = MODEL_INPUT_DIR / "tiles"
NEAREST_STATS_CSV = MODEL_INPUT_DIR / "nearest_tile_statistics.csv"

PHASE1_DIR = PROJECT_ROOT / "SIH26143" / "phase1_segmentation"
CHECKPOINT_DIR = PHASE1_DIR / "checkpoints"

OUTPUT_DIR = SATELLITE_DIR / "predictions_test5_db30"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# CHANGE ONLY THIS if your checkpoint has a different name.
#
# Example:
#   best_model.keras
#   best_model.weights.h5
#   unet_best.weights.h5
#
# To see what you actually have:
#   ls -lh SIH26143/phase1_segmentation/checkpoints

MODEL_PATH = CHECKPOINT_DIR / "best_model.keras"

# True  -> MODEL_PATH contains weights and model.py must construct U-Net.
# False -> MODEL_PATH is a complete Keras .keras/.h5 model.
BUILD_MODEL_FROM_PY = False

# Locked project inference threshold.
THRESHOLD = 0.20

# Conservative batch size for GPU/RAM safety.
BATCH_SIZE = 2

IMAGE_SIZE = (256, 256)

# First-stage test only.
N_TILES = 5


# ============================================================
# GPU MEMORY SAFETY
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth: enabled")
    except RuntimeError as exc:
        print("GPU memory-growth warning:", exc)

print("TensorFlow version:", tf.__version__)
print("GPUs detected:", len(gpus))


# ============================================================
# MODEL LOADING
# ============================================================

def load_complete_model():
    """Load a complete Keras model."""
    print("\nLoading complete Keras model:")
    print(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Model loaded successfully.")
    print("Input shape :", model.input_shape)
    print("Output shape:", model.output_shape)

    return model


def build_model_from_module():
    """
    Import phase1_segmentation/model.py and try common U-Net builder names.

    This path is intended for weights-only checkpoints.
    """
    if str(PHASE1_DIR) not in sys.path:
        sys.path.insert(0, str(PHASE1_DIR))

    try:
        import model as model_module
    except Exception as exc:
        raise RuntimeError(
            f"\nCould not import model.py from:\n{PHASE1_DIR}\n\n"
            f"Original error:\n{exc}"
        )

    candidate_names = [
        "build_unet",
        "build_model",
        "create_unet",
        "create_model",
        "unet",
        "UNet",
        "get_model",
    ]

    for name in candidate_names:
        builder = getattr(model_module, name, None)

        if not callable(builder):
            continue

        print(f"Trying model builder: model.{name}()")

        candidate_kwargs = [
            {"input_shape": (256, 256, 3)},
            {"img_size": (256, 256), "channels": 3},
            {"input_size": (256, 256, 3)},
            {},
        ]

        for kwargs in candidate_kwargs:
            try:
                model = builder(**kwargs)

                print("Model created successfully.")
                print("Input shape :", model.input_shape)
                print("Output shape:", model.output_shape)

                return model

            except TypeError:
                # Try another likely function signature.
                continue

            except Exception as exc:
                print(
                    f"Builder {name} failed with {kwargs}: {exc}"
                )
                break

    public_names = [
        name for name in dir(model_module)
        if not name.startswith("_")
    ]

    raise RuntimeError(
        "\nCould not find a usable U-Net builder in model.py.\n"
        f"Tried: {candidate_names}\n\n"
        f"Names available in model.py:\n{public_names}"
    )


def load_weights_model():
    """Construct U-Net and load a weights-only checkpoint."""
    model = build_model_from_module()

    print("\nLoading weights:")
    print(MODEL_PATH)

    model.load_weights(MODEL_PATH)

    print("Weights loaded successfully.")

    return model


def load_model():
    """Load model according to the configuration above."""

    if not MODEL_PATH.exists():
        available = sorted(
            p.name
            for p in CHECKPOINT_DIR.iterdir()
            if p.is_file()
        ) if CHECKPOINT_DIR.exists() else []

        raise FileNotFoundError(
            "\nMODEL_PATH does not exist:\n"
            f"{MODEL_PATH}\n\n"
            "Checkpoint directory:\n"
            f"{CHECKPOINT_DIR}\n\n"
            "Files currently found there:\n"
            f"{available}\n\n"
            "Update MODEL_PATH at the top of predict.py."
        )

    if BUILD_MODEL_FROM_PY:
        return load_weights_model()

    return load_complete_model()


# ============================================================
# TILE SELECTION
# ============================================================

def get_nearest_tiles():
    """Read the previously generated nearest-tile CSV."""

    if not NEAREST_STATS_CSV.exists():
        raise FileNotFoundError(
            "\nCould not find:\n"
            f"{NEAREST_STATS_CSV}\n\n"
            "Run inspect_wakashio_tiles.py first."
        )

    rows = []

    with NEAREST_STATS_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError(
            "nearest_tile_statistics.csv is empty."
        )

    rows.sort(
        key=lambda row: float(row["distance_km"])
    )

    return rows[:N_TILES]


# ============================================================
# INPUT PREPROCESSING
# ============================================================

def load_tile(path):
    """
    Reproduce the image part of the training pipeline:

        uint8 [0,255]
            ↓
        float32
            ↓
          /255
            ↓
        [0,1]

    The tile already has R=G=B=VV.
    """

    image = np.array(
        Image.open(path).convert("RGB"),
        dtype=np.uint8,
    )

    if image.shape != (256, 256, 3):
        raise ValueError(
            f"Unexpected tile shape for {path}:\n"
            f"{image.shape}"
        )

    # Verify that this is still the duplicated-VV format.
    if not (
        np.array_equal(image[:, :, 0], image[:, :, 1])
        and np.array_equal(image[:, :, 1], image[:, :, 2])
    ):
        print(
            f"WARNING: R/G/B are not identical in {path.name}"
        )

    # Exact normalization used in your training data pipeline.
    image = image.astype(np.float32) / 255.0

    return image


# ============================================================
# OUTPUT
# ============================================================

def save_prediction(tile_id, probability, input_image):
    """Save input, probability and binary mask."""

    tile_dir = OUTPUT_DIR / tile_id
    tile_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    probability = np.squeeze(
        np.asarray(probability)
    )

    if probability.shape != IMAGE_SIZE:
        raise ValueError(
            f"Unexpected prediction shape for {tile_id}: "
            f"{probability.shape}"
        )

    probability = np.clip(
        probability,
        0.0,
        1.0,
    )

    binary = probability >= THRESHOLD

    # Probability image
    probability_uint8 = (
        probability * 255.0
    ).round().astype(np.uint8)

    Image.fromarray(
        probability_uint8,
        mode="L",
    ).save(
        tile_dir / "probability.png"
    )

    # Binary mask
    binary_uint8 = (
        binary.astype(np.uint8) * 255
    )

    Image.fromarray(
        binary_uint8,
        mode="L",
    ).save(
        tile_dir / "binary_mask.png"
    )

    # Copy of the actual model input in displayable form.
    input_uint8 = (
        np.clip(input_image, 0.0, 1.0) * 255.0
    ).round().astype(np.uint8)

    Image.fromarray(
        input_uint8,
        mode="RGB",
    ).save(
        tile_dir / "input.png"
    )

    return {
        "tile_id": tile_id,
        "mean_probability": float(
            probability.mean()
        ),
        "max_probability": float(
            probability.max()
        ),
        "predicted_fraction": float(
            binary.mean()
        ),
        "predicted_pixels": int(
            binary.sum()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("WAKASHIO U-NET TEST-5")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nTiles:")
    print(TILES_DIR)

    print("\nCheckpoint:")
    print(MODEL_PATH)

    print("\nThreshold:", THRESHOLD)
    print("Batch size:", BATCH_SIZE)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Select nearest tiles
    # --------------------------------------------------------

    nearest_rows = get_nearest_tiles()

    print("\nSelected tiles:")

    for row in nearest_rows:
        print(
            f"  {row['tile_id']} "
            f"({float(row['distance_km']):.3f} km)"
        )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    results = []

    batch_images = []
    batch_rows = []

    for index, row in enumerate(nearest_rows):

        tile_id = row["tile_id"]
        tile_path = TILES_DIR / row["filename"]

        if not tile_path.exists():
            print(
                f"\nWARNING: missing tile:\n{tile_path}"
            )
            continue

        print(
            f"\nLoading {tile_id} "
            f"({row['distance_km']} km)..."
        )

        image = load_tile(tile_path)

        batch_images.append(image)
        batch_rows.append(row)

        is_final_tile = (
            index == len(nearest_rows) - 1
        )

        if (
            len(batch_images) == BATCH_SIZE
            or is_final_tile
        ):

            batch = np.stack(
                batch_images,
                axis=0,
            )

            print(
                f"Running inference on "
                f"{len(batch_images)} tile(s)..."
            )

            predictions = model.predict(
                batch,
                batch_size=len(batch_images),
                verbose=0,
            )

            predictions = np.asarray(
                predictions
            )

            print(
                "Prediction batch shape:",
                predictions.shape
            )

            for i, prediction in enumerate(predictions):

                result = save_prediction(
                    batch_rows[i]["tile_id"],
                    prediction,
                    batch_images[i],
                )

                result["filename"] = (
                    batch_rows[i]["filename"]
                )

                result["distance_km"] = float(
                    batch_rows[i]["distance_km"]
                )

                results.append(result)

                print(
                    f"  {result['tile_id']}: "
                    f"mean={result['mean_probability']:.4f}, "
                    f"max={result['max_probability']:.4f}, "
                    f"predicted={result['predicted_fraction']:.2%}"
                )

            # Release small arrays before the next batch.
            del batch
            del predictions

            batch_images.clear()
            batch_rows.clear()

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_csv = (
        OUTPUT_DIR / "prediction_summary.csv"
    )

    fields = [
        "tile_id",
        "filename",
        "distance_km",
        "mean_probability",
        "max_probability",
        "predicted_fraction",
        "predicted_pixels",
    ]

    if results:
        with summary_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(results)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print("Output directory:")
    print(OUTPUT_DIR)

    print("\nSummary:")
    print(summary_csv)

    print(
        "\nEach successful tile contains:"
        "\n  input.png"
        "\n  probability.png"
        "\n  binary_mask.png"
    )

    print(
        "\nNext: visually inspect the five binary masks "
        "before running the complete 225-tile inference."
    )


if __name__ == "__main__":
    main()
