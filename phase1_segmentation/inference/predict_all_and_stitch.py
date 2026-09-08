#!/usr/bin/env python3
"""
predict_all_and_stitch.py

Run the trained U-Net on ALL 225 DB30 Wakashio tiles and stitch the
predictions back into one georeferenced probability map.

Project structure:

SIH project/
├── satellite data/
│   └── model_input_db30/
│       ├── tiles/
│       │   ├── tile_00000.png
│       │   └── ...
│       ├── tile_metadata.csv
│       └── wakashio_aoi_candidate_db30.tif
│
├── satellite data/
│   └── predictions_full_db30/
│
├── SIH26143/
│   └── phase1_segmentation/
│       └── checkpoints/
│           └── best_model.keras
│
└── predict_all_and_stitch.py

What it does:

    225 PNG tiles
         ↓
    /255 normalization
         ↓
    trained U-Net
         ↓
    probability prediction
         ↓
    disk-backed accumulation
         ↓
    average overlapping predictions
         ↓
    georeferenced full AOI probability GeoTIFF
         ↓
    threshold 0.20
         ↓
    full AOI binary prediction GeoTIFF

IMPORTANT:
- Processes only small batches (BATCH_SIZE=2 by default).
- Never loads the entire satellite scene or prediction mosaic into RAM.
- Uses row_offset / col_offset from tile_metadata.csv, so the stitch is
  based on the exact tiling grid.
- Ignores padded pixels using valid_height / valid_width.
- Original PNG tiles are not modified.
- This script does NOT apply the experimental land mask by default.
  The raw stitched result is saved first. This avoids making the land-mask
  issue part of the primary inference result.
"""

from pathlib import Path
import csv
import json
import shutil
import gc

import numpy as np
import tensorflow as tf
import rasterio
from rasterio.windows import Window
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SATELLITE_DIR = PROJECT_ROOT / "satellite data"

MODEL_INPUT_DIR = SATELLITE_DIR / "model_input_db30"
TILES_DIR = MODEL_INPUT_DIR / "tiles"
METADATA_CSV = MODEL_INPUT_DIR / "tile_metadata.csv"
AOI_TIF = MODEL_INPUT_DIR / "wakashio_aoi_candidate_db30.tif"

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "SIH26143"
    / "phase1_segmentation"
    / "checkpoints"
)

MODEL_PATH = CHECKPOINT_DIR / "best_model.keras"

OUTPUT_DIR = SATELLITE_DIR / "predictions_full_db30"

PROBABILITY_TIF = OUTPUT_DIR / "wakashio_probability_db30.tif"
BINARY_TIF = OUTPUT_DIR / "wakashio_binary_db30.tif"

SUMMARY_JSON = OUTPUT_DIR / "summary.json"

# Disk-backed temporary accumulation files.
TEMP_DIR = OUTPUT_DIR / "_stitch_temp"
SUM_FILE = TEMP_DIR / "probability_sum.dat"
COUNT_FILE = TEMP_DIR / "prediction_count.dat"


# ============================================================
# MODEL / INFERENCE SETTINGS
# ============================================================

TILE_SIZE = 256
BATCH_SIZE = 2

# Same threshold used in the previous experiments.
THRESHOLD = 0.20

# Exact expected model input.
EXPECTED_INPUT_SHAPE = (256, 256, 3)

# Keep TensorFlow from grabbing all GPU memory.
ENABLE_GPU_MEMORY_GROWTH = True


# ============================================================
# GPU MEMORY SAFETY
# ============================================================

if ENABLE_GPU_MEMORY_GROWTH:
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(
                    gpu,
                    True,
                )
            print("GPU memory growth: enabled")
        except RuntimeError as exc:
            print("GPU memory-growth warning:", exc)

print("TensorFlow:", tf.__version__)
print(
    "GPUs:",
    len(tf.config.list_physical_devices("GPU")),
)


# ============================================================
# METADATA
# ============================================================

def load_metadata():
    """
    Load all tile metadata.

    Required fields:
        tile_id
        filename
        row_offset
        col_offset
        valid_height
        valid_width
    """

    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Missing metadata CSV:\n{METADATA_CSV}"
        )

    with METADATA_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "tile_id",
            "filename",
            "row_offset",
            "col_offset",
            "valid_height",
            "valid_width",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                "tile_metadata.csv is missing:\n"
                f"{sorted(missing)}"
            )

        rows = list(reader)

    if not rows:
        raise RuntimeError(
            "tile_metadata.csv is empty."
        )

    # Sort by tile_id to make execution deterministic.
    rows.sort(
        key=lambda r: r["tile_id"]
    )

    return rows


def validate_metadata(rows, aoi_width, aoi_height):
    """Check that the metadata is compatible with the AOI."""

    for row in rows:

        row_offset = int(row["row_offset"])
        col_offset = int(row["col_offset"])
        valid_height = int(row["valid_height"])
        valid_width = int(row["valid_width"])

        if valid_height <= 0 or valid_height > TILE_SIZE:
            raise ValueError(
                f"{row['tile_id']}: invalid valid_height={valid_height}"
            )

        if valid_width <= 0 or valid_width > TILE_SIZE:
            raise ValueError(
                f"{row['tile_id']}: invalid valid_width={valid_width}"
            )

        if row_offset < 0 or col_offset < 0:
            raise ValueError(
                f"{row['tile_id']}: negative offset."
            )

        if row_offset + valid_height > aoi_height:
            raise ValueError(
                f"{row['tile_id']}: tile exceeds AOI height."
            )

        if col_offset + valid_width > aoi_width:
            raise ValueError(
                f"{row['tile_id']}: tile exceeds AOI width."
            )


# ============================================================
# INPUT TILE
# ============================================================

def load_tile(path):
    """
    Read one PNG and reproduce the existing training preprocessing:

        uint8 [0,255]
            ↓
        float32
            ↓
          /255
            ↓
        [0,1]

    """
    image = np.array(
        Image.open(path).convert("RGB"),
        dtype=np.uint8,
    )

    if image.shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(
            f"Unexpected tile shape for {path.name}: "
            f"{image.shape}"
        )

    # Our generated model images are R=G=B=VV.
    if not (
        np.array_equal(
            image[:, :, 0],
            image[:, :, 1],
        )
        and
        np.array_equal(
            image[:, :, 1],
            image[:, :, 2],
        )
    ):
        print(
            f"WARNING: R/G/B differ in {path.name}"
        )

    return image.astype(
        np.float32
    ) / 255.0


# ============================================================
# MODEL
# ============================================================

def load_model():
    """Load the complete trained Keras model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    print("\nLoading model:")
    print(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Model loaded.")
    print("Input :", model.input_shape)
    print("Output:", model.output_shape)

    expected = (
        None,
        256,
        256,
        3,
    )

    if tuple(model.input_shape) != expected:
        print(
            "\nWARNING: model input shape is not exactly "
            f"{expected}."
        )

    return model


# ============================================================
# DISK-BACKED ACCUMULATORS
# ============================================================

def create_memmaps(width, height):
    """
    Create disk-backed arrays:

        probability_sum = float32
        prediction_count = uint16

    Sizes for this AOI are manageable on disk, while avoiding a large
    RAM allocation.
    """

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove stale accumulation files from a previous run.
    for path in (
        SUM_FILE,
        COUNT_FILE,
    ):
        if path.exists():
            path.unlink()

    print("\nCreating disk-backed stitch arrays...")
    print(
        "  Probability sum:",
        SUM_FILE,
    )
    print(
        "  Prediction count:",
        COUNT_FILE,
    )

    probability_sum = np.memmap(
        SUM_FILE,
        dtype=np.float32,
        mode="w+",
        shape=(height, width),
    )

    prediction_count = np.memmap(
        COUNT_FILE,
        dtype=np.uint16,
        mode="w+",
        shape=(height, width),
    )

    # Initialize without creating a second giant RAM array.
    probability_sum[:] = 0.0
    prediction_count[:] = 0

    probability_sum.flush()
    prediction_count.flush()

    return (
        probability_sum,
        prediction_count,
    )


# ============================================================
# ACCUMULATION
# ============================================================

def accumulate_batch(
    probability_sum,
    prediction_count,
    predictions,
    batch_rows,
):
    """
    Add predicted probabilities to the disk-backed mosaic.

    Only the valid (non-padded) part of each tile is accumulated.
    """

    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    for i, row in enumerate(batch_rows):

        tile_id = row["tile_id"]

        row_offset = int(
            row["row_offset"]
        )

        col_offset = int(
            row["col_offset"]
        )

        valid_height = int(
            row["valid_height"]
        )

        valid_width = int(
            row["valid_width"]
        )

        pred = predictions[i]

        # Handle common Keras output shapes:
        # (256,256,1)
        # (256,256)
        if pred.ndim == 3:
            if pred.shape[-1] != 1:
                raise ValueError(
                    f"{tile_id}: expected one output channel, "
                    f"got {pred.shape}"
                )
            pred = pred[:, :, 0]

        elif pred.ndim != 2:
            raise ValueError(
                f"{tile_id}: unexpected prediction shape {pred.shape}"
            )

        if pred.shape != (
            TILE_SIZE,
            TILE_SIZE,
        ):
            raise ValueError(
                f"{tile_id}: expected prediction "
                f"{TILE_SIZE}x{TILE_SIZE}, got {pred.shape}"
            )

        pred = np.clip(
            pred,
            0.0,
            1.0,
        )

        valid_pred = pred[
            :valid_height,
            :valid_width,
        ]

        r0 = row_offset
        r1 = row_offset + valid_height
        c0 = col_offset
        c1 = col_offset + valid_width

        # Only a small 256x256 region is touched.
        probability_sum[
            r0:r1,
            c0:c1,
        ] += valid_pred

        prediction_count[
            r0:r1,
            c0:c1,
        ] += 1


# ============================================================
# WRITE FINAL GEOTIFFS
# ============================================================

def write_outputs(
    probability_sum,
    prediction_count,
    aoi_width,
    aoi_height,
):
    """
    Write the averaged probability and thresholded binary map
    block-by-block to GeoTIFFs.

    This avoids loading the entire stitched image into RAM.
    """

    with rasterio.open(
        AOI_TIF,
    ) as src:

        profile = src.profile.copy()

        profile.update(
            driver="GTiff",
            width=aoi_width,
            height=aoi_height,
            count=1,
            dtype="uint8",
            nodata=0,
            compress="deflate",
            BIGTIFF="IF_SAFER",
        )

        print("\nWriting:")
        print(PROBABILITY_TIF)
        print(BINARY_TIF)

        with rasterio.open(
            PROBABILITY_TIF,
            "w",
            **profile,
        ) as prob_dst:

            binary_profile = profile.copy()

            with rasterio.open(
                BINARY_TIF,
                "w",
                **binary_profile,
            ) as bin_dst:

                # Safe block size.
                block_size = 512

                for r0 in range(
                    0,
                    aoi_height,
                    block_size,
                ):

                    h = min(
                        block_size,
                        aoi_height - r0,
                    )

                    for c0 in range(
                        0,
                        aoi_width,
                        block_size,
                    ):

                        w = min(
                            block_size,
                            aoi_width - c0,
                        )

                        sum_block = np.asarray(
                            probability_sum[
                                r0:r0 + h,
                                c0:c0 + w,
                            ]
                        )

                        count_block = np.asarray(
                            prediction_count[
                                r0:r0 + h,
                                c0:c0 + w,
                            ]
                        )

                        avg = np.zeros(
                            (h, w),
                            dtype=np.float32,
                        )

                        valid = (
                            count_block > 0
                        )

                        avg[valid] = (
                            sum_block[valid]
                            / count_block[valid]
                        )

                        # Convert probability [0,1] -> uint8 [0,255].
                        prob_u8 = (
                            np.clip(
                                avg,
                                0.0,
                                1.0,
                            )
                            * 255.0
                        ).round().astype(
                            np.uint8
                        )

                        binary_u8 = (
                            (
                                avg
                                >= THRESHOLD
                            ).astype(
                                np.uint8
                            )
                            * 255
                        )

                        window = Window(
                            c0,
                            r0,
                            w,
                            h,
                        )

                        prob_dst.write(
                            prob_u8,
                            1,
                            window=window,
                        )

                        bin_dst.write(
                            binary_u8,
                            1,
                            window=window,
                        )

        print("GeoTIFF outputs written.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FULL WAKASHIO DB30 U-NET INFERENCE + STITCH")
    print("=" * 70)

    # --------------------------------------------------------
    # Verify input directories/files.
    # --------------------------------------------------------

    if not TILES_DIR.exists():
        raise FileNotFoundError(
            f"Tiles directory not found:\n{TILES_DIR}"
        )

    if not AOI_TIF.exists():
        raise FileNotFoundError(
            f"AOI GeoTIFF not found:\n{AOI_TIF}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Read AOI geometry only.
    # --------------------------------------------------------

    with rasterio.open(AOI_TIF) as src:

        aoi_width = src.width
        aoi_height = src.height

        aoi_profile = src.profile.copy()

        print("\nAOI:")
        print("  Width :", aoi_width)
        print("  Height:", aoi_height)
        print("  CRS   :", src.crs)
        print("  Pixel :", src.res)
        print("  Bounds:", src.bounds)

    # --------------------------------------------------------
    # Metadata.
    # --------------------------------------------------------

    rows = load_metadata()

    print(
        "\nTiles in metadata:",
        len(rows),
    )

    validate_metadata(
        rows,
        aoi_width,
        aoi_height,
    )

    if len(rows) != 225:
        print(
            "\nWARNING: expected 225 tiles, "
            f"but metadata contains {len(rows)}."
        )

    # --------------------------------------------------------
    # Model.
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Disk-backed stitch arrays.
    # --------------------------------------------------------

    probability_sum, prediction_count = (
        create_memmaps(
            aoi_width,
            aoi_height,
        )
    )

    # --------------------------------------------------------
    # Process all tiles in small batches.
    # --------------------------------------------------------

    results = []

    total = len(rows)

    batch_images = []
    batch_rows = []

    for index, row in enumerate(
        rows,
        start=1,
    ):

        tile_path = (
            TILES_DIR
            / row["filename"]
        )

        if not tile_path.exists():
            raise FileNotFoundError(
                f"Missing tile {row['tile_id']}:\n"
                f"{tile_path}"
            )

        image = load_tile(
            tile_path
        )

        batch_images.append(
            image
        )

        batch_rows.append(
            row
        )

        is_full = (
            len(batch_images)
            >= BATCH_SIZE
        )

        is_last = (
            index == total
        )

        if is_full or is_last:

            batch = np.stack(
                batch_images,
                axis=0,
            )

            print(
                f"Inference: "
                f"{index - len(batch_images) + 1}"
                f"-{index}/{total}",
                flush=True,
            )

            predictions = model.predict(
                batch,
                batch_size=len(batch_images),
                verbose=0,
            )

            # Accumulate immediately.
            accumulate_batch(
                probability_sum,
                prediction_count,
                predictions,
                batch_rows,
            )

            # Simple batch summary.
            predictions_np = np.asarray(
                predictions
            )

            print(
                "  prediction range:",
                float(predictions_np.min()),
                "to",
                float(predictions_np.max()),
                flush=True,
            )

            # Release batch memory immediately.
            del batch
            del predictions
            del predictions_np

            batch_images.clear()
            batch_rows.clear()

            # Flush periodically so a long run is resilient.
            probability_sum.flush()
            prediction_count.flush()

            # Encourage Python/TF to release temporary objects.
            gc.collect()

    # --------------------------------------------------------
    # Write final georeferenced outputs.
    # --------------------------------------------------------

    write_outputs(
        probability_sum,
        prediction_count,
        aoi_width,
        aoi_height,
    )

    # --------------------------------------------------------
    # Final statistics.
    # --------------------------------------------------------

    counts = np.asarray(
        prediction_count
    )

    print("\nCoverage statistics:")

    covered = counts > 0

    print(
        "  Covered pixels:",
        int(covered.sum()),
    )

    print(
        "  Uncovered pixels:",
        int((~covered).sum()),
    )

    print(
        "  Minimum tile predictions/pixel:",
        int(counts[covered].min())
        if np.any(covered)
        else 0,
    )

    print(
        "  Maximum tile predictions/pixel:",
        int(counts.max()),
    )

    # Save lightweight run metadata.
    summary = {
        "model_path": str(MODEL_PATH),
        "input_directory": str(TILES_DIR),
        "aoi_tif": str(AOI_TIF),
        "num_tiles": total,
        "expected_tiles": 225,
        "tile_size": TILE_SIZE,
        "batch_size": BATCH_SIZE,
        "threshold": THRESHOLD,
        "normalization": "/255",
        "channels": "R=G=B=VV",
        "stitch_method": "mean of overlapping tile probabilities",
        "probability_output": str(PROBABILITY_TIF),
        "binary_output": str(BINARY_TIF),
        "land_mask_applied": False,
    }

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # Flush and close memmaps.
    probability_sum.flush()
    prediction_count.flush()

    del probability_sum
    del prediction_count

    # --------------------------------------------------------
    # Remove temporary disk files.
    # --------------------------------------------------------

    if TEMP_DIR.exists():
        print("\nRemoving temporary stitch files...")
        shutil.rmtree(TEMP_DIR)

    print("\n" + "=" * 70)
    print("FULL INFERENCE + STITCH COMPLETE")
    print("=" * 70)

    print("\nProbability map:")
    print(PROBABILITY_TIF)

    print("\nBinary mask:")
    print(BINARY_TIF)

    print("\nSummary:")
    print(SUMMARY_JSON)

    print(
        "\nThe final GeoTIFFs retain the AOI's original "
        "georeferencing."
    )

    print(
        "\nLand masking was intentionally not applied in this "
        "primary result."
    )


if __name__ == "__main__":
    main()
