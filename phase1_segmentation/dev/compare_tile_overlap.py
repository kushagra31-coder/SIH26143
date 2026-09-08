#!/usr/bin/env python3
"""
compare_tile_overlap.py

Compare the REAL 64-pixel overlap between tile_00109 and tile_00110.

Your tiler used:
    TILE_SIZE = 256
    STRIDE    = 192

Therefore adjacent horizontal tiles overlap by:
    256 - 192 = 64 pixels

From the tile metadata, this script verifies that relationship and then
compares:

    tile_00109 rightmost 64 columns
                    vs
    tile_00110 leftmost 64 columns

It compares:
    1. U-Net probability maps
    2. Binary masks at threshold 0.20

Outputs:
    satellite data/overlap_inspection/
        overlap_probability_comparison.png
        overlap_binary_comparison.png
        overlap_metrics.txt

This script only loads tiny 256x64 arrays, so it is very low-memory.
It does NOT run the U-Net and does NOT modify any existing files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_INPUT_DIR = (
    PROJECT_ROOT / "satellite data" / "model_input_db30"
)

METADATA_CSV = (
    MODEL_INPUT_DIR / "tile_metadata.csv"
)

PREDICTIONS_DIR = (
    PROJECT_ROOT / "satellite data" / "predictions_test5_db30"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "satellite data" / "overlap_inspection"
)

TILE_A = "tile_00109"
TILE_B = "tile_00110"

TILE_SIZE = 256
EXPECTED_OVERLAP = 64
THRESHOLD = 0.20


# ============================================================
# HELPERS
# ============================================================

def load_metadata():
    """Load tile metadata."""

    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_CSV}"
        )

    df = pd.read_csv(METADATA_CSV)

    required = {
        "tile_id",
        "row_offset",
        "col_offset",
        "valid_width",
        "valid_height",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Metadata is missing columns: {sorted(missing)}"
        )

    return df


def get_row(df, tile_id):
    """Get one tile metadata row."""

    match = df[df["tile_id"] == tile_id]

    if match.empty:
        raise RuntimeError(
            f"{tile_id} not found in tile_metadata.csv"
        )

    return match.iloc[0]


def load_probability(tile_id):
    """Load one probability PNG as float32 [0,1]."""

    path = (
        PREDICTIONS_DIR
        / tile_id
        / "probability.png"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction not found:\n{path}"
        )

    img = np.array(
        Image.open(path).convert("L"),
        dtype=np.uint8,
    )

    if img.shape != (
        TILE_SIZE,
        TILE_SIZE,
    ):
        raise ValueError(
            f"{tile_id}: expected "
            f"{TILE_SIZE}x{TILE_SIZE}, got {img.shape}"
        )

    return img.astype(np.float32) / 255.0


def make_gray_u8(arr):
    """Convert [0,1] array to uint8 grayscale."""
    return (
        np.clip(arr, 0.0, 1.0) * 255.0
    ).round().astype(np.uint8)


def make_binary_u8(arr):
    """Threshold a probability array."""
    return (
        (arr >= THRESHOLD).astype(np.uint8)
        * 255
    )


def save_side_by_side(left, right, output_path, title_left, title_right):
    """
    Save two 64x256 grayscale images side-by-side with labels.
    """

    left_img = Image.fromarray(
        make_gray_u8(left),
        mode="L",
    ).convert("RGB")

    right_img = Image.fromarray(
        make_gray_u8(right),
        mode="L",
    ).convert("RGB")

    label_h = 35

    canvas = Image.new(
        "RGB",
        (
            left_img.width + right_img.width,
            left_img.height + label_h,
        ),
        "white",
    )

    canvas.paste(
        left_img,
        (0, label_h),
    )

    canvas.paste(
        right_img,
        (left_img.width, label_h),
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (5, 8),
        title_left,
        fill="black",
    )

    draw.text(
        (left_img.width + 5, 8),
        title_right,
        fill="black",
    )

    canvas.save(output_path)


def calculate_metrics(a, b):
    """
    Calculate overlap similarity metrics.

    Correlation:
        similarity of spatial probability patterns.

    Mean absolute difference:
        average probability disagreement.

    Threshold IoU:
        agreement of positive pixels at threshold 0.20.

    Positive fractions:
        fraction predicted positive in each overlap strip.
    """

    a_flat = a.ravel()
    b_flat = b.ravel()

    diff = np.abs(a_flat - b_flat)

    if (
        np.std(a_flat) > 0
        and np.std(b_flat) > 0
    ):
        correlation = float(
            np.corrcoef(a_flat, b_flat)[0, 1]
        )
    else:
        correlation = float("nan")

    mean_abs_diff = float(
        np.mean(diff)
    )

    mask_a = a >= THRESHOLD
    mask_b = b >= THRESHOLD

    intersection = np.logical_and(
        mask_a,
        mask_b,
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b,
    ).sum()

    iou = (
        float(intersection / union)
        if union > 0
        else 1.0
    )

    return {
        "correlation": correlation,
        "mean_abs_difference": mean_abs_diff,
        "iou_at_threshold": iou,
        "positive_fraction_a": float(mask_a.mean()),
        "positive_fraction_b": float(mask_b.mean()),
        "positive_pixels_a": int(mask_a.sum()),
        "positive_pixels_b": int(mask_b.sum()),
        "intersection_pixels": int(intersection),
        "union_pixels": int(union),
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
    print("REAL TILE OVERLAP COMPARISON")
    print("=" * 70)

    df = load_metadata()

    row_a = get_row(
        df,
        TILE_A,
    )

    row_b = get_row(
        df,
        TILE_B,
    )

    # --------------------------------------------------------
    # Verify that these really are horizontally adjacent.
    # --------------------------------------------------------

    row_a_index = int(row_a["row_offset"])
    col_a_index = int(row_a["col_offset"])

    row_b_index = int(row_b["row_offset"])
    col_b_index = int(row_b["col_offset"])

    row_delta = row_b_index - row_a_index
    col_delta = col_b_index - col_a_index

    width_a = int(row_a["valid_width"])
    width_b = int(row_b["valid_width"])

    print("\nTile A:", TILE_A)
    print(
        "  row_offset:",
        row_a_index,
        "col_offset:",
        col_a_index,
    )
    print("  width:", width_a)

    print("\nTile B:", TILE_B)
    print(
        "  row_offset:",
        row_b_index,
        "col_offset:",
        col_b_index,
    )
    print("  width:", width_b)

    print("\nOffset difference:")
    print("  row delta:", row_delta)
    print("  col delta:", col_delta)

    # The expected relationship for horizontal neighbors is:
    # same row and B starts 192 px after A.
    if row_delta != 0:
        raise RuntimeError(
            f"{TILE_A} and {TILE_B} are not on the same row."
        )

    actual_col_delta = col_delta

    if actual_col_delta <= 0:
        raise RuntimeError(
            f"{TILE_B} is not to the right of {TILE_A}."
        )

    actual_overlap = (
        width_a - actual_col_delta
    )

    print(
        "\nCalculated horizontal overlap:",
        actual_overlap,
        "pixels",
    )

    if actual_overlap != EXPECTED_OVERLAP:
        print(
            "\nWARNING: overlap is not 64 pixels. "
            "The script will use the actual metadata-derived overlap."
        )

    # --------------------------------------------------------
    # Load only the two tiny probability images.
    # --------------------------------------------------------

    prob_a = load_probability(TILE_A)
    prob_b = load_probability(TILE_B)

    # --------------------------------------------------------
    # Extract TRUE common geographic columns.
    #
    # A: rightmost overlap columns
    # B: leftmost overlap columns
    # --------------------------------------------------------

    overlap_a = prob_a[
        :,
        TILE_SIZE - actual_overlap:
        TILE_SIZE,
    ]

    overlap_b = prob_b[
        :,
        :actual_overlap,
    ]

    if overlap_a.shape != overlap_b.shape:
        raise RuntimeError(
            f"Overlap shapes do not match:\n"
            f"A: {overlap_a.shape}\n"
            f"B: {overlap_b.shape}"
        )

    print(
        "\nOverlap array shape:",
        overlap_a.shape,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        overlap_a,
        overlap_b,
    )

    print("\n" + "-" * 70)
    print("OVERLAP METRICS")
    print("-" * 70)

    print(
        "Probability correlation :",
        f"{metrics['correlation']:.4f}",
    )

    print(
        "Mean absolute difference:",
        f"{metrics['mean_abs_difference']:.4f}",
    )

    print(
        f"IoU @ {THRESHOLD:.2f}            :",
        f"{metrics['iou_at_threshold']:.4f}",
    )

    print(
        "Positive fraction A     :",
        f"{metrics['positive_fraction_a']:.2%}",
    )

    print(
        "Positive fraction B     :",
        f"{metrics['positive_fraction_b']:.2%}",
    )

    print(
        "Positive pixels A       :",
        metrics["positive_pixels_a"],
    )

    print(
        "Positive pixels B       :",
        metrics["positive_pixels_b"],
    )

    print(
        "Intersection pixels     :",
        metrics["intersection_pixels"],
    )

    print(
        "Union pixels            :",
        metrics["union_pixels"],
    )

    # --------------------------------------------------------
    # Save probability comparison.
    # --------------------------------------------------------

    probability_path = (
        OUTPUT_DIR
        / "overlap_probability_comparison.png"
    )

    save_side_by_side(
        overlap_a,
        overlap_b,
        probability_path,
        f"{TILE_A} right {actual_overlap}px",
        f"{TILE_B} left {actual_overlap}px",
    )

    # --------------------------------------------------------
    # Save binary comparison.
    # --------------------------------------------------------

    binary_a = (
        overlap_a >= THRESHOLD
    ).astype(np.float32)

    binary_b = (
        overlap_b >= THRESHOLD
    ).astype(np.float32)

    binary_path = (
        OUTPUT_DIR
        / "overlap_binary_comparison.png"
    )

    save_side_by_side(
        binary_a,
        binary_b,
        binary_path,
        f"{TILE_A} binary",
        f"{TILE_B} binary",
    )

    # --------------------------------------------------------
    # Save textual metrics.
    # --------------------------------------------------------

    metrics_path = (
        OUTPUT_DIR
        / "overlap_metrics.txt"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "REAL TILE OVERLAP COMPARISON\n"
        )
        f.write(
            "============================\n\n"
        )

        f.write(
            f"Tile A: {TILE_A}\n"
        )
        f.write(
            f"Tile B: {TILE_B}\n"
        )
        f.write(
            f"Row delta: {row_delta}\n"
        )
        f.write(
            f"Column delta: {col_delta}\n"
        )
        f.write(
            f"Actual overlap: {actual_overlap} pixels\n\n"
        )

        for key, value in metrics.items():
            f.write(
                f"{key}: {value}\n"
            )

    print("\nSaved:")
    print(probability_path)
    print(binary_path)
    print(metrics_path)

    # --------------------------------------------------------
    # Interpretation guide.
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    print(
        "High correlation + low mean difference + high IoU "
        "means the model response is spatially consistent "
        "in the shared region."
    )

    print(
        "Low correlation or near-zero IoU means the response "
        "changes substantially between overlapping tiles."
    )

    print(
        "\nThis is a consistency test, NOT proof that the "
        "predicted region is oil."
    )

    print("\nDONE")


if __name__ == "__main__":
    main()
