#!/usr/bin/env python3
"""
inspect_tile_neighbors.py

Inspect the spatial neighborhood around a chosen Wakashio tile.

Purpose:
    Determine whether an offshore U-Net prediction is spatially
    consistent across neighboring/overlapping tiles.

Current test target:
    tile_00110

The script:
    1. Reads tile_metadata.csv.
    2. Finds the target tile.
    3. Finds nearby tiles based on the tiling grid offsets.
    4. Reports which nearby tiles already have U-Net predictions.
    5. Reads only those small probability PNGs one at a time.
    6. Prints their statistics.
    7. Creates a contact sheet of available probability maps.
    8. Creates a simple local mosaic using tile row/column offsets.

IMPORTANT:
    - It does NOT run the U-Net.
    - It does NOT modify predictions.
    - It never loads the full satellite raster.
    - It works with the existing DB30 predictions.

Expected structure:

SIH project/
├── satellite data/
│   ├── model_input_db30/
│   │   ├── tiles/
│   │   └── tile_metadata.csv
│   │
│   └── predictions_test5_db30/
│       ├── tile_00110/
│       │   └── probability.png
│       └── ...
│
└── inspect_tile_neighbors.py
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIG
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
    PROJECT_ROOT / "satellite data" / "neighbor_inspection"
)

TARGET_TILE = "tile_00110"

TILE_SIZE = 256

# Neighbor search:
# 1 means the immediate 8 positions in the tiling grid.
# We also include any tile whose top-left offset is within
# one stride of the target.
GRID_RADIUS = 1


# ============================================================
# HELPERS
# ============================================================

def load_metadata():
    """Load tiling metadata."""

    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_CSV}"
        )

    df = pd.read_csv(METADATA_CSV)

    required = {
        "tile_id",
        "filename",
        "row_offset",
        "col_offset",
        "valid_height",
        "valid_width",
        "min_lon",
        "max_lon",
        "min_lat",
        "max_lat",
        "center_lon",
        "center_lat",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "tile_metadata.csv is missing columns:\n"
            f"{sorted(missing)}\n\n"
            f"Found:\n{list(df.columns)}"
        )

    return df


def find_target(df):
    """Find the target tile row."""

    match = df[df["tile_id"] == TARGET_TILE]

    if match.empty:
        raise RuntimeError(
            f"{TARGET_TILE} was not found in tile_metadata.csv."
        )

    return match.iloc[0]


def find_neighbors(df, target):
    """
    Find nearby tiles using row/column pixel offsets.

    Tiles were generated with STRIDE=192, so adjacent tile starts
    differ by 192 pixels. GRID_RADIUS=1 gives the immediate
    neighborhood around the target.
    """

    target_row = int(target["row_offset"])
    target_col = int(target["col_offset"])

    # Estimate stride from metadata rather than hardcoding it.
    row_values = sorted(
        set(df["row_offset"].astype(int))
    )
    col_values = sorted(
        set(df["col_offset"].astype(int))
    )

    def smallest_positive_step(values):
        diffs = [
            b - a
            for a, b in zip(values[:-1], values[1:])
            if b > a
        ]
        return min(diffs) if diffs else 192

    row_step = smallest_positive_step(row_values)
    col_step = smallest_positive_step(col_values)

    row_radius = GRID_RADIUS * row_step
    col_radius = GRID_RADIUS * col_step

    neighbors = df[
        (
            np.abs(
                df["row_offset"].astype(int)
                - target_row
            ) <= row_radius
        )
        &
        (
            np.abs(
                df["col_offset"].astype(int)
                - target_col
            ) <= col_radius
        )
    ].copy()

    neighbors["row_delta"] = (
        neighbors["row_offset"].astype(int)
        - target_row
    )

    neighbors["col_delta"] = (
        neighbors["col_offset"].astype(int)
        - target_col
    )

    neighbors["grid_distance"] = np.sqrt(
        neighbors["row_delta"] ** 2
        + neighbors["col_delta"] ** 2
    )

    neighbors["prediction_exists"] = neighbors[
        "tile_id"
    ].apply(
        lambda tile_id: (
            PREDICTIONS_DIR
            / str(tile_id)
            / "probability.png"
        ).exists()
    )

    neighbors = neighbors.sort_values(
        ["grid_distance", "row_delta", "col_delta"]
    )

    return neighbors, row_step, col_step


def inspect_probability(path):
    """Read one small probability image and return statistics."""

    img = np.array(
        Image.open(path).convert("L"),
        dtype=np.uint8,
    )

    if img.shape != (
        TILE_SIZE,
        TILE_SIZE,
    ):
        raise ValueError(
            f"Unexpected shape for {path}: "
            f"{img.shape}"
        )

    p = img.astype(np.float32) / 255.0

    binary = p >= 0.20

    return {
        "mean_probability": float(p.mean()),
        "max_probability": float(p.max()),
        "p50": float(np.percentile(p, 50)),
        "p90": float(np.percentile(p, 90)),
        "p95": float(np.percentile(p, 95)),
        "p99": float(np.percentile(p, 99)),
        "predicted_fraction": float(binary.mean()),
        "predicted_pixels": int(binary.sum()),
    }


def normalize_position(delta, step):
    """Convert pixel offset delta to a grid index."""
    if step == 0:
        return 0
    return int(round(delta / step))


def make_contact_sheet(available):
    """Create a labeled contact sheet of available probability maps."""

    if not available:
        return

    cols = 3
    cell_size = 256
    label_height = 34

    rows = math.ceil(len(available) / cols)

    sheet = Image.new(
        "RGB",
        (
            cols * cell_size,
            rows * (cell_size + label_height),
        ),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    for i, item in enumerate(available):

        img = Image.open(
            item["probability_path"]
        ).convert("RGB")

        row = i // cols
        col = i % cols

        x = col * cell_size
        y = row * (cell_size + label_height)

        sheet.paste(
            img.resize(
                (cell_size, cell_size)
            ),
            (x, y),
        )

        label = (
            f"{item['tile_id']}  "
            f"pred={item['predicted_fraction']:.1%}"
        )

        draw.text(
            (x + 4, y + cell_size + 7),
            label,
            fill="black",
        )

    contact_path = (
        OUTPUT_DIR / "neighbor_probability_contact_sheet.png"
    )

    sheet.save(
        contact_path
    )

    print("\nContact sheet:")
    print(contact_path)


def make_mosaic(available, target):
    """
    Place available probability maps using their actual tiling-grid
    row/column offsets.

    Missing predictions remain black.
    """

    if not available:
        return

    target_row = int(target["row_offset"])
    target_col = int(target["col_offset"])

    # Estimate step from nearest available differences.
    row_steps = []
    col_steps = []

    for item in available:
        row_steps.append(
            abs(
                int(item["row_offset"])
                - target_row
            )
        )
        col_steps.append(
            abs(
                int(item["col_offset"])
                - target_col
            )
        )

    positive_rows = [x for x in row_steps if x > 0]
    positive_cols = [x for x in col_steps if x > 0]

    row_step = min(positive_rows) if positive_rows else 192
    col_step = min(positive_cols) if positive_cols else 192

    # Use normalized grid coordinates.
    positions = []

    for item in available:

        dr = (
            int(item["row_offset"])
            - target_row
        )

        dc = (
            int(item["col_offset"])
            - target_col
        )

        # Actual relative positions in tile units.
        grid_r = normalize_position(
            dr,
            row_step,
        )

        grid_c = normalize_position(
            dc,
            col_step,
        )

        positions.append(
            (item, grid_r, grid_c)
        )

    min_r = min(x[1] for x in positions)
    max_r = max(x[1] for x in positions)
    min_c = min(x[2] for x in positions)
    max_c = max(x[2] for x in positions)

    mosaic_rows = max_r - min_r + 1
    mosaic_cols = max_c - min_c + 1

    mosaic = Image.new(
        "RGB",
        (
            mosaic_cols * TILE_SIZE,
            mosaic_rows * TILE_SIZE,
        ),
        "black",
    )

    draw = ImageDraw.Draw(mosaic)

    for item, grid_r, grid_c in positions:

        img = Image.open(
            item["probability_path"]
        ).convert("RGB")

        x = (
            grid_c - min_c
        ) * TILE_SIZE

        y = (
            grid_r - min_r
        ) * TILE_SIZE

        mosaic.paste(
            img,
            (x, y),
        )

        draw.rectangle(
            (
                x,
                y,
                x + TILE_SIZE - 1,
                y + TILE_SIZE - 1,
            ),
            outline="red",
            width=2,
        )

        draw.text(
            (
                x + 5,
                y + 5,
            ),
            str(item["tile_id"]),
            fill="yellow",
        )

    mosaic_path = (
        OUTPUT_DIR / "neighbor_probability_mosaic.png"
    )

    mosaic.save(
        mosaic_path
    )

    print("Mosaic:")
    print(mosaic_path)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("WAKASHIO TILE NEIGHBOR INSPECTION")
    print("=" * 70)

    print("\nTarget:", TARGET_TILE)

    df = load_metadata()

    target = find_target(df)

    print("\nTarget geographic center:")
    print(
        "  lon:",
        float(target["center_lon"])
    )
    print(
        "  lat:",
        float(target["center_lat"])
    )

    neighbors, row_step, col_step = find_neighbors(
        df,
        target,
    )

    print("\nDetected grid spacing:")
    print("  row step:", row_step)
    print("  col step:", col_step)

    print("\nNearby tiles:")
    print(
        neighbors[
            [
                "tile_id",
                "row_delta",
                "col_delta",
                "grid_distance",
                "center_lon",
                "center_lat",
                "prediction_exists",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Inspect available predictions.
    # --------------------------------------------------------

    available = []

    print("\n" + "-" * 70)
    print("AVAILABLE PREDICTIONS")
    print("-" * 70)

    for _, row in neighbors.iterrows():

        tile_id = str(row["tile_id"])

        probability_path = (
            PREDICTIONS_DIR
            / tile_id
            / "probability.png"
        )

        if not probability_path.exists():
            continue

        stats = inspect_probability(
            probability_path
        )

        item = {
            "tile_id": tile_id,
            "row_offset": int(row["row_offset"]),
            "col_offset": int(row["col_offset"]),
            "probability_path": probability_path,
            **stats,
        }

        available.append(item)

        print(
            f"\n{tile_id}"
        )

        print(
            "  mean probability : "
            f"{stats['mean_probability']:.4f}"
        )

        print(
            "  max probability  : "
            f"{stats['max_probability']:.4f}"
        )

        print(
            "  P50              : "
            f"{stats['p50']:.4f}"
        )

        print(
            "  P95              : "
            f"{stats['p95']:.4f}"
        )

        print(
            "  P99              : "
            f"{stats['p99']:.4f}"
        )

        print(
            "  predicted @0.20  : "
            f"{stats['predicted_fraction']:.2%}"
        )

    # --------------------------------------------------------
    # Output visualizations.
    # --------------------------------------------------------

    make_contact_sheet(
        available
    )

    make_mosaic(
        available,
        target,
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        "\nUse the mosaic/contact sheet to check whether a high "
        "probability region persists across neighboring tiles."
    )

    print(
        "\nMissing neighboring predictions are expected because "
        "we have only run the U-Net on the first five test tiles."
    )


if __name__ == "__main__":
    main()
