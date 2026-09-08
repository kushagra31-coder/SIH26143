#!/usr/bin/env python3
"""
inspect_wakashio_tiles.py

Find the tiles geographically closest to the approximate Wakashio spill
location and print their pixel statistics.

This script:
- reads only tile_metadata.csv
- selects the N closest tile centers to the target location
- loads one 256x256 PNG at a time
- prints statistics
- optionally creates a contact sheet for quick visual inspection

It does NOT modify the tiles or run the U-Net.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

MODEL_INPUT_DIR = Path(
    r"/mnt/e/SIH project/satellite data/model_input_db30"
)

METADATA_CSV = MODEL_INPUT_DIR / "tile_metadata.csv"
TILES_DIR = MODEL_INPUT_DIR / "tiles"

# Approximate Wakashio spill location.
TARGET_LAT = -20.438
TARGET_LON = 57.745

# Number of nearest tiles to inspect.
N_TILES = 5

# Create a contact sheet of the selected tiles.
MAKE_CONTACT_SHEET = True
CONTACT_SHEET = MODEL_INPUT_DIR / "wakashio_nearest_tiles.png"


# ============================================================
# DISTANCE
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres."""
    R = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1

    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return 2.0 * R * np.arcsin(np.sqrt(a))


# ============================================================
# TILE STATISTICS
# ============================================================

def inspect_tile(path):
    """Load one tile and calculate useful statistics."""
    img = np.array(Image.open(path))

    # Convert to grayscale using first channel because
    # R=G=B for our data.
    gray = img[:, :, 0]

    stats = {
        "shape": img.shape,
        "dtype": str(img.dtype),
        "min": int(gray.min()),
        "max": int(gray.max()),
        "mean": float(gray.mean()),
        "median": float(np.median(gray)),
    }

    for p in [1, 5, 25, 50, 75, 95, 99]:
        stats[f"P{p}"] = float(np.percentile(gray, p))

    return img, stats


# ============================================================
# MAIN
# ============================================================

def main():

    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_CSV}"
        )

    if not TILES_DIR.exists():
        raise FileNotFoundError(
            f"Tiles directory not found:\n{TILES_DIR}"
        )

    print("=" * 70)
    print("WAKASHIO NEAREST TILE INSPECTION")
    print("=" * 70)

    print("\nTarget location:")
    print(f"  Latitude : {TARGET_LAT}")
    print(f"  Longitude: {TARGET_LON}")

    # --------------------------------------------------------
    # Read only metadata, not image pixels.
    # --------------------------------------------------------

    df = pd.read_csv(METADATA_CSV)

    required = {
        "tile_id",
        "filename",
        "center_lat",
        "center_lon",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Metadata CSV is missing columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Distance from each tile center to Wakashio.
    # --------------------------------------------------------

    df["distance_km"] = haversine_km(
        TARGET_LAT,
        TARGET_LON,
        df["center_lat"].to_numpy(),
        df["center_lon"].to_numpy(),
    )

    nearest = (
        df.sort_values("distance_km")
        .head(N_TILES)
        .reset_index(drop=True)
    )

    print("\nNearest tiles:")
    print(
        nearest[
            [
                "tile_id",
                "filename",
                "distance_km",
                "center_lon",
                "center_lat",
                "min_lon",
                "max_lon",
                "min_lat",
                "max_lat",
            ]
        ].to_string(index=False)
    )

    print("\n" + "-" * 70)
    print("PIXEL STATISTICS")
    print("-" * 70)

    images = []
    records = []

    for i, row in nearest.iterrows():

        tile_path = TILES_DIR / row["filename"]

        if not tile_path.exists():
            print(
                f"\nWARNING: tile not found: {tile_path}"
            )
            continue

        image, stats = inspect_tile(tile_path)

        print(f"\n[{i + 1}] {row['tile_id']}")
        print(
            f"Distance from target: "
            f"{row['distance_km']:.3f} km"
        )
        print("Shape :", stats["shape"])
        print("Dtype :", stats["dtype"])
        print("Min   :", stats["min"])
        print("Max   :", stats["max"])
        print("Mean  :", stats["mean"])
        print("Median:", stats["median"])

        for p in [1, 5, 25, 50, 75, 95, 99]:
            print(
                f"P{p:<2}:",
                stats[f"P{p}"]
            )

        images.append(
            (
                row["tile_id"],
                image,
            )
        )

        records.append({
            "tile_id": row["tile_id"],
            "filename": row["filename"],
            "distance_km": float(row["distance_km"]),
            **stats,
        })

    # --------------------------------------------------------
    # Contact sheet
    # --------------------------------------------------------

    if MAKE_CONTACT_SHEET and images:

        tile_w = 256
        tile_h = 256

        # 2 columns, enough rows for N_TILES.
        cols = 2
        rows = math.ceil(len(images) / cols)

        sheet = Image.new(
            "RGB",
            (cols * tile_w, rows * tile_h),
            "white",
        )

        for idx, (_, image) in enumerate(images):

            tile_image = Image.fromarray(
                image,
                mode="RGB",
            )

            x = (idx % cols) * tile_w
            y = (idx // cols) * tile_h

            sheet.paste(
                tile_image,
                (x, y),
            )

        sheet.save(
            CONTACT_SHEET,
            format="PNG",
        )

        print("\nContact sheet saved:")
        print(CONTACT_SHEET)

    # --------------------------------------------------------
    # Simple CSV output of statistics.
    # --------------------------------------------------------

    stats_csv = MODEL_INPUT_DIR / "nearest_tile_statistics.csv"

    if records:
        pd.DataFrame(records).to_csv(
            stats_csv,
            index=False,
        )

        print("\nStatistics CSV saved:")
        print(stats_csv)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
