#!/usr/bin/env python3
"""
crop_and_tile_wakashio.py

Memory-safe cropping + tiling for the Wakashio Sentinel-1 candidate.

Pipeline:
    georeferenced uint8 GeoTIFF
        -> Wakashio AOI
        -> 256x256 overlapping tiles
        -> RGB PNGs (R=G=B=VV)
        -> tile_metadata.csv
        -> summary.json

The script never loads the full source GeoTIFF into RAM.
"""

from pathlib import Path
import csv
import json
import math

import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_TIF = Path(
    r"/mnt/e/SIH project/satellite data/model_candidates/"
    r"wakashio_candidate_db_-30_to_0.tif"
)

OUTPUT_DIR = Path(
    r"/mnt/e/SIH project/satellite data/model_input_db30"
)

TILES_DIR = OUTPUT_DIR / "tiles"
CROPPED_TIF = OUTPUT_DIR / "wakashio_aoi_candidate_db30.tif"
METADATA_CSV = OUTPUT_DIR / "tile_metadata.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"

TILE_SIZE = 256
STRIDE = 192                 # 64-pixel overlap
PAD_MODE = "reflect"

# Wakashio-focused AOI
MIN_LON = 57.65
MAX_LON = 57.90
MIN_LAT = -20.55
MAX_LAT = -20.30

# Save a georeferenced crop for easy inspection in QGIS.
SAVE_CROPPED_TIF = True


# ============================================================
# HELPERS
# ============================================================

def get_aoi_window(src):
    """Convert lon/lat AOI to a raster window."""
    window = from_bounds(
        MIN_LON,
        MIN_LAT,
        MAX_LON,
        MAX_LAT,
        transform=src.transform,
    )

    return window.round_offsets().round_lengths()


def pad_tile(tile, target_height, target_width):
    """Pad an edge tile to 256x256 without introducing a large black border."""
    height, width = tile.shape

    pad_bottom = max(0, target_height - height)
    pad_right = max(0, target_width - width)

    if pad_bottom == 0 and pad_right == 0:
        return tile

    mode = PAD_MODE

    # Reflect padding needs dimensions > 1.
    if mode == "reflect" and (height <= 1 or width <= 1):
        mode = "edge"

    return np.pad(
        tile,
        (
            (0, pad_bottom),
            (0, pad_right),
        ),
        mode=mode,
    )


def grayscale_to_rgb(tile):
    """Duplicate the VV grayscale channel into R,G,B."""
    return np.stack(
        [tile, tile, tile],
        axis=-1,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_TIF.exists():
        raise FileNotFoundError(
            f"Input TIFF not found:\n{INPUT_TIF}\n\n"
            "Edit INPUT_TIF near the top of this script."
        )

    if STRIDE <= 0 or STRIDE > TILE_SIZE:
        raise ValueError("STRIDE must be > 0 and <= TILE_SIZE.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TILES_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old generated tiles so a failed previous run cannot
    # leave stale files mixed with the new run.
    old_tiles = list(TILES_DIR.glob("tile_*.png"))
    for path in old_tiles:
        path.unlink()

    print("=" * 60)
    print("WAKASHIO CROP + TILE (DB30)")
    print("=" * 60)

    with rasterio.open(INPUT_TIF) as src:

        print("\nInput:")
        print(INPUT_TIF)

        print("\nSource raster:")
        print("  Width :", src.width)
        print("  Height:", src.height)
        print("  CRS   :", src.crs)
        print("  Pixel :", src.res)
        print("  Dtype :", src.dtypes[0])

        # ----------------------------------------------------
        # AOI window
        # ----------------------------------------------------

        aoi_window = get_aoi_window(src)

        aoi_width = int(aoi_window.width)
        aoi_height = int(aoi_window.height)

        print("\nAOI:")
        print(f"  Longitude: {MIN_LON} to {MAX_LON}")
        print(f"  Latitude : {MIN_LAT} to {MAX_LAT}")

        print("\nAOI pixel window:")
        print(" ", aoi_window)

        print("\nAOI raster size:")
        print("  Width :", aoi_width)
        print("  Height:", aoi_height)

        # ----------------------------------------------------
        # Optional small AOI GeoTIFF
        # ----------------------------------------------------

        if SAVE_CROPPED_TIF:

            print("\nSaving AOI GeoTIFF...")

            aoi_data = src.read(
                1,
                window=aoi_window,
            )

            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                width=aoi_width,
                height=aoi_height,
                count=1,
                dtype="uint8",
                transform=src.window_transform(aoi_window),
                nodata=0,
                compress="deflate",
                BIGTIFF="IF_SAFER",
            )

            with rasterio.open(CROPPED_TIF, "w", **profile) as dst:
                dst.write(aoi_data.astype(np.uint8), 1)

            del aoi_data

            print("Saved:", CROPPED_TIF)

        # ----------------------------------------------------
        # Tile grid
        # ----------------------------------------------------

        # We generate starts at 0, stride, 2*stride, ...
        # and keep the final partial tile, which will be padded.
        row_starts = list(range(0, aoi_height, STRIDE))
        col_starts = list(range(0, aoi_width, STRIDE))

        total_tiles = len(row_starts) * len(col_starts)

        print("\nTile settings:")
        print("  Tile size :", TILE_SIZE)
        print("  Stride    :", STRIDE)
        print("  Overlap   :", TILE_SIZE - STRIDE)
        print("  Rows      :", len(row_starts))
        print("  Cols      :", len(col_starts))
        print("  Total     :", total_tiles)

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata_fields = [
            "tile_id",
            "filename",
            "row_offset",
            "col_offset",
            "valid_height",
            "valid_width",
            "padded_height",
            "padded_width",
            "min_lon",
            "min_lat",
            "max_lon",
            "max_lat",
            "center_lon",
            "center_lat",
        ]

        metadata_rows = []

        tile_index = 0

        # ----------------------------------------------------
        # Process ONE tile at a time
        # ----------------------------------------------------

        for row in row_starts:
            for col in col_starts:

                valid_height = min(
                    TILE_SIZE,
                    aoi_height - row,
                )

                valid_width = min(
                    TILE_SIZE,
                    aoi_width - col,
                )

                if valid_height <= 0 or valid_width <= 0:
                    continue

                # Absolute source-raster pixel coordinates.
                window = Window(
                    col_off=int(aoi_window.col_off + col),
                    row_off=int(aoi_window.row_off + row),
                    width=int(valid_width),
                    height=int(valid_height),
                )

                # ONLY this small tile is read.
                gray = src.read(
                    1,
                    window=window,
                )

                gray = gray.astype(np.uint8)

                # Pad edge tiles.
                gray = pad_tile(
                    gray,
                    TILE_SIZE,
                    TILE_SIZE,
                )

                # Make R=G=B.
                rgb = grayscale_to_rgb(gray)

                tile_id = f"tile_{tile_index:05d}"
                filename = f"{tile_id}.png"
                tile_path = TILES_DIR / filename

                Image.fromarray(
                    rgb,
                    mode="RGB",
                ).save(
                    tile_path,
                    format="PNG",
                    optimize=False,
                )

                # rasterio.windows.bounds() returns:
                # (left, bottom, right, top)
                left, bottom, right, top = rasterio.windows.bounds(
                    window,
                    transform=src.transform,
                )

                metadata_rows.append({
                    "tile_id": tile_id,
                    "filename": filename,
                    "row_offset": int(row),
                    "col_offset": int(col),
                    "valid_height": int(valid_height),
                    "valid_width": int(valid_width),
                    "padded_height": TILE_SIZE,
                    "padded_width": TILE_SIZE,
                    "min_lon": float(left),
                    "min_lat": float(bottom),
                    "max_lon": float(right),
                    "max_lat": float(top),
                    "center_lon": float((left + right) / 2.0),
                    "center_lat": float((bottom + top) / 2.0),
                })

                tile_index += 1

                # Release small arrays immediately.
                del gray
                del rgb

                if tile_index % 25 == 0 or tile_index == total_tiles:
                    print(
                        f"Generated {tile_index}/{total_tiles} tiles",
                        flush=True,
                    )

        # ----------------------------------------------------
        # Save CSV
        # ----------------------------------------------------

        with METADATA_CSV.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=metadata_fields,
            )

            writer.writeheader()
            writer.writerows(metadata_rows)

        # ----------------------------------------------------
        # Save summary
        # ----------------------------------------------------

        summary = {
            "input_tif": str(INPUT_TIF),
            "output_directory": str(OUTPUT_DIR),
            "aoi": {
                "min_lon": MIN_LON,
                "max_lon": MAX_LON,
                "min_lat": MIN_LAT,
                "max_lat": MAX_LAT,
            },
            "aoi_raster_size": {
                "width": aoi_width,
                "height": aoi_height,
            },
            "tile_size": TILE_SIZE,
            "stride": STRIDE,
            "overlap_pixels": TILE_SIZE - STRIDE,
            "num_tiles": len(metadata_rows),
            "channels": 3,
            "channel_structure": "R=G=B=VV",
            "tile_format": "PNG uint8",
            "existing_model_loader_normalization": "/255",
            "padding": PAD_MODE,
        }

        with SUMMARY_JSON.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print("Tiles :", len(metadata_rows))
    print("Tiles :", TILES_DIR)
    print("CSV   :", METADATA_CSV)
    print("JSON  :", SUMMARY_JSON)

    if SAVE_CROPPED_TIF:
        print("AOI   :", CROPPED_TIF)

    print("\nMemory-safe mode:")
    print("Only one 256x256 tile is loaded at a time.")


if __name__ == "__main__":
    main()
