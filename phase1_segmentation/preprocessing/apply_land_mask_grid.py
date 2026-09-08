#!/usr/bin/env python3
"""
apply_land_mask_grid.py

Create an EXACTLY ALIGNED land/ocean mask using the georeferenced
Wakashio AOI GeoTIFF as the authoritative pixel grid.

This fixes the previous approach, which built a separate transform for
each PNG tile. Here the workflow is:

    georeferenced AOI GeoTIFF
            +
      Natural Earth land
            ↓
    ONE land mask on the AOI's exact grid
            ↓
    use tile_metadata.csv row/col offsets
            ↓
    extract the matching 256x256 mask
            ↓
    mask U-Net predictions

The existing U-Net predictions are NOT modified.

Expected structure:

SIH project/
├── satellite data/
│   └── model_input_db30/
│       ├── wakashio_aoi_candidate_db30.tif
│       ├── tile_metadata.csv
│       └── ...
│
├── satellite data/
│   └── predictions_test5_db30/
│       ├── tile_00109/
│       │   └── probability.png
│       └── ...
│
└── apply_land_mask_grid.py

Memory safety:
    - The full source satellite image is NEVER loaded.
    - The AOI land mask is stored as uint8 (1 byte/pixel).
    - One 256x256 prediction tile is processed at a time.
"""

from pathlib import Path
import csv
import zipfile

import numpy as np
import requests
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_INPUT_DIR = (
    PROJECT_ROOT / "satellite data" / "model_input_db30"
)

AOI_TIF = (
    MODEL_INPUT_DIR
    / "wakashio_aoi_candidate_db30.tif"
)

METADATA_CSV = (
    MODEL_INPUT_DIR / "tile_metadata.csv"
)

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "satellite data"
    / "predictions_test5_db30"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "satellite data"
    / "predictions_test5_db30_landmasked_v2"
)

LAND_MASK_TIF = (
    OUTPUT_DIR / "aoi_land_mask.tif"
)

SUMMARY_CSV = (
    OUTPUT_DIR / "land_mask_summary.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TILE_SIZE = 256
THRESHOLD = 0.20

# Natural Earth 10m land polygons.
NATURAL_EARTH_URL = (
    "https://naturalearth.s3.amazonaws.com/"
    "10m_physical/ne_10m_land.zip"
)

CACHE_DIR = PROJECT_ROOT / ".cache"
LAND_ZIP = CACHE_DIR / "ne_10m_land.zip"
LAND_EXTRACT_DIR = CACHE_DIR / "ne_10m_land"


# ============================================================
# NATURAL EARTH
# ============================================================

def get_land_shapefile():
    """Download Natural Earth 10m land once and cache it."""

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LAND_EXTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    shp_files = list(
        LAND_EXTRACT_DIR.glob("*.shp")
    )

    if shp_files:
        return shp_files[0]

    if not LAND_ZIP.exists():

        print("Downloading Natural Earth 10m land data...")
        print(NATURAL_EARTH_URL)

        response = requests.get(
            NATURAL_EARTH_URL,
            timeout=120,
        )
        response.raise_for_status()

        LAND_ZIP.write_bytes(
            response.content
        )

        print(
            f"Downloaded "
            f"{LAND_ZIP.stat().st_size / 1e6:.2f} MB"
        )

    print("Extracting Natural Earth data...")

    with zipfile.ZipFile(
        LAND_ZIP,
        "r",
    ) as zf:
        zf.extractall(LAND_EXTRACT_DIR)

    shp_files = list(
        LAND_EXTRACT_DIR.glob("*.shp")
    )

    if not shp_files:
        raise RuntimeError(
            "Natural Earth shapefile was not found after extraction."
        )

    return shp_files[0]


def load_land():
    """Read Natural Earth land polygons and convert to WGS84."""

    shp = get_land_shapefile()

    print("\nLoading land polygons:")
    print(shp)

    land = gpd.read_file(shp)

    if land.crs is None:
        raise RuntimeError(
            "Land polygons have no CRS."
        )

    land = land.to_crs("EPSG:4326")

    print(
        "Loaded land polygons:",
        len(land),
    )

    return land


# ============================================================
# METADATA
# ============================================================

def load_metadata():

    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Missing metadata CSV:\n{METADATA_CSV}"
        )

    rows = []

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

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise RuntimeError(
                "tile_metadata.csv is missing:\n"
                f"{sorted(missing)}"
            )

        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError(
            "tile_metadata.csv is empty."
        )

    return {
        row["tile_id"]: row
        for row in rows
    }


# ============================================================
# CREATE AOI LAND MASK ON EXACT AOI GRID
# ============================================================

def build_aoi_land_mask(land):

    if not AOI_TIF.exists():
        raise FileNotFoundError(
            f"Missing AOI GeoTIFF:\n{AOI_TIF}\n\n"
            "Regenerate the DB30 tiles first so the AOI GeoTIFF exists."
        )

    print("\nOpening authoritative AOI grid:")
    print(AOI_TIF)

    with rasterio.open(AOI_TIF) as src:

        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs

        if crs is None:
            raise RuntimeError(
                "AOI GeoTIFF has no CRS."
            )

        print("\nAOI grid:")
        print("  Width :", width)
        print("  Height:", height)
        print("  CRS   :", crs)
        print("  Transform:", transform)
        print("  Pixel :", src.res)

        if str(crs) != "EPSG:4326":
            raise RuntimeError(
                "This script expects the AOI GeoTIFF in EPSG:4326."
            )

        # AOI bounding box from the exact source grid.
        left = src.bounds.left
        bottom = src.bounds.bottom
        right = src.bounds.right
        top = src.bounds.top

        # Only land geometries intersecting the AOI are needed.
        aoi_box = gpd.GeoSeries.from_wkt(
            [
                (
                    f"POLYGON(("
                    f"{left} {bottom},"
                    f"{right} {bottom},"
                    f"{right} {top},"
                    f"{left} {top},"
                    f"{left} {bottom}"
                    f"))"
                )
            ],
            crs="EPSG:4326",
        ).iloc[0]

        try:
            candidates = land[
                land.intersects(aoi_box)
            ]
        except Exception:
            candidates = land

        geometries = [
            geom
            for geom in candidates.geometry
            if geom is not None and not geom.is_empty
        ]

        print(
            "Land geometries intersecting AOI:",
            len(geometries),
        )

        if not geometries:
            raise RuntimeError(
                "No land polygons intersect the AOI. "
                "The coastline/AOI CRS should be checked."
            )

        # CRITICAL:
        # Rasterize directly onto the exact AOI GeoTIFF grid.
        land_mask = rasterize(
            [(geom, 1) for geom in geometries],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        )

        print(
            "AOI land pixels:",
            int(np.count_nonzero(land_mask)),
        )

        print(
            "AOI land fraction:",
            float(np.mean(land_mask > 0)),
        )

        # Save the entire AOI land mask as a georeferenced TIFF.
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        profile = src.profile.copy()

        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            width=width,
            height=height,
            nodata=0,
            compress="deflate",
        )

        with rasterio.open(
            LAND_MASK_TIF,
            "w",
            **profile,
        ) as dst:
            dst.write(
                land_mask,
                1,
            )

    print("\nSaved authoritative AOI land mask:")
    print(LAND_MASK_TIF)

    return land_mask


# ============================================================
# PROCESS PREDICTIONS
# ============================================================

def process_prediction_tile(
    land_mask,
    row,
):

    tile_id = row["tile_id"]

    probability_path = (
        PREDICTIONS_DIR
        / tile_id
        / "probability.png"
    )

    if not probability_path.exists():
        print(
            f"WARNING: probability missing for {tile_id}"
        )
        return None

    # Read one small prediction.
    probability_u8 = np.array(
        Image.open(
            probability_path
        ).convert("L"),
        dtype=np.uint8,
    )

    if probability_u8.shape != (
        TILE_SIZE,
        TILE_SIZE,
    ):
        raise ValueError(
            f"{tile_id}: probability shape is "
            f"{probability_u8.shape}, expected "
            f"{TILE_SIZE}x{TILE_SIZE}."
        )

    probability = (
        probability_u8.astype(np.float32)
        / 255.0
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # row_offset / col_offset were generated relative to the
    # exact AOI GeoTIFF in crop_and_tile_wakashio_db30.py.
    # Therefore we take the land mask from exactly the same
    # pixel coordinates.
    # --------------------------------------------------------

    row_offset = int(row["row_offset"])
    col_offset = int(row["col_offset"])

    valid_height = int(row["valid_height"])
    valid_width = int(row["valid_width"])

    # Extract the corresponding land region.
    land_tile_valid = land_mask[
        row_offset:row_offset + valid_height,
        col_offset:col_offset + valid_width,
    ]

    if land_tile_valid.shape != (
        valid_height,
        valid_width,
    ):
        raise RuntimeError(
            f"{tile_id}: land-mask window shape "
            f"{land_tile_valid.shape} does not match "
            f"expected {(valid_height, valid_width)}."
        )

    # The prediction is always 256x256 because the tiler padded
    # partial edge tiles. Apply the SAME padding to the land mask.
    if (
        valid_height != TILE_SIZE
        or valid_width != TILE_SIZE
    ):

        pad_bottom = TILE_SIZE - valid_height
        pad_right = TILE_SIZE - valid_width

        if valid_height > 1 and valid_width > 1:
            land_tile = np.pad(
                land_tile_valid,
                (
                    (0, pad_bottom),
                    (0, pad_right),
                ),
                mode="edge",
            )
        else:
            land_tile = np.pad(
                land_tile_valid,
                (
                    (0, pad_bottom),
                    (0, pad_right),
                ),
                mode="edge",
            )

    else:
        land_tile = land_tile_valid

    land_tile = (
        land_tile.astype(bool)
    )

    if land_tile.shape != (
        TILE_SIZE,
        TILE_SIZE,
    ):
        raise RuntimeError(
            f"{tile_id}: final land mask shape "
            f"{land_tile.shape} is not 256x256."
        )

    ocean_tile = ~land_tile

    # Remove only land predictions.
    ocean_probability = np.where(
        ocean_tile,
        probability,
        0.0,
    )

    ocean_binary = (
        ocean_probability >= THRESHOLD
    )

    # --------------------------------------------------------
    # Save outputs.
    # --------------------------------------------------------

    output_tile_dir = (
        OUTPUT_DIR / tile_id
    )

    output_tile_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        (
            land_tile.astype(np.uint8)
            * 255
        ),
        mode="L",
    ).save(
        output_tile_dir
        / "land_mask.png"
    )

    Image.fromarray(
        (
            ocean_probability
            * 255.0
        ).round().astype(np.uint8),
        mode="L",
    ).save(
        output_tile_dir
        / "ocean_probability.png"
    )

    Image.fromarray(
        (
            ocean_binary.astype(np.uint8)
            * 255
        ),
        mode="L",
    ).save(
        output_tile_dir
        / "ocean_binary_mask.png"
    )

    land_fraction = float(
        land_tile.mean()
    )

    original_fraction = float(
        (probability >= THRESHOLD).mean()
    )

    ocean_fraction = float(
        ocean_binary.mean()
    )

    marine_mean = (
        float(
            probability[ocean_tile].mean()
        )
        if np.any(ocean_tile)
        else 0.0
    )

    return {
        "tile_id": tile_id,
        "land_fraction": land_fraction,
        "original_predicted_fraction": original_fraction,
        "ocean_predicted_fraction": ocean_fraction,
        "marine_mean_probability": marine_mean,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("WAKASHIO EXACT-GRID LAND MASK")
    print("=" * 70)

    print("\nInput AOI:")
    print(AOI_TIF)

    print("\nPredictions:")
    print(PREDICTIONS_DIR)

    print("\nOutput:")
    print(OUTPUT_DIR)

    print("\nThreshold:", THRESHOLD)

    if not PREDICTIONS_DIR.exists():
        raise FileNotFoundError(
            f"Predictions directory not found:\n{PREDICTIONS_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load polygons ONCE.
    # --------------------------------------------------------

    land = load_land()

    # --------------------------------------------------------
    # Load tile metadata.
    # --------------------------------------------------------

    metadata = load_metadata()

    # --------------------------------------------------------
    # Build ONE exact-grid land mask.
    # --------------------------------------------------------

    land_mask = build_aoi_land_mask(
        land
    )

    # Release GeoDataFrame and keep only the small uint8 mask.
    del land

    # --------------------------------------------------------
    # Process ONLY prediction directories that already exist.
    # This means the script currently processes the five test
    # predictions, not all 225 tiles.
    # --------------------------------------------------------

    prediction_dirs = sorted(
        p for p in PREDICTIONS_DIR.iterdir()
        if p.is_dir()
    )

    rows_to_process = []

    for prediction_dir in prediction_dirs:

        tile_id = prediction_dir.name

        if tile_id in metadata:
            rows_to_process.append(
                metadata[tile_id]
            )
        else:
            print(
                f"WARNING: {tile_id} missing from metadata; skipping."
            )

    if not rows_to_process:
        raise RuntimeError(
            "No prediction directories matched tile_metadata.csv."
        )

    print(
        "\nMatched prediction tiles:",
        len(rows_to_process),
    )

    results = []

    # --------------------------------------------------------
    # One 256x256 tile at a time.
    # --------------------------------------------------------

    for index, row in enumerate(
        rows_to_process,
        start=1,
    ):

        print(
            f"\n[{index}/{len(rows_to_process)}] "
            f"{row['tile_id']}"
        )

        result = process_prediction_tile(
            land_mask,
            row,
        )

        if result is not None:

            results.append(result)

            print(
                "  Land fraction       : "
                f"{result['land_fraction']:.2%}"
            )

            print(
                "  Original prediction : "
                f"{result['original_predicted_fraction']:.2%}"
            )

            print(
                "  Ocean prediction    : "
                f"{result['ocean_predicted_fraction']:.2%}"
            )

            print(
                "  Marine mean prob.   : "
                f"{result['marine_mean_probability']:.4f}"
            )

    # --------------------------------------------------------
    # Save summary.
    # --------------------------------------------------------

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fields = [
            "tile_id",
            "land_fraction",
            "original_predicted_fraction",
            "ocean_predicted_fraction",
            "marine_mean_probability",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print("\nLand-masked output:")
    print(OUTPUT_DIR)

    print("\nAOI land mask:")
    print(LAND_MASK_TIF)

    print("\nSummary:")
    print(SUMMARY_CSV)

    print(
        "\nOriginal U-Net predictions were NOT modified."
    )


if __name__ == "__main__":
    main()
