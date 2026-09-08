from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds


# ============================================================
# CONFIG
# ============================================================

INPUT_TIF = Path(
    r"/mnt/e/SIH project/satellite data/processed/wakashio_sigma0_vv_tc.tif"
)

OUTPUT_DIR = Path(
    r"/mnt/e/SIH project/satellite data/model_candidates"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# WAKASHIO AOI
# ============================================================

MIN_LON = 57.50
MAX_LON = 58.05

MIN_LAT = -20.75
MAX_LAT = -20.10


# ============================================================
# DESIRED dB RANGE
# ============================================================

# Experimental representation we want to test next:
# -30 dB -> 0
#   0 dB -> 255
LOWER_DB = -30.0
UPPER_DB = 0.0

OUTPUT_NAME = "wakashio_candidate_db_-30_to_0.tif"


# ============================================================
# CONVERSION
# ============================================================

def sigma0_to_db(sigma0):
    """Linear Sigma0 -> dB. Non-positive values become NaN."""
    db = np.full(sigma0.shape, np.nan, dtype=np.float32)

    valid = np.isfinite(sigma0) & (sigma0 > 0)
    db[valid] = 10.0 * np.log10(sigma0[valid])

    return db


def db_to_uint8(db, lower_db, upper_db):
    """
    Map lower_db..upper_db to 0..255 and clip outside the range.
    Invalid values become 0.
    """
    scaled = (db - lower_db) / (upper_db - lower_db)
    scaled = np.clip(scaled, 0.0, 1.0)

    result = np.rint(scaled * 255.0).astype(np.uint8)
    result[~np.isfinite(db)] = 0

    return result


# ============================================================
# MAIN
# ============================================================

with rasterio.open(INPUT_TIF) as src:

    print("=== ORIGINAL RASTER ===")
    print("Size:", src.width, "x", src.height)
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Pixel size:", src.res)

    # --------------------------------------------------------
    # Convert geographic AOI into raster window
    # --------------------------------------------------------

    window = from_bounds(
        MIN_LON,
        MIN_LAT,
        MAX_LON,
        MAX_LAT,
        transform=src.transform
    )

    window = window.round_offsets().round_lengths()

    print("\n=== AOI WINDOW ===")
    print("Window:", window)

    # --------------------------------------------------------
    # Read ONLY the AOI
    # --------------------------------------------------------

    sigma0 = src.read(
        1,
        window=window
    ).astype(np.float32)

    print("\nAOI shape:", sigma0.shape)

    # --------------------------------------------------------
    # Convert linear Sigma0 -> dB
    # --------------------------------------------------------

    db = sigma0_to_db(sigma0)

    valid = db[np.isfinite(db)]

    if valid.size == 0:
        raise RuntimeError("No valid positive Sigma0 pixels found.")

    print("\n=== AOI dB STATISTICS ===")

    for p in [1, 2, 5, 25, 50, 75, 95, 98, 99]:
        print(
            f"P{p}:",
            float(np.percentile(valid, p))
        )

    print("Min:", float(valid.min()))
    print("Max:", float(valid.max()))

    # --------------------------------------------------------
    # Create the -30 dB -> 0 dB candidate
    # --------------------------------------------------------

    print("\n=== CONVERSION ===")
    print(f"{LOWER_DB} dB -> 0")
    print(f"{UPPER_DB} dB -> 255")

    candidate = db_to_uint8(
        db,
        LOWER_DB,
        UPPER_DB
    )

    # --------------------------------------------------------
    # Write a small, georeferenced uint8 GeoTIFF
    # --------------------------------------------------------

    output_path = OUTPUT_DIR / OUTPUT_NAME

    profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        width=candidate.shape[1],
        height=candidate.shape[0],
        transform=src.window_transform(window),
        nodata=0,
        compress="deflate",
        BIGTIFF="IF_SAFER"
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(candidate, 1)

    print("\nSaved:")
    print(output_path)

    print("\nOutput shape:", candidate.shape)
    print("Output dtype:", candidate.dtype)
    print("Output min :", int(candidate.min()))
    print("Output max :", int(candidate.max()))

    print("\nNo full-scene array was loaded.")
