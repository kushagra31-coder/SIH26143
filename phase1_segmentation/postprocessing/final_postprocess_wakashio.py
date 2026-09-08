#!/usr/bin/env python3
"""
final_postprocess_wakashio.py

FINAL GEOSPATIAL POST-PROCESSING for the full DB30 U-Net result.

Pipeline:
    stitched U-Net probability GeoTIFF
                ↓
       exact-grid land mask
                ↓
       ocean-only probability
                ↓
        threshold = 0.20
                ↓
       connected components
                ↓
       remove tiny components
                ↓
       candidate region polygons
                ↓
       calculate area / centroid / probability / distance
                ↓
       GeoJSON + GeoTIFF + CSV

This script is intentionally a POST-PROCESSING step:
- it does not retrain the U-Net
- it does not change the model input
- it does not modify the stitched probability raster
- it does not claim that candidate regions are confirmed oil

Memory:
- Current AOI is only ~2783 x 2783 pixels, so arrays are small.
- Land mask and probability are loaded as compact numpy arrays.
- No Sentinel-1 source scene is loaded.
"""

from pathlib import Path
import csv
import json
import zipfile

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, shapes
from shapely.geometry import box, shape
from scipy import ndimage


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "satellite data"
    / "predictions_full_db30"
)

PROBABILITY_TIF = (
    PREDICTIONS_DIR
    / "wakashio_probability_db30.tif"
)

# Using the AOI GeoTIFF as the exact reference grid.
MODEL_INPUT_DIR = (
    PROJECT_ROOT
    / "satellite data"
    / "model_input_db30"
)

AOI_TIF = (
    MODEL_INPUT_DIR
    / "wakashio_aoi_candidate_db30.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "satellite data"
    / "final_postprocess_db30"
)

OCEAN_PROBABILITY_TIF = (
    OUTPUT_DIR
    / "ocean_probability_db30.tif"
)

RAW_MARINE_MASK_TIF = (
    OUTPUT_DIR
    / "marine_binary_raw_db30.tif"
)

FILTERED_MARINE_MASK_TIF = (
    OUTPUT_DIR
    / "marine_binary_filtered_db30.tif"
)

COMPONENTS_TIF = (
    OUTPUT_DIR
    / "marine_components_db30.tif"
)

GEOJSON_PATH = (
    OUTPUT_DIR
    / "oil_spill_candidates_db30.geojson"
)

STATS_CSV = (
    OUTPUT_DIR
    / "oil_spill_candidates_db30.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "final_postprocess_summary.json"
)

LAND_MASK_TIF = (
    OUTPUT_DIR
    / "aoi_land_mask_db30.tif"
)


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

THRESHOLD = 0.20

# Remove extremely small connected components only.
# 100 pixels at ~10 m spacing is roughly 0.01 km².
MIN_PIXELS = 100

# 8-connected component labeling.
CONNECTIVITY = 8

# Approximate Wakashio reference point used earlier in the project.
WAKASHIO_LON = 57.745
WAKASHIO_LAT = -20.438

# Optional light morphological cleanup.
#
# False by default because we do not want to invent or erase spatial
# structures before inspecting the raw model result.
USE_MORPHOLOGY = False

# If morphology is enabled:
MORPHOLOGY_ITERATIONS = 1

# Natural Earth 10m land.
NATURAL_EARTH_URL = (
    "https://naturalearth.s3.amazonaws.com/"
    "10m_physical/ne_10m_land.zip"
)

CACHE_DIR = PROJECT_ROOT / ".cache"
LAND_ZIP = CACHE_DIR / "ne_10m_land.zip"
LAND_EXTRACT_DIR = CACHE_DIR / "ne_10m_land"


# ============================================================
# NATURAL EARTH LAND
# ============================================================

def get_land_shapefile():
    """Return cached Natural Earth 10m land shapefile."""

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

        print("Downloading Natural Earth 10m land...")
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

    print("Extracting Natural Earth...")

    with zipfile.ZipFile(
        LAND_ZIP,
        "r",
    ) as zf:
        zf.extractall(
            LAND_EXTRACT_DIR
        )

    shp_files = list(
        LAND_EXTRACT_DIR.glob("*.shp")
    )

    if not shp_files:
        raise RuntimeError(
            "Natural Earth shapefile not found."
        )

    return shp_files[0]


def load_land_for_aoi(aoi_bounds):
    """Load only land polygons that intersect the AOI."""

    shp = get_land_shapefile()

    land = gpd.read_file(shp)

    if land.crs is None:
        raise RuntimeError(
            "Natural Earth land layer has no CRS."
        )

    land = land.to_crs("EPSG:4326")

    aoi_box = box(
        aoi_bounds.left,
        aoi_bounds.bottom,
        aoi_bounds.right,
        aoi_bounds.top,
    )

    try:
        land = land[
            land.intersects(aoi_box)
        ]
    except Exception:
        pass

    if land.empty:
        raise RuntimeError(
            "No Natural Earth land geometry intersects the AOI."
        )

    print(
        "Land geometries intersecting AOI:",
        len(land),
    )

    return land


# ============================================================
# EXACT-GRID LAND MASK
# ============================================================

def create_land_mask(
    aoi_width,
    aoi_height,
    transform,
    bounds,
):
    """
    Rasterize land directly onto the exact AOI GeoTIFF grid.

    True  = land
    False = water
    """

    land = load_land_for_aoi(
        bounds
    )

    geometries = [
        geom
        for geom in land.geometry
        if geom is not None and not geom.is_empty
    ]

    land_mask = rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=(
            aoi_height,
            aoi_width,
        ),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)

    return land_mask


# ============================================================
# DISTANCE
# ============================================================

def distance_km(
    lon,
    lat,
):
    """
    Approximate great-circle distance from the Wakashio reference point.

    Good enough for ranking nearby candidates.
    """

    lat1 = np.deg2rad(WAKASHIO_LAT)
    lat2 = np.deg2rad(lat)

    dlat = np.deg2rad(
        lat - WAKASHIO_LAT
    )

    dlon = np.deg2rad(
        lon - WAKASHIO_LON
    )

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return float(
        6371.0
        * 2.0
        * np.arcsin(
            np.sqrt(a)
        )
    )


# ============================================================
# AREA
# ============================================================

def rough_pixel_area_km2(
    transform,
):
    """
    Rough pixel-area estimate at the Wakashio latitude.

    Exact region area is calculated later from projected polygons.
    """

    xres = abs(
        transform.a
    )

    yres = abs(
        transform.e
    )

    km_per_degree_lat = 111.32

    km_per_degree_lon = (
        111.32
        * np.cos(
            np.deg2rad(
                WAKASHIO_LAT
            )
        )
    )

    return (
        xres
        * km_per_degree_lon
        * yres
        * km_per_degree_lat
    )


# ============================================================
# COMPONENT STATISTICS
# ============================================================

def calculate_regions(
    labels,
    probability,
    transform,
    crs,
):
    """Calculate statistics for every retained component."""

    component_ids = np.unique(
        labels
    )

    component_ids = component_ids[
        component_ids != 0
    ]

    if len(component_ids) == 0:
        return []

    flat_labels = labels.ravel()
    flat_probability = probability.ravel()

    pixel_counts = np.bincount(
        flat_labels
    )

    probability_sums = np.bincount(
        flat_labels,
        weights=flat_probability,
    )

    maxima = np.zeros(
        int(labels.max()) + 1,
        dtype=np.float32,
    )

    valid = flat_labels > 0

    np.maximum.at(
        maxima,
        flat_labels[valid],
        flat_probability[valid],
    )

    rough_pixel_area = (
        rough_pixel_area_km2(
            transform
        )
    )

    regions = []

    for component_id in component_ids:

        count = int(
            pixel_counts[component_id]
        )

        if count < MIN_PIXELS:
            continue

        rows, cols = np.where(
            labels == component_id
        )

        mean_probability = float(
            probability_sums[component_id]
            / count
        )

        max_probability = float(
            maxima[component_id]
        )

        mean_row = float(
            rows.mean()
        )

        mean_col = float(
            cols.mean()
        )

        lon, lat = rasterio.transform.xy(
            transform,
            mean_row,
            mean_col,
            offset="center",
        )

        min_row = int(
            rows.min()
        )

        max_row = int(
            rows.max()
        )

        min_col = int(
            cols.min()
        )

        max_col = int(
            cols.max()
        )

        min_lon, max_lat = rasterio.transform.xy(
            transform,
            min_row,
            min_col,
            offset="ul",
        )

        max_lon, min_lat = rasterio.transform.xy(
            transform,
            max_row + 1,
            max_col + 1,
            offset="lr",
        )

        regions.append({
            "component_id": int(component_id),
            "pixel_count": count,
            "approx_area_km2": (
                count
                * rough_pixel_area
            ),
            "mean_probability": mean_probability,
            "max_probability": max_probability,
            "centroid_lon": float(lon),
            "centroid_lat": float(lat),
            "min_lon": float(min_lon),
            "min_lat": float(min_lat),
            "max_lon": float(max_lon),
            "max_lat": float(max_lat),
            "distance_to_wakashio_km": distance_km(
                lon,
                lat,
            ),
        })

    # Rank primarily by marine area, then mean probability,
    # then distance to the known event area.
    regions.sort(
        key=lambda x: (
            -x["pixel_count"],
            -x["mean_probability"],
            x["distance_to_wakashio_km"],
        )
    )

    for rank, region in enumerate(
        regions,
        start=1,
    ):
        region["rank"] = rank

    return regions


# ============================================================
# RASTER OUTPUT
# ============================================================

def save_raster(
    data,
    reference_profile,
    output_path,
    dtype,
):
    """Save an array using the AOI georeferencing."""

    profile = reference_profile.copy()

    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=0,
        compress="deflate",
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:
        dst.write(
            data.astype(dtype),
            1,
        )


# ============================================================
# POLYGONIZATION
# ============================================================

def polygonize_regions(
    labels,
    regions,
    transform,
    crs,
):
    """Convert retained component labels into polygons."""

    if not regions:
        return gpd.GeoDataFrame(
            geometry=[],
            crs=crs,
        )

    retained_ids = {
        int(region["component_id"])
        for region in regions
    }

    geometries = []
    component_ids = []

    for component_id in retained_ids:

        component_mask = (
            labels == component_id
        ).astype(np.uint8)

        for geom_mapping, value in shapes(
            component_mask,
            mask=component_mask.astype(bool),
            transform=transform,
        ):

            if int(value) != 1:
                continue

            geom = shape(
                geom_mapping
            )

            if geom.is_empty:
                continue

            geometries.append(
                geom
            )

            component_ids.append(
                component_id
            )

    gdf = gpd.GeoDataFrame(
        {
            "component_id":
                component_ids
        },
        geometry=geometries,
        crs=crs,
    )

    if gdf.empty:
        return gdf

    stats_by_id = {
        int(region["component_id"]):
            region
        for region in regions
    }

    for field in [
        "rank",
        "pixel_count",
        "approx_area_km2",
        "mean_probability",
        "max_probability",
        "centroid_lon",
        "centroid_lat",
        "distance_to_wakashio_km",
    ]:
        gdf[field] = (
            gdf["component_id"]
            .map(
                lambda cid:
                stats_by_id[
                    int(cid)
                ][field]
            )
        )

    # Exact area in UTM Zone 40S.
    # Suitable for the Mauritius/Wakashio AOI.
    projected = gdf.to_crs(
        "EPSG:32740"
    )

    gdf["area_km2"] = (
        projected.geometry.area
        / 1_000_000.0
    )

    # Back to WGS84 for interoperable GeoJSON.
    gdf = gdf.to_crs(
        "EPSG:4326"
    )

    return gdf


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 75)
    print("FINAL WAKASHIO DB30 POST-PROCESSING")
    print("=" * 75)

    if not PROBABILITY_TIF.exists():
        raise FileNotFoundError(
            f"Probability raster not found:\n{PROBABILITY_TIF}"
        )

    if not AOI_TIF.exists():
        raise FileNotFoundError(
            f"AOI reference raster not found:\n{AOI_TIF}"
        )

    # --------------------------------------------------------
    # Read full stitched probability map.
    # Current raster is only ~7.7M pixels.
    # --------------------------------------------------------

    with rasterio.open(
        PROBABILITY_TIF
    ) as prob_src:

        probability_u8 = prob_src.read(
            1
        )

        reference_profile = (
            prob_src.profile.copy()
        )

        transform = prob_src.transform
        crs = prob_src.crs
        width = prob_src.width
        height = prob_src.height
        bounds = prob_src.bounds

    probability = (
        probability_u8.astype(
            np.float32
        )
        / 255.0
    )

    print("\nProbability raster:")
    print("  Size:", width, "x", height)
    print("  CRS :", crs)
    print("  Bounds:", bounds)
    print("  Threshold:", THRESHOLD)

    # --------------------------------------------------------
    # Verify the AOI reference grid matches the probability map.
    # --------------------------------------------------------

    with rasterio.open(
        AOI_TIF
    ) as aoi_src:

        if (
            aoi_src.width != width
            or aoi_src.height != height
        ):
            raise RuntimeError(
                "AOI grid dimensions do not match "
                "the stitched probability raster."
            )

        if aoi_src.crs != crs:
            raise RuntimeError(
                "AOI CRS does not match probability raster."
            )

        if not np.allclose(
            np.array(aoi_src.transform),
            np.array(transform),
        ):
            raise RuntimeError(
                "AOI transform does not match "
                "probability raster transform."
            )

        if not np.allclose(
            [
                aoi_src.bounds.left,
                aoi_src.bounds.bottom,
                aoi_src.bounds.right,
                aoi_src.bounds.top,
            ],
            [
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
            ],
            rtol=0.0,
            atol=1e-8,
        ):
            raise RuntimeError(
                "AOI bounds do not match probability raster."
            )

    print(
        "\nAOI grid verification: PASSED"
    )

    # --------------------------------------------------------
    # Exact-grid land mask.
    # --------------------------------------------------------

    print("\nCreating exact-grid land mask...")

    land_mask = create_land_mask(
        width,
        height,
        transform,
        bounds,
    )

    ocean_mask = ~land_mask

    print(
        "Land fraction:",
        float(land_mask.mean()),
    )

    print(
        "Ocean fraction:",
        float(ocean_mask.mean()),
    )

    save_raster(
        land_mask.astype(np.uint8) * 255,
        reference_profile,
        LAND_MASK_TIF,
        "uint8",
    )

    print(
        "Land mask:",
        LAND_MASK_TIF,
    )

    # --------------------------------------------------------
    # Ocean-only probability.
    # --------------------------------------------------------

    ocean_probability = np.where(
        ocean_mask,
        probability,
        0.0,
    ).astype(np.float32)

    # Save ocean probability as uint8 for easy QGIS visualization.
    save_raster(
        (
            ocean_probability * 255.0
        ).round().astype(np.uint8),
        reference_profile,
        OCEAN_PROBABILITY_TIF,
        "uint8",
    )

    # --------------------------------------------------------
    # Threshold.
    # --------------------------------------------------------

    marine_binary = (
        ocean_probability
        >= THRESHOLD
    )

    print(
        "\nMarine pixels above threshold:",
        int(marine_binary.sum()),
    )

    print(
        "Marine candidate fraction:",
        float(marine_binary.mean()),
    )

    save_raster(
        marine_binary.astype(np.uint8) * 255,
        reference_profile,
        RAW_MARINE_MASK_TIF,
        "uint8",
    )

    # --------------------------------------------------------
    # Optional morphology.
    # --------------------------------------------------------

    processed_binary = marine_binary.copy()

    if USE_MORPHOLOGY:

        print(
            "\nApplying light morphological cleanup..."
        )

        processed_binary = (
            ndimage.binary_opening(
                processed_binary,
                iterations=MORPHOLOGY_ITERATIONS,
            )
        )

        processed_binary = (
            ndimage.binary_closing(
                processed_binary,
                iterations=MORPHOLOGY_ITERATIONS,
            )
        )

    # --------------------------------------------------------
    # Connected components.
    # --------------------------------------------------------

    print(
        "\nFinding connected marine components..."
    )

    if CONNECTIVITY == 8:
        structure = ndimage.generate_binary_structure(
            2,
            2,
        )
    else:
        structure = ndimage.generate_binary_structure(
            2,
            1,
        )

    labels, total_components = (
        ndimage.label(
            processed_binary,
            structure=structure,
        )
    )

    labels = labels.astype(
        np.int32,
        copy=False,
    )

    print(
        "Raw connected components:",
        int(total_components),
    )

    # --------------------------------------------------------
    # Build and filter region summaries.
    # --------------------------------------------------------

    regions = calculate_regions(
        labels,
        probability,
        transform,
        crs,
    )

    retained_ids = [
        region["component_id"]
        for region in regions
    ]

    filtered_binary = np.isin(
        labels,
        retained_ids,
    )

    print(
        "Components retained:",
        len(regions),
    )

    print(
        "Filtered marine pixels:",
        int(filtered_binary.sum()),
    )

    print(
        "Filtered marine fraction:",
        float(filtered_binary.mean()),
    )

    # --------------------------------------------------------
    # Save filtered mask.
    # --------------------------------------------------------

    save_raster(
        filtered_binary.astype(np.uint8) * 255,
        reference_profile,
        FILTERED_MARINE_MASK_TIF,
        "uint8",
    )

    # --------------------------------------------------------
    # Save retained component IDs.
    # --------------------------------------------------------

    retained_labels = np.where(
        filtered_binary,
        labels,
        0,
    ).astype(np.int32)

    save_raster(
        retained_labels,
        reference_profile,
        COMPONENTS_TIF,
        "int32",
    )

    # --------------------------------------------------------
    # Print candidate table.
    # --------------------------------------------------------

    if regions:

        print(
            "\nTOP CANDIDATE REGIONS"
        )
        print(
            "-" * 75
        )

        for region in regions:

            print(
                f"#{region['rank']:02d} "
                f"ID={region['component_id']} "
                f"pixels={region['pixel_count']} "
                f"area≈{region['approx_area_km2']:.4f} km² "
                f"exact-area=pending "
                f"meanP={region['mean_probability']:.3f} "
                f"maxP={region['max_probability']:.3f} "
                f"distance={region['distance_to_wakashio_km']:.2f} km "
                f"center=({region['centroid_lon']:.5f}, "
                f"{region['centroid_lat']:.5f})"
            )

    else:

        print(
            "\nNo candidate regions survived the "
            f"{MIN_PIXELS}-pixel minimum-size filter."
        )

    # --------------------------------------------------------
    # Polygonize.
    # --------------------------------------------------------

    print(
        "\nPolygonizing retained regions..."
    )

    gdf = polygonize_regions(
        labels,
        regions,
        transform,
        crs,
    )

    if not gdf.empty:

        gdf.to_file(
            GEOJSON_PATH,
            driver="GeoJSON",
        )

        print(
            "GeoJSON:",
            GEOJSON_PATH,
        )

    else:

        print(
            "No GeoJSON polygons to write."
        )

    # --------------------------------------------------------
    # Write CSV with exact polygon area when available.
    # --------------------------------------------------------

    if regions:

        # Add exact area if a polygon exists.
        if not gdf.empty:

            exact_area_by_id = (
                gdf.groupby(
                    "component_id"
                )["area_km2"]
                .sum()
                .to_dict()
            )

            for region in regions:
                region["area_km2"] = float(
                    exact_area_by_id.get(
                        region["component_id"],
                        region["approx_area_km2"],
                    )
                )

        else:

            for region in regions:
                region["area_km2"] = region[
                    "approx_area_km2"
                ]

        fields = [
            "rank",
            "component_id",
            "pixel_count",
            "approx_area_km2",
            "area_km2",
            "mean_probability",
            "max_probability",
            "centroid_lon",
            "centroid_lat",
            "min_lon",
            "min_lat",
            "max_lon",
            "max_lat",
            "distance_to_wakashio_km",
        ]

        with STATS_CSV.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(regions)

    else:

        # Still create an empty CSV with headers.
        with STATS_CSV.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "rank",
                "component_id",
                "pixel_count",
                "approx_area_km2",
                "area_km2",
                "mean_probability",
                "max_probability",
                "centroid_lon",
                "centroid_lat",
                "min_lon",
                "min_lat",
                "max_lon",
                "max_lat",
                "distance_to_wakashio_km",
            ])

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    summary = {
        "probability_raster": str(
            PROBABILITY_TIF
        ),
        "threshold": THRESHOLD,
        "minimum_component_pixels": MIN_PIXELS,
        "connectivity": CONNECTIVITY,
        "morphology_enabled": USE_MORPHOLOGY,
        "morphology_iterations": (
            MORPHOLOGY_ITERATIONS
            if USE_MORPHOLOGY
            else 0
        ),
        "raster_width": width,
        "raster_height": height,
        "pixels_above_threshold_before_filter": int(
            marine_binary.sum()
        ),
        "fraction_above_threshold_before_filter": float(
            marine_binary.mean()
        ),
        "raw_component_count": int(
            total_components
        ),
        "retained_component_count": len(
            regions
        ),
        "filtered_pixels": int(
            filtered_binary.sum()
        ),
        "filtered_fraction": float(
            filtered_binary.mean()
        ),
        "land_fraction": float(
            land_mask.mean()
        ),
        "ocean_fraction": float(
            ocean_mask.mean()
        ),
        "wakashio_reference": {
            "longitude": WAKASHIO_LON,
            "latitude": WAKASHIO_LAT,
        },
        "outputs": {
            "land_mask": str(LAND_MASK_TIF),
            "ocean_probability": str(
                OCEAN_PROBABILITY_TIF
            ),
            "raw_marine_mask": str(
                RAW_MARINE_MASK_TIF
            ),
            "filtered_marine_mask": str(
                FILTERED_MARINE_MASK_TIF
            ),
            "components": str(
                COMPONENTS_TIF
            ),
            "geojson": str(
                GEOJSON_PATH
            ),
            "statistics_csv": str(
                STATS_CSV
            ),
        },
        "interpretation": (
            "Regions are U-Net candidate regions after "
            "land exclusion and minimum-size filtering; "
            "they are not independently confirmed oil spills."
        ),
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

    print("\n" + "=" * 75)
    print("FINAL POST-PROCESSING COMPLETE")
    print("=" * 75)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nKey files:")
    print("  Ocean probability :", OCEAN_PROBABILITY_TIF)
    print("  Filtered mask     :", FILTERED_MARINE_MASK_TIF)
    print("  Components        :", COMPONENTS_TIF)
    print("  GeoJSON            :", GEOJSON_PATH)
    print("  Statistics         :", STATS_CSV)

    print(
        "\nIMPORTANT: inspect the GeoJSON/mask in QGIS before "
        "treating any region as a confirmed spill."
    )


if __name__ == "__main__":
    main()
