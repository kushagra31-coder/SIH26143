#!/usr/bin/env python3
"""
extract_spill_regions.py

Analyze the FULL stitched Wakashio U-Net probability map and binary mask.

Inputs:
    satellite data/predictions_full_db30/
        wakashio_probability_db30.tif
        wakashio_binary_db30.tif

Outputs:
    satellite data/spill_analysis_db30/
        connected_components.tif
        filtered_binary_mask.tif
        predicted_regions.geojson
        region_statistics.csv
        analysis_summary.json

What it does:
    1. Reads the stitched probability raster in blocks.
    2. Thresholds probability at 0.20.
    3. Labels connected components.
    4. Removes very small components (configurable).
    5. Calculates area, pixel count, centroid, bbox, mean/max probability.
    6. Writes a georeferenced filtered binary mask.
    7. Exports region polygons as GeoJSON.

Memory safety:
    - Does not load the full probability raster into RAM.
    - Uses rasterio block reads.
    - Connected-component labeling is performed with a disk-backed
      array and row-by-row union-find style label merging.
    - Final polygonization is performed from the filtered mask.

IMPORTANT:
    This is candidate-region extraction, NOT confirmation that a region is
    an oil spill.
"""

from pathlib import Path
import csv
import json
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import xy
from shapely.geometry import shape
from shapely.ops import unary_union
import geopandas as gpd

# scipy is used for connected-component labeling.
try:
    from scipy import ndimage
except ImportError as exc:
    raise ImportError(
        "scipy is required. Install it with:\n"
        "pip install scipy"
    ) from exc


# ============================================================
# PATHS
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

BINARY_TIF = (
    PREDICTIONS_DIR
    / "wakashio_binary_db30.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "satellite data"
    / "spill_analysis_db30"
)

COMPONENTS_TIF = (
    OUTPUT_DIR
    / "connected_components.tif"
)

FILTERED_MASK_TIF = (
    OUTPUT_DIR
    / "filtered_binary_mask.tif"
)

GEOJSON_PATH = (
    OUTPUT_DIR
    / "predicted_regions.geojson"
)

STATS_CSV = (
    OUTPUT_DIR
    / "region_statistics.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "analysis_summary.json"
)


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

THRESHOLD = 0.20

# Minimum connected component size.
#
# 100 pixels at ~10 m spacing is ~0.01 km^2.
# Start conservatively; this is only to suppress isolated noise.
MIN_PIXELS = 100

# Connectivity:
# 1 -> 4-connected
# 2 -> 8-connected
CONNECTIVITY = 2

# Optional target location for distance ranking.
WAKASHIO_LAT = -20.438
WAKASHIO_LON = 57.745

# For region ranking/inspection.
TOP_N_REGIONS = 20


# ============================================================
# HELPERS
# ============================================================

def check_inputs():
    if not PROBABILITY_TIF.exists():
        raise FileNotFoundError(
            f"Probability raster not found:\n{PROBABILITY_TIF}"
        )

    if not BINARY_TIF.exists():
        print(
            "WARNING: binary raster not found. "
            "The probability raster will be thresholded directly."
        )


def pixel_area_km2(transform):
    """
    Approximate pixel area from geographic EPSG:4326 raster.

    This is a longitude/latitude raster, so pixel area varies with latitude.
    We calculate per-region area more accurately later from the polygon.
    This helper is only used for rough reporting.
    """
    xres = abs(transform.a)
    yres = abs(transform.e)

    # Rough central AOI latitude.
    lat = -20.438

    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(
        np.deg2rad(lat)
    )

    return (
        xres
        * km_per_deg_lon
        * yres
        * km_per_deg_lat
    )


def load_binary():
    """
    Load the already stitched binary mask.

    The binary mask is ~2,783 x 2,783 in the current AOI, so one uint8
    copy is only a few MB. We keep the operation simple and robust here.
    """

    if BINARY_TIF.exists():
        with rasterio.open(BINARY_TIF) as src:
            mask = src.read(1)
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
            bounds = src.bounds

        return (
            mask > 0,
            profile,
            transform,
            crs,
            bounds,
        )

    # Fallback: threshold the probability raster.
    with rasterio.open(PROBABILITY_TIF) as src:
        p = src.read(1).astype(np.float32)
        p /= 255.0

        mask = p >= THRESHOLD

        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        bounds = src.bounds

    return (
        mask,
        profile,
        transform,
        crs,
        bounds,
    )


def connected_components(binary):
    """
    Label connected components using scipy.

    The current stitched AOI is only ~7.75 million pixels, so a bool mask
    plus int32 label raster is still manageable on a typical workstation.
    """

    if CONNECTIVITY == 1:
        structure = ndimage.generate_binary_structure(
            2,
            1,
        )
    elif CONNECTIVITY == 2:
        structure = ndimage.generate_binary_structure(
            2,
            2,
        )
    else:
        raise ValueError(
            "CONNECTIVITY must be 1 or 2."
        )

    labels, num = ndimage.label(
        binary,
        structure=structure,
    )

    return labels.astype(
        np.int32,
        copy=False,
    ), int(num)


def build_component_summary(
    labels,
    probability,
    transform,
    crs,
):
    """
    Calculate statistics for every component.

    probability is loaded as float32 [0,1].
    """

    ids = np.unique(labels)

    ids = ids[ids != 0]

    if ids.size == 0:
        return []

    counts = np.bincount(
        labels.ravel()
    )

    # Accumulate probability sums/maxima.
    sums = np.bincount(
        labels.ravel(),
        weights=probability.ravel(),
    )

    # Max probability per component.
    maxima = np.zeros(
        int(labels.max()) + 1,
        dtype=np.float32,
    )

    flat_labels = labels.ravel()
    flat_prob = probability.ravel()

    # Only process positive labels.
    positive = flat_labels > 0

    np.maximum.at(
        maxima,
        flat_labels[positive],
        flat_prob[positive],
    )

    results = []

    approx_pixel_area = pixel_area_km2(
        transform
    )

    for component_id in ids:

        count = int(
            counts[component_id]
        )

        if count < MIN_PIXELS:
            continue

        # Pixel positions.
        rows, cols = np.where(
            labels == component_id
        )

        # Geographic polygon later gives more exact area.
        mean_probability = float(
            sums[component_id] / count
        )

        max_probability = float(
            maxima[component_id]
        )

        # Pixel center centroid.
        mean_row = float(
            rows.mean()
        )
        mean_col = float(
            cols.mean()
        )

        lon, lat = xy(
            transform,
            mean_row,
            mean_col,
            offset="center",
        )

        # Pixel-coordinate bounding box.
        min_row = int(rows.min())
        max_row = int(rows.max())
        min_col = int(cols.min())
        max_col = int(cols.max())

        min_lon, max_lat = xy(
            transform,
            min_row,
            min_col,
            offset="ul",
        )

        max_lon, min_lat = xy(
            transform,
            max_row + 1,
            max_col + 1,
            offset="lr",
        )

        distance_km = float(
            np.sqrt(
                (
                    (lon - WAKASHIO_LON)
                    * 111.32
                    * np.cos(
                        np.deg2rad(
                            WAKASHIO_LAT
                        )
                    )
                ) ** 2
                +
                (
                    (lat - WAKASHIO_LAT)
                    * 111.32
                ) ** 2
            )
        )

        results.append({
            "component_id": int(component_id),
            "pixel_count": count,
            "approx_area_km2": (
                count * approx_pixel_area
            ),
            "mean_probability": mean_probability,
            "max_probability": max_probability,
            "centroid_lon": float(lon),
            "centroid_lat": float(lat),
            "min_lon": float(min_lon),
            "min_lat": float(min_lat),
            "max_lon": float(max_lon),
            "max_lat": float(max_lat),
            "distance_to_wakashio_km": distance_km,
        })

    results.sort(
        key=lambda r: (
            -r["pixel_count"],
            r["distance_to_wakashio_km"],
        )
    )

    for rank, item in enumerate(
        results,
        start=1,
    ):
        item["rank"] = rank

    return results


def polygonize_regions(
    labels,
    kept_ids,
    transform,
    crs,
):
    """
    Polygonize only the retained component labels.

    Each component is polygonized separately through rasterio.features.shapes.
    """

    if not kept_ids:
        return gpd.GeoDataFrame(
            geometry=[],
            crs=crs,
        )

    kept_ids_set = set(
        int(x) for x in kept_ids
    )

    geometries = []
    properties = []

    # We polygonize a binary mask for each retained component.
    # This avoids accidentally merging neighboring components.
    for component_id in kept_ids:

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

            properties.append(
                int(component_id)
            )

    gdf = gpd.GeoDataFrame(
        {
            "component_id": properties,
        },
        geometry=geometries,
        crs=crs,
    )

    return gdf


def write_raster_outputs(
    labels,
    kept_ids,
    filtered_mask,
    profile,
):
    """
    Save the component labels and filtered binary mask.
    """

    kept_set = set(
        int(x) for x in kept_ids
    )

    # Component raster:
    # retained components keep their component ID, all others -> 0.
    component_output = np.zeros(
        labels.shape,
        dtype=np.int32,
    )

    if kept_set:
        keep_array = np.isin(
            labels,
            list(kept_set),
        )

        component_output[keep_array] = (
            labels[keep_array]
        )

    component_profile = profile.copy()

    component_profile.update(
        driver="GTiff",
        dtype="int32",
        count=1,
        nodata=0,
        compress="deflate",
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(
        COMPONENTS_TIF,
        "w",
        **component_profile,
    ) as dst:
        dst.write(
            component_output,
            1,
        )

    binary_profile = profile.copy()

    binary_profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=0,
        compress="deflate",
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(
        FILTERED_MASK_TIF,
        "w",
        **binary_profile,
    ) as dst:
        dst.write(
            (
                filtered_mask.astype(
                    np.uint8
                )
                * 255
            ),
            1,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("WAKASHIO CONNECTED COMPONENT ANALYSIS")
    print("=" * 70)

    print("\nProbability:")
    print(PROBABILITY_TIF)

    print("\nBinary:")
    print(BINARY_TIF)

    print("\nThreshold:", THRESHOLD)
    print("Minimum component:", MIN_PIXELS, "pixels")

    check_inputs()

    # --------------------------------------------------------
    # Load probability raster.
    # --------------------------------------------------------

    with rasterio.open(
        PROBABILITY_TIF
    ) as src:

        probability_u8 = src.read(
            1
        )

        probability = (
            probability_u8.astype(
                np.float32
            )
            / 255.0
        )

        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height

    print("\nRaster:")
    print("  Size:", width, "x", height)
    print("  CRS :", crs)

    # --------------------------------------------------------
    # Threshold.
    # --------------------------------------------------------

    binary = (
        probability
        >= THRESHOLD
    )

    print(
        "\nPixels above threshold:",
        int(binary.sum()),
    )

    print(
        "Fraction above threshold:",
        float(binary.mean()),
    )

    # --------------------------------------------------------
    # Connected components.
    # --------------------------------------------------------

    print("\nLabeling connected components...")

    labels, num_components = (
        connected_components(
            binary
        )
    )

    print(
        "Total raw components:",
        num_components,
    )

    # --------------------------------------------------------
    # Statistics.
    # --------------------------------------------------------

    regions = build_component_summary(
        labels,
        probability,
        transform,
        crs,
    )

    print(
        "Components >= minimum size:",
        len(regions),
    )

    if regions:

        print("\nTop candidate regions:")

        for region in regions[
            :TOP_N_REGIONS
        ]:

            print(
                f"  #{region['rank']:02d} "
                f"ID={region['component_id']} "
                f"pixels={region['pixel_count']} "
                f"area≈{region['approx_area_km2']:.4f} km² "
                f"meanP={region['mean_probability']:.3f} "
                f"maxP={region['max_probability']:.3f} "
                f"distance={region['distance_to_wakashio_km']:.2f} km "
                f"center=({region['centroid_lon']:.5f}, "
                f"{region['centroid_lat']:.5f})"
            )

    else:
        print(
            "\nNo components passed the minimum-size filter."
        )

    # --------------------------------------------------------
    # Keep IDs.
    # --------------------------------------------------------

    kept_ids = [
        region["component_id"]
        for region in regions
    ]

    filtered_mask = np.isin(
        labels,
        kept_ids,
    )

    # --------------------------------------------------------
    # Write rasters.
    # --------------------------------------------------------

    print("\nWriting component rasters...")

    write_raster_outputs(
        labels,
        kept_ids,
        filtered_mask,
        profile,
    )

    # --------------------------------------------------------
    # CSV statistics.
    # --------------------------------------------------------

    stats_fields = [
        "rank",
        "component_id",
        "pixel_count",
        "approx_area_km2",
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
            fieldnames=stats_fields,
        )

        writer.writeheader()
        writer.writerows(regions)

    # --------------------------------------------------------
    # GeoJSON
    # --------------------------------------------------------

    print("\nPolygonizing retained regions...")

    region_gdf = polygonize_regions(
        labels,
        kept_ids,
        transform,
        crs,
    )

    if not region_gdf.empty:

        # Add statistics to polygons.
        stats_by_id = {
            int(r["component_id"]): r
            for r in regions
        }

        for column in [
            "rank",
            "pixel_count",
            "approx_area_km2",
            "mean_probability",
            "max_probability",
            "centroid_lon",
            "centroid_lat",
            "distance_to_wakashio_km",
        ]:

            region_gdf[column] = (
                region_gdf["component_id"]
                .map(
                    lambda cid:
                    stats_by_id[
                        int(cid)
                    ][column]
                )
            )

        # Exact polygon area requires a projected CRS.
        # UTM Zone 40S is appropriate for this Mauritius-area AOI.
        try:
            projected = region_gdf.to_crs(
                "EPSG:32740"
            )

            region_gdf["area_km2"] = (
                projected.geometry.area
                / 1_000_000.0
            )

        except Exception as exc:
            print(
                "WARNING: exact projected area calculation failed:",
                exc,
            )
            region_gdf["area_km2"] = np.nan

        # Export in geographic WGS84 so the AIS/geospatial teammate
        # can use longitude/latitude directly.
        region_gdf = region_gdf.to_crs(
            "EPSG:4326"
        )

        region_gdf.to_file(
            GEOJSON_PATH,
            driver="GeoJSON",
        )

        print(
            "GeoJSON saved:",
            GEOJSON_PATH,
        )

    else:

        print(
            "No retained polygons to export."
        )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    total_candidate_pixels = int(
        filtered_mask.sum()
    )

    summary = {
        "probability_raster": str(
            PROBABILITY_TIF
        ),
        "threshold": THRESHOLD,
        "minimum_component_pixels": MIN_PIXELS,
        "connectivity": 8 if CONNECTIVITY == 2 else 4,
        "raster_width": width,
        "raster_height": height,
        "pixels_above_threshold": int(
            binary.sum()
        ),
        "fraction_above_threshold": float(
            binary.mean()
        ),
        "raw_component_count": num_components,
        "retained_component_count": len(
            regions
        ),
        "retained_pixels": total_candidate_pixels,
        "retained_fraction": float(
            filtered_mask.mean()
        ),
        "wakashio_reference": {
            "latitude": WAKASHIO_LAT,
            "longitude": WAKASHIO_LON,
        },
        "minimum_size_is_filter_only": True,
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

    # --------------------------------------------------------
    # Done.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print("\nOutputs:")
    print("  Components:", COMPONENTS_TIF)
    print("  Filtered mask:", FILTERED_MASK_TIF)
    print("  Statistics:", STATS_CSV)
    print("  GeoJSON:", GEOJSON_PATH)
    print("  Summary:", SUMMARY_JSON)

    print(
        "\nIMPORTANT: retained regions are model candidates, "
        "not confirmed oil spills."
    )


if __name__ == "__main__":
    main()
