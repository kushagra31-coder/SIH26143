#!/usr/bin/env python3
"""
Evaluate a U-Net oil-spill segmentation against a UNOSAT polygon reference.

Usage (WSL):
python evaluate_wakashio_unosat.py \
  --reference "/mnt/e/SIH project/UNOSAT/TSX_20200810_OilSpillExtent_....shp"

Outputs:
  evaluation_unosat/
    evaluation_metrics.csv
    confusion_matrix.csv
    unosat_reference_mask.tif
    evaluation_summary.json

The UNOSAT polygon is a satellite-derived reference extent, not field-validated
absolute ground truth. Metrics therefore measure agreement with that reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from shapely.ops import unary_union
from pyproj import CRS


DEFAULT_PRED = Path(
    "/mnt/e/SIH project/satellite data/predictions_full_db30/"
    "wakashio_binary_db30.tif"
)


def safe_div(a, b):
    return float(a / b) if b != 0 else 0.0


def pixel_area_km2(transform, crs):
    crs = CRS.from_user_input(crs)

    if crs.is_projected:
        # Expected for a metric projected CRS.
        return abs(transform.a * transform.e) / 1_000_000.0

    # Geographic CRS: compute geodesic area of one center pixel.
    from pyproj import Geod
    geod = Geod(ellps="WGS84")

    x0 = transform.c
    y0 = transform.f
    dx = abs(transform.a)
    dy = abs(transform.e)

    lons = [x0-dx/2, x0+dx/2, x0+dx/2, x0-dx/2]
    lats = [y0-dy/2, y0-dy/2, y0+dy/2, y0+dy/2]

    area_m2, _ = geod.polygon_area_perimeter(lons, lats)
    return abs(area_m2) / 1_000_000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction",
        type=Path,
        default=DEFAULT_PRED,
        help="U-Net binary prediction GeoTIFF (0/255 or 0/1).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="UNOSAT OilSpillExtent .shp file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output folder. Default: <prediction folder>/evaluation_unosat",
    )
    args = parser.parse_args()

    pred_path = args.prediction.resolve()
    ref_path = args.reference.resolve()
    outdir = (
        args.outdir.resolve()
        if args.outdir
        else pred_path.parent / "evaluation_unosat"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction not found: {pred_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference shapefile not found: {ref_path}")

    # ------------------------------------------------------------------
    # 1. Read prediction raster
    # ------------------------------------------------------------------
    print("[1/6] Reading prediction raster...")
    with rasterio.open(pred_path) as src:
        pred_raw = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        pred_crs = src.crs
        bounds = src.bounds
        height, width = src.height, src.width
        nodata = src.nodata

    # IMPORTANT:
    # For a binary segmentation mask, 0 is a VALID negative/background pixel.
    # Do NOT treat raster nodata=0 as invalid, otherwise all negative pixels
    # disappear from the evaluation and TN/FN become artificially zero.
    #
    # This prediction is a binary 0/255 mask, so evaluate the entire grid.
    # We only exclude non-finite values (which are not expected in this mask).
    valid = np.isfinite(pred_raw)

    finite = pred_raw[valid]
    if finite.size == 0:
        raise ValueError("Prediction raster contains no valid pixels.")

    # Supports normal 0/255 or 0/1 binary masks.
    maxv = float(np.nanmax(finite))
    if maxv > 1:
        pred_bin = pred_raw > 127.5
    else:
        pred_bin = pred_raw > 0.5
    pred_bin &= valid

    print(f"  Raster: {width} x {height}")
    print(f"  CRS: {pred_crs}")
    print(f"  Prediction nodata metadata: {nodata!r} (ignored for binary 0/255 mask)")
    print(f"  Evaluation pixels: {valid.sum():,} (entire raster grid, excluding only non-finite values)")
    print(f"  Predicted positive pixels: {pred_bin.sum():,}")

    # ------------------------------------------------------------------
    # 2. Read UNOSAT polygon
    # ------------------------------------------------------------------
    print("[2/6] Reading UNOSAT reference...")
    gdf = gpd.read_file(ref_path)

    if gdf.empty:
        raise ValueError("UNOSAT shapefile contains no features.")
    if gdf.crs is None:
        print("  WARNING: UNOSAT .prj CRS could not be read.")
        print("  Assuming UNOSAT layer is EPSG:4326 (WGS 84).")
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError("UNOSAT shapefile contains no valid geometries.")

    print(f"  Features: {len(gdf)}")
    print(f"  Reference CRS: {gdf.crs}")

    # ------------------------------------------------------------------
    # 3. Reproject and rasterize on EXACT prediction grid
    # ------------------------------------------------------------------
    print("[3/6] Reprojecting and rasterizing UNOSAT polygons...")
    if pred_crs is None:
        raise ValueError("Prediction raster has no CRS.")

    if gdf.crs != pred_crs:
        gdf = gdf.to_crs(pred_crs)

    pred_box = box(
        bounds.left, bounds.bottom, bounds.right, bounds.top
    )
    gdf = gdf[gdf.intersects(pred_box)].copy()

    if gdf.empty:
        raise ValueError(
            "No UNOSAT polygon intersects the prediction raster. "
            "Check that you selected the correct OilSpillExtent layer."
        )

    reference_geometry = unary_union(gdf.geometry)

    ref_bin = rasterize(
        [(reference_geometry, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)

    valid_eval = valid
    p = pred_bin[valid_eval]
    r = ref_bin[valid_eval]

    # ------------------------------------------------------------------
    # 4. Confusion matrix and segmentation scores
    # ------------------------------------------------------------------
    print("[4/6] Computing evaluation scores...")

    tp = int(np.logical_and(p, r).sum())
    tn = int(np.logical_and(~p, ~r).sum())
    fp = int(np.logical_and(p, ~r).sum())
    fn = int(np.logical_and(~p, r).sum())

    total = tp + tn + fp + fn

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, total)

    iou = safe_div(tp, tp + fp + fn)
    dice_f1 = safe_div(2 * tp, 2 * tp + fp + fn)

    balanced_accuracy = (recall + specificity) / 2.0
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)

    # Matthews correlation coefficient
    denom = np.sqrt(
        float(tp + fp)
        * float(tp + fn)
        * float(tn + fp)
        * float(tn + fn)
    )
    mcc = safe_div(tp * tn - fp * fn, denom)

    # Cohen's kappa
    observed = accuracy
    pred_pos = safe_div(p.sum(), total)
    ref_pos = safe_div(r.sum(), total)
    expected = pred_pos * ref_pos + (1 - pred_pos) * (1 - ref_pos)
    kappa = safe_div(observed - expected, 1 - expected)

    # ------------------------------------------------------------------
    # 5. Area statistics
    # ------------------------------------------------------------------
    print("[5/6] Computing area statistics...")
    px_area = pixel_area_km2(transform, pred_crs)

    pred_area = pred_bin[valid].sum() * px_area
    ref_area = ref_bin[valid].sum() * px_area
    intersection_area = tp * px_area
    union_area = (tp + fp + fn) * px_area

    # ------------------------------------------------------------------
    # 6. Save everything
    # ------------------------------------------------------------------
    metrics = {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "IoU_Jaccard": iou,
        "Dice_F1": dice_f1,
        "Precision": precision,
        "Recall_Sensitivity": recall,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_accuracy,
        "False_Positive_Rate": fpr,
        "False_Negative_Rate": fnr,
        "MCC": mcc,
        "Cohen_Kappa": kappa,
        "Predicted_Positive_Fraction": safe_div(p.sum(), total),
        "Reference_Positive_Fraction": safe_div(r.sum(), total),
        "Pixel_Area_km2": px_area,
        "Predicted_Area_km2": float(pred_area),
        "Reference_Area_km2": float(ref_area),
        "Intersection_Area_km2": float(intersection_area),
        "Union_Area_km2": float(union_area),
        "Predicted_to_Reference_Area_Ratio": safe_div(pred_area, ref_area),
    }

    # Reference mask on exact prediction grid.
    ref_profile = profile.copy()
    ref_profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=0,
        compress="deflate",
    )
    reference_mask_path = outdir / "unosat_reference_mask.tif"
    with rasterio.open(reference_mask_path, "w", **ref_profile) as dst:
        dst.write(ref_bin.astype("uint8"), 1)

    pd.DataFrame(
        [{"metric": k, "value": v} for k, v in metrics.items()]
    ).to_csv(outdir / "evaluation_metrics.csv", index=False)

    pd.DataFrame(
        [
            ["Reference Positive", tp, fn],
            ["Reference Negative", fp, tn],
        ],
        columns=["", "Prediction Positive", "Prediction Negative"],
    ).to_csv(outdir / "confusion_matrix.csv", index=False)

    summary = {
        "prediction": str(pred_path),
        "reference": str(ref_path),
        "prediction_crs": str(pred_crs),
        "reference_crs_used": str(gdf.crs),
        "metrics": metrics,
        "note": (
            "UNOSAT is a satellite-derived reference oil-spill extent, "
            "not field-validated absolute ground truth. Scores measure "
            "agreement with the UNOSAT reference."
        ),
    }
    with (outdir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n========== EVALUATION RESULTS ==========")
    print(f"TP:                 {tp:,}")
    print(f"TN:                 {tn:,}")
    print(f"FP:                 {fp:,}")
    print(f"FN:                 {fn:,}")
    print()
    print(f"IoU / Jaccard:      {iou:.4f}")
    print(f"Dice / F1:          {dice_f1:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"Specificity:        {specificity:.4f}")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")
    print(f"False Positive Rate:{fpr:.4f}")
    print(f"False Negative Rate:{fnr:.4f}")
    print(f"MCC:                {mcc:.4f}")
    print(f"Cohen's Kappa:      {kappa:.4f}")
    print()
    print(f"Predicted area:     {pred_area:.4f} km²")
    print(f"Reference area:     {ref_area:.4f} km²")
    print(f"Intersection area:  {intersection_area:.4f} km²")
    print(f"Union area:         {union_area:.4f} km²")
    print(f"Area ratio:         {safe_div(pred_area, ref_area):.4f}")
    print()
    print(f"Saved results to:\n  {outdir}")


if __name__ == "__main__":
    main()
