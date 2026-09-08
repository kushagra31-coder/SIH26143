"""
failure_analysis.py
-------------------
Analyze per-image validation failures for the oil-spill segmentation model.

Run from:
    SIH26143/phase1_segmentation/

Example:
    python failure_analysis.py

Outputs:
    outputs/failure_analysis/per_image_metrics.csv
    outputs/failure_analysis/worst_20/
    outputs/failure_analysis/best_10/
    outputs/failure_analysis/median_10/
    outputs/failure_analysis/summary.json

The analysis uses the existing validation split:
    oil_spill_dataset/val/images/
    oil_spill_dataset/val/masks/
"""

import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from data_pipeline_updated import (
    build_datasets,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
)

from losses_metrics import bce_dice_loss, DiceMetric, IoUMetric


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("failure_analysis")


THRESHOLD = 0.2
OUTPUT_DIR = Path(config.OUTPUTS_DIR) / "failure_analysis"


def dice_iou_precision_recall(
    truth: np.ndarray,
    pred_prob: np.ndarray,
    threshold: float = 0.5,
):
    """Return Dice, IoU, precision, recall, and area statistics."""
    truth = truth.astype(np.float32).flatten()
    pred = (pred_prob >= threshold).astype(np.float32).flatten()

    tp = np.sum(truth * pred)
    fp = np.sum((1.0 - truth) * pred)
    fn = np.sum(truth * (1.0 - pred))

    eps = 1e-6

    dice = (2.0 * tp + eps) / (
        np.sum(truth) + np.sum(pred) + eps
    )

    iou = (tp + eps) / (
        np.sum(truth) + np.sum(pred) - tp + eps
    )

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    gt_area = np.mean(truth)
    pred_area = np.mean(pred)

    if gt_area > eps:
        area_ratio = pred_area / gt_area
    else:
        area_ratio = np.nan

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "gt_area_fraction": float(gt_area),
        "pred_area_fraction": float(pred_area),
        "pred_to_gt_area_ratio": (
            float(area_ratio) if np.isfinite(area_ratio) else None
        ),
    }


def classify_failure(row):
    """
    Simple diagnostic classification.

    This is not a scientific ground-truth taxonomy; it is a practical
    way to prioritize inspection.
    """
    dice = row["dice"]
    precision = row["precision"]
    recall = row["recall"]
    ratio = row["pred_to_gt_area_ratio"]

    if dice >= 0.8:
        return "good"

    # Model predicts too much unrelated area.
    if (
        precision < 0.5
        and ratio is not None
        and ratio > 1.5
    ):
        return "over-segmentation"

    # Model misses a substantial part of the actual spill.
    if (
        recall < 0.5
        and ratio is not None
        and ratio < 0.5
    ):
        return "under-segmentation"

    if precision < 0.5 and recall < 0.5:
        return "mixed_failure"

    if recall < 0.5:
        return "mostly_missed"

    if precision < 0.5:
        return "mostly_false_positive"

    return "boundary_or_shape_error"


def collect_validation_predictions(model):
    """Run the model over the complete validation set."""
    _, val_ds, _, _ = build_datasets()

    all_preds = []
    all_masks = []

    for images, masks in val_ds:
        preds = model(images, training=False)

        all_preds.append(preds.numpy())
        all_masks.append(masks.numpy())

    if not all_preds:
        raise RuntimeError("Validation dataset produced no batches.")

    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_masks, axis=0),
    )


def get_validation_image_paths():
    """Return validation image paths in the same stem-sorted order."""
    image_paths = [
        p for p in Path(VAL_IMG_DIR).iterdir()
        if p.is_file()
        and p.suffix.lower() in config.IMAGE_EXTENSIONS
    ]

    # Keep only images with a corresponding mask.
    matched = []
    for p in image_paths:
        has_mask = any(
            (Path(VAL_MASK_DIR) / f"{p.stem}{ext}").exists()
            for ext in config.IMAGE_EXTENSIONS
        )
        if has_mask:
            matched.append(p)

    return sorted(matched, key=lambda p: p.stem)


def save_case(
    image,
    mask,
    pred,
    row,
    output_path,
):
    """Save one 4-panel failure-analysis visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    binary = (pred >= THRESHOLD).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("SAR Image")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(pred, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Predicted Probability")

    axes[3].imshow(binary, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Predicted Binary")

    for ax in axes:
        ax.axis("off")

    title = (
        f"{row['image']} | "
        f"Dice={row['dice']:.3f} | "
        f"IoU={row['iou']:.3f} | "
        f"P={row['precision']:.3f} | "
        f"R={row['recall']:.3f} | "
        f"{row['failure_type']}"
    )

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = Path(config.BEST_MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model not found: {model_path}"
        )

    logger.info("Loading model: %s", model_path)

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "DiceMetric": DiceMetric,
            "IoUMetric": IoUMetric,
        },
        compile=False,
    )

    logger.info("Running prediction on the complete validation set...")

    preds, masks = collect_validation_predictions(model)
    image_paths = get_validation_image_paths()

    if len(image_paths) != len(preds):
        raise RuntimeError(
            "Validation image count does not match prediction count:\n"
            f"image paths = {len(image_paths)}\n"
            f"predictions = {len(preds)}"
        )

    rows = []

    for idx in range(len(preds)):
        metrics = dice_iou_precision_recall(
            masks[idx, :, :, 0],
            preds[idx, :, :, 0],
            threshold=THRESHOLD,
        )

        row = {
            "index": idx,
            "image": image_paths[idx].name,
            **metrics,
        }

        row["failure_type"] = classify_failure(row)
        rows.append(row)

    # Sort by Dice from worst to best.
    sorted_indices = sorted(
        range(len(rows)),
        key=lambda i: rows[i]["dice"],
    )

    # Store rank after sorting.
    for rank, idx in enumerate(sorted_indices, start=1):
        rows[idx]["rank"] = rank

    # Save complete metrics table.
    import csv

    csv_path = OUTPUT_DIR / "per_image_metrics.csv"

    fieldnames = [
        "rank",
        "index",
        "image",
        "dice",
        "iou",
        "precision",
        "recall",
        "gt_area_fraction",
        "pred_area_fraction",
        "pred_to_gt_area_ratio",
        "failure_type",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx in sorted_indices:
            writer.writerow({
                key: rows[idx].get(key)
                for key in fieldnames
            })

    # Print useful summary.
    dice_values = np.array([r["dice"] for r in rows])
    iou_values = np.array([r["iou"] for r in rows])

    failure_counts = {}
    for r in rows:
        failure_counts[r["failure_type"]] = (
            failure_counts.get(r["failure_type"], 0) + 1
        )

    summary = {
        "n_samples": len(rows),
        "mean_dice": float(np.mean(dice_values)),
        "median_dice": float(np.median(dice_values)),
        "std_dice": float(np.std(dice_values)),
        "mean_iou": float(np.mean(iou_values)),
        "median_iou": float(np.median(iou_values)),
        "std_iou": float(np.std(iou_values)),
        "failure_counts": failure_counts,
        "threshold": THRESHOLD,
    }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== Failure Analysis Summary ===")
    logger.info("Samples:      %d", len(rows))
    logger.info("Mean Dice:    %.4f", summary["mean_dice"])
    logger.info("Median Dice:  %.4f", summary["median_dice"])
    logger.info("Std Dice:     %.4f", summary["std_dice"])
    logger.info("Mean IoU:     %.4f", summary["mean_iou"])
    logger.info("Median IoU:   %.4f", summary["median_iou"])
    logger.info("Std IoU:      %.4f", summary["std_iou"])

    logger.info("\nFailure categories:")
    for category, count in sorted(failure_counts.items()):
        logger.info("  %-25s %d", category, count)

    logger.info("\nWorst 20:")
    for idx in sorted_indices[:20]:
        r = rows[idx]
        logger.info(
            "  #%02d %-12s Dice=%.3f IoU=%.3f P=%.3f R=%.3f %s",
            r["rank"],
            r["image"],
            r["dice"],
            r["iou"],
            r["precision"],
            r["recall"],
            r["failure_type"],
        )

    # ------------------------------------------------------------------
    # Save selected individual cases.
    # ------------------------------------------------------------------
    # We need the actual SAR images for visualization.
    import cv2

    def load_sar(path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Could not read {path}")
        image = cv2.resize(
            image,
            (masks.shape[2], masks.shape[1]),
            interpolation=cv2.INTER_LINEAR,
        )
        return image

    selections = {
        "worst_20": sorted_indices[:20],
        "best_10": sorted_indices[-10:][::-1],
        "median_10": sorted_indices[
            max(0, len(sorted_indices)//2 - 5):
            len(sorted_indices)//2 + 5
        ],
    }

    for group_name, indices in selections.items():
        group_dir = OUTPUT_DIR / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        for idx in indices:
            row = rows[idx]
            sar = load_sar(image_paths[idx])

            output_path = (
                group_dir
                / f"{row['rank']:03d}_{row['image']}_"
                  f"dice_{row['dice']:.3f}_"
                  f"iou_{row['iou']:.3f}.png"
            )

            save_case(
                image=sar,
                mask=masks[idx, :, :, 0],
                pred=preds[idx, :, :, 0],
                row=row,
                output_path=output_path,
            )

    logger.info("\nSaved analysis to: %s", OUTPUT_DIR)
    logger.info("Open:")
    logger.info("  %s", csv_path)
    logger.info("  %s/worst_20/", OUTPUT_DIR)


if __name__ == "__main__":
    main()
