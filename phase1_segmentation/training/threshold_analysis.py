"""
threshold_analysis.py
---------------------
Evaluate the SAME trained segmentation model at multiple probability
thresholds on the validation set.

This does NOT retrain the model.

Run from:
    SIH26143/phase1_segmentation/

Example:
    python threshold_analysis.py

Outputs:
    outputs/threshold_analysis/threshold_metrics.csv
    outputs/threshold_analysis/threshold_metrics.png
    outputs/threshold_analysis/best_threshold.json
"""

import csv
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from data_pipeline_updated import build_datasets
from losses_metrics import bce_dice_loss, DiceMetric, IoUMetric


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]

OUTPUT_DIR = Path(config.OUTPUTS_DIR) / "threshold_analysis"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("threshold_analysis")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
):
    """
    Calculate mean Dice and IoU across validation images.

    Also calculates metrics separately for:
      - all images
      - images with non-empty ground-truth masks
    """
    y_true = y_true[..., 0].astype(np.float32)
    y_prob = y_prob[..., 0].astype(np.float32)

    y_pred = (y_prob >= threshold).astype(np.float32)

    dice_all = []
    iou_all = []

    dice_nonempty = []
    iou_nonempty = []

    precision_all = []
    recall_all = []

    for i in range(len(y_true)):

        t = y_true[i].flatten()
        p = y_pred[i].flatten()

        tp = np.sum(t * p)
        fp = np.sum((1.0 - t) * p)
        fn = np.sum(t * (1.0 - p))

        smooth = 1e-6

        dice = (
            2.0 * tp + smooth
        ) / (
            np.sum(t) + np.sum(p) + smooth
        )

        iou = (
            tp + smooth
        ) / (
            np.sum(t) + np.sum(p) - tp + smooth
        )

        precision = (
            tp + smooth
        ) / (
            tp + fp + smooth
        )

        recall = (
            tp + smooth
        ) / (
            tp + fn + smooth
        )

        dice_all.append(float(dice))
        iou_all.append(float(iou))
        precision_all.append(float(precision))
        recall_all.append(float(recall))

        # Non-empty ground truth only.
        if np.sum(t) > 0:
            dice_nonempty.append(float(dice))
            iou_nonempty.append(float(iou))

    return {
        "threshold": threshold,

        "mean_dice": float(np.mean(dice_all)),
        "std_dice": float(np.std(dice_all)),

        "mean_iou": float(np.mean(iou_all)),
        "std_iou": float(np.std(iou_all)),

        "mean_precision": float(np.mean(precision_all)),
        "mean_recall": float(np.mean(recall_all)),

        "nonempty_mean_dice": (
            float(np.mean(dice_nonempty))
            if dice_nonempty else 0.0
        ),

        "nonempty_mean_iou": (
            float(np.mean(iou_nonempty))
            if iou_nonempty else 0.0
        ),

        "n_samples": len(dice_all),
        "n_nonempty_gt": len(dice_nonempty),
    }


# ---------------------------------------------------------------------
# Load predictions once
# ---------------------------------------------------------------------

def get_validation_predictions(model):
    """Run inference over the complete validation set once."""
    _, val_ds, _, _ = build_datasets()

    all_preds = []
    all_masks = []

    for imgs, masks in val_ds:
        preds = model(imgs, training=False)

        all_preds.append(preds.numpy())
        all_masks.append(masks.numpy())

    if not all_preds:
        raise RuntimeError("Validation dataset is empty.")

    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_masks, axis=0),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = Path(config.BEST_MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model not found:\n{model_path}"
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

    logger.info("Running inference on validation set once...")

    preds, masks = get_validation_predictions(model)

    logger.info(
        "Validation samples: %d",
        len(preds)
    )

    # -------------------------------------------------------------
    # Sweep thresholds
    # -------------------------------------------------------------

    results = []

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            masks,
            preds,
            threshold,
        )

        results.append(metrics)

        logger.info(
            "threshold=%.2f | Dice=%.4f | IoU=%.4f | "
            "nonempty Dice=%.4f | precision=%.4f | recall=%.4f",
            threshold,
            metrics["mean_dice"],
            metrics["mean_iou"],
            metrics["nonempty_mean_dice"],
            metrics["mean_precision"],
            metrics["mean_recall"],
        )

    # -------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------

    csv_path = OUTPUT_DIR / "threshold_metrics.csv"

    fieldnames = list(results[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    logger.info("Saved metrics -> %s", csv_path)

    # -------------------------------------------------------------
    # Find best thresholds
    # -------------------------------------------------------------

    best_by_dice = max(
        results,
        key=lambda x: x["mean_dice"]
    )

    best_by_iou = max(
        results,
        key=lambda x: x["mean_iou"]
    )

    best_by_nonempty_dice = max(
        results,
        key=lambda x: x["nonempty_mean_dice"]
    )

    summary = {
        "best_threshold_by_mean_dice": best_by_dice,
        "best_threshold_by_mean_iou": best_by_iou,
        "best_threshold_by_nonempty_dice": best_by_nonempty_dice,
    }

    with open(
        OUTPUT_DIR / "best_threshold.json",
        "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        thresholds = [
            r["threshold"]
            for r in results
        ]

        dice = [
            r["mean_dice"]
            for r in results
        ]

        iou = [
            r["mean_iou"]
            for r in results
        ]

        nonempty_dice = [
            r["nonempty_mean_dice"]
            for r in results
        ]

        precision = [
            r["mean_precision"]
            for r in results
        ]

        recall = [
            r["mean_recall"]
            for r in results
        ]

        plt.figure(figsize=(9, 6))

        plt.plot(
            thresholds,
            dice,
            marker="o",
            label="Mean Dice"
        )

        plt.plot(
            thresholds,
            iou,
            marker="o",
            label="Mean IoU"
        )

        plt.plot(
            thresholds,
            nonempty_dice,
            marker="o",
            label="Dice (non-empty GT only)"
        )

        plt.plot(
            thresholds,
            precision,
            marker="o",
            label="Precision"
        )

        plt.plot(
            thresholds,
            recall,
            marker="o",
            label="Recall"
        )

        plt.xlabel("Probability Threshold")
        plt.ylabel("Score")
        plt.title("Validation Threshold Analysis")
        plt.xticks(thresholds)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()

        plot_path = OUTPUT_DIR / "threshold_metrics.png"

        plt.savefig(
            plot_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        logger.info("Saved plot -> %s", plot_path)

    except ImportError:
        logger.warning(
            "matplotlib not installed; skipping plot."
        )

    # -------------------------------------------------------------
    # Print final recommendation
    # -------------------------------------------------------------

    logger.info("\n" + "=" * 60)
    logger.info("THRESHOLD ANALYSIS")
    logger.info("=" * 60)

    logger.info(
        "Best threshold by mean Dice: %.2f "
        "(Dice=%.4f)",
        best_by_dice["threshold"],
        best_by_dice["mean_dice"],
    )

    logger.info(
        "Best threshold by mean IoU: %.2f "
        "(IoU=%.4f)",
        best_by_iou["threshold"],
        best_by_iou["mean_iou"],
    )

    logger.info(
        "Best threshold by non-empty Dice: %.2f "
        "(Dice=%.4f)",
        best_by_nonempty_dice["threshold"],
        best_by_nonempty_dice["nonempty_mean_dice"],
    )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
