"""
evaluate.py — Phase 1 Final Evaluation Helper
==============================================
Loads the saved best model and runs evaluation.

Usage
-----
  python evaluate.py                     # evaluate on val set
  python evaluate.py --split test        # evaluate on test set (after training)
  python evaluate.py --threshold 0.4     # custom binarisation threshold
  python evaluate.py --visualise 8       # save 8 sample prediction images

NOTE: TEST set evaluation should only be run ONCE after all model selection
      decisions are finalised. Do not use it for hyperparameter tuning.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from data_pipeline_updated import build_datasets
from losses_metrics  import DiceMetric, IoUMetric, bce_dice_tversky_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


def _parse_args():
    p = argparse.ArgumentParser(description="Phase 1: Evaluate saved model")
    p.add_argument("--split",     choices=["val", "test"], default="val",
                   help="Which dataset split to evaluate on.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binarisation threshold for Dice/IoU computation.")
    p.add_argument("--visualise", type=int,   default=4,
                   help="Number of sample predictions to visualise.")
    p.add_argument("--model-path", type=str, default=config.BEST_MODEL_PATH,
                   help="Path to .keras model file.")
    return p.parse_args()


def compute_metrics_numpy(y_true: np.ndarray, y_pred: np.ndarray,
                           threshold: float = 0.5) -> dict:
    """
    Compute per-image Dice and IoU, then aggregate.
    Operates on numpy arrays for flexibility.
    """
    SMOOTH = 1e-6
    y_bin = (y_pred >= threshold).astype(np.float32)

    # Flatten per image
    dice_scores = []
    iou_scores  = []

    for i in range(len(y_true)):
        t = y_true[i].flatten().astype(np.float32)
        p = y_bin[i].flatten()

        intersection = np.sum(t * p)
        dice = (2 * intersection + SMOOTH) / (np.sum(t) + np.sum(p) + SMOOTH)
        union = np.sum(t) + np.sum(p) - intersection + SMOOTH
        iou  = (intersection + SMOOTH) / union

        dice_scores.append(float(dice))
        iou_scores.append(float(iou))

    return {
        "mean_dice"   : float(np.mean(dice_scores)),
        "std_dice"    : float(np.std(dice_scores)),
        "mean_iou"    : float(np.mean(iou_scores)),
        "std_iou"     : float(np.std(iou_scores)),
        "min_dice"    : float(np.min(dice_scores)),
        "max_dice"    : float(np.max(dice_scores)),
        "n_samples"   : len(dice_scores),
        "threshold"   : threshold,
        "_dice_scores": dice_scores,
        "_iou_scores" : iou_scores,
    }


def main():
    args = _parse_args()

    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error("Model not found at %s. Run train.py first.", model_path)
        sys.exit(1)

    logger.info("Loading model from %s …", model_path)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "bce_dice_loss": bce_dice_tversky_loss,
            "DiceMetric"   : DiceMetric,
            "IoUMetric"    : IoUMetric,
        },
    )
    logger.info("Model loaded. Output shape: %s", model.output_shape)

    # Build datasets
    _, val_ds, test_ds, meta = build_datasets()
    ds = val_ds if args.split == "val" else test_ds
    split_label = args.split.upper()

    logger.info("Evaluating on %s split …", split_label)

    # Collect all predictions and ground truths
    all_preds = []
    all_masks = []

    for imgs, masks in ds:
        preds = model(imgs, training=False)
        all_preds.append(preds.numpy())
        all_masks.append(masks.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    logger.info("Total samples evaluated: %d", len(all_preds))

    # Compute metrics
    results = compute_metrics_numpy(all_masks, all_preds, threshold=args.threshold)
    results["split"] = split_label

    logger.info("=" * 50)
    logger.info("%s metrics (threshold=%.2f):", split_label, args.threshold)
    for k, v in results.items():
        if isinstance(v, float):
            logger.info("  %-20s : %.6f", k, v)
        else:
            logger.info("  %-20s : %s", k, v)
    logger.info("=" * 50)

    # Save results
    out_dir = Path(config.OUTPUTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{args.split}_metrics.json"
    results_to_save = {
        k: v for k, v in results.items()
        if not k.startswith("_")
    }

    with open(out_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    logger.info("Evaluation metrics saved → %s", out_path)

    # Visualise
    if args.visualise > 0:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Select the correct image/mask directories.
            if args.split == "val":
                image_dir = Path(build_datasets.__globals__["VAL_IMG_DIR"])
                mask_dir = Path(build_datasets.__globals__["VAL_MASK_DIR"])
            else:
                image_dir = Path(build_datasets.__globals__["TEST_IMG_DIR"])
                mask_dir = Path(build_datasets.__globals__["TEST_MASK_DIR"])

            # Build image list using the same stem-based matching approach.
            image_paths = sorted(
                [
                    p for p in image_dir.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in config.IMAGE_EXTENSIONS
                    and any(
                        (mask_dir / f"{p.stem}{ext}").exists()
                        for ext in config.IMAGE_EXTENSIONS
                    )
                ],
                key=lambda p: p.stem
            )

            n = min(args.visualise, len(all_masks), len(image_paths))

            if n == 0:
                logger.warning("No samples available for visualization.")
            else:
                vis_dir = out_dir / f"eval_{args.split}_individual"
                vis_dir.mkdir(parents=True, exist_ok=True)

                # Grab enough image batches to cover the requested number.
                imgs_list = []
                collected = 0

                for imgs_batch, _ in ds:
                    imgs_list.append(imgs_batch.numpy())
                    collected += imgs_batch.shape[0]
                    if collected >= n:
                        break

                sample_imgs = np.concatenate(imgs_list, axis=0)[:n]
                sample_masks = all_masks[:n]
                sample_preds = all_preds[:n]

                summary = []

                for i in range(n):
                    disp = sample_imgs[i, :, :, 0]
                    mask = sample_masks[i, :, :, 0]
                    pred = sample_preds[i, :, :, 0]
                    binn = (pred >= args.threshold).astype(float)

                    t = mask.flatten().astype(np.float32)
                    p = binn.flatten().astype(np.float32)

                    smooth = 1e-6
                    intersection = np.sum(t * p)

                    dice_i = (
                        2 * intersection + smooth
                    ) / (
                        np.sum(t) + np.sum(p) + smooth
                    )

                    union = (
                        np.sum(t) + np.sum(p) - intersection + smooth
                    )

                    iou_i = (intersection + smooth) / union

                    fig, axes = plt.subplots(
                        1, 4,
                        figsize=(14, 3.5)
                    )

                    titles = [
                        "SAR Image",
                        "Ground Truth",
                        "Predicted Probability",
                        "Predicted Binary",
                    ]

                    axes[0].imshow(disp, cmap="gray")
                    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
                    axes[2].imshow(pred, cmap="hot", vmin=0, vmax=1)
                    axes[3].imshow(binn, cmap="gray", vmin=0, vmax=1)

                    for ax, title in zip(axes, titles):
                        ax.set_title(title)
                        ax.axis("off")

                    fig.suptitle(
                        f"{image_paths[i].name} | "
                        f"Dice={dice_i:.4f} | IoU={iou_i:.4f}",
                        fontsize=12
                    )

                    plt.tight_layout()

                    save_path = (
                        vis_dir
                        / f"{i:03d}_{image_paths[i].stem}"
                        f"_dice_{dice_i:.3f}_iou_{iou_i:.3f}.png"
                    )

                    plt.savefig(
                        save_path,
                        dpi=150,
                        bbox_inches="tight"
                    )
                    plt.close(fig)

                    summary.append({
                        "index": i,
                        "image": image_paths[i].name,
                        "dice": float(dice_i),
                        "iou": float(iou_i),
                    })

                with open(vis_dir / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2)

                logger.info(
                    "Saved %d individual visualisations -> %s",
                    n,
                    vis_dir
                )

        except ImportError:
            logger.warning(
                "matplotlib not installed — skipping visualisations."
            )


if __name__ == "__main__":
    main()
