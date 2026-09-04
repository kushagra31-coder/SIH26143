"""
train.py — Phase 1 Training Script
====================================
Orchestrates:
  1. Reproducibility seeding
  2. Dataset inspection (optional, via --inspect flag)
  3. tf.data pipeline construction
  4. U-Net model construction
  5. Compilation with BCE+Dice loss, Dice metric, IoU metric
  6. EarlyStopping + ModelCheckpoint callbacks
  7. Training loop
  8. History saving
  9. Final validation metric reporting
 10. Sample prediction visualisation

Usage
-----
  python train.py                    # full training run
  python train.py --inspect-only     # run dataset inspector then exit
  python train.py --epochs 10        # override epoch count
  python train.py --batch-size 4     # override batch size
  python train.py --base-filters 32  # smaller model
  python train.py --seed 123         # different random seed
  python train.py --lr 5e-5          # custom learning rate
"""

import os
import sys
import json
import math
import logging
import argparse
import random
from pathlib import Path

import numpy as np

# Suppress TF C++ / oneDNN noise on Windows BEFORE TF is imported
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

# ── Reproducibility: must happen BEFORE importing TF ──────────────────────────
def _set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # TF seed set after import below


sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# Parse args early so seed is known before TF import
def _parse_args():
    p = argparse.ArgumentParser(description="Phase 1: Train U-Net oil-spill segmentation")
    p.add_argument("--inspect-only",  action="store_true",
                   help="Run dataset inspection and exit without training.")
    p.add_argument("--epochs",        type=int,   default=config.EPOCHS)
    p.add_argument("--batch-size",    type=int,   default=config.BATCH_SIZE)
    p.add_argument("--base-filters",  type=int,   default=config.BASE_FILTERS)
    p.add_argument("--dropout",       type=float, default=config.DROPOUT_RATE)
    p.add_argument("--lr",            type=float, default=config.LEARNING_RATE)
    p.add_argument("--seed",          type=int,   default=config.SEED)
    p.add_argument("--no-inspect",    action="store_true",
                   help="Skip dataset inspection before training.")
    return p.parse_args()


args = _parse_args()
_set_seeds(args.seed)

import tensorflow as tf
tf.random.set_seed(args.seed)

from data_pipeline      import build_datasets
from dataset_inspector  import run_inspection
from model              import build_unet, model_summary_str
from losses_metrics     import bce_dice_loss, DiceMetric, IoUMetric

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_sample_predictions(
    model: "tf.keras.Model",
    val_ds: "tf.data.Dataset",
    out_dir: Path,
    n_samples: int = 4,
):
    """
    Save a figure with n_samples columns of:
        [SAR image | Ground truth | Prediction (prob) | Prediction (binary)]
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # headless — no GUI required
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping visualisation.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Grab one batch
    for imgs, masks in val_ds.take(1):
        break

    imgs  = imgs[:n_samples].numpy()
    masks = masks[:n_samples].numpy()
    preds = model.predict(imgs, verbose=0)

    fig, axes = plt.subplots(
        n_samples, 4,
        figsize=(14, 3.5 * n_samples),
        squeeze=False,
    )
    col_titles = ["SAR Image", "Ground Truth", "Prediction (prob)", "Prediction (binary)"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=12, fontweight="bold")

    for row in range(n_samples):
        img  = imgs[row]
        mask = masks[row, :, :, 0]
        pred = preds[row, :, :, 0]
        bin_ = (pred >= 0.5).astype(float)

        # Display first channel of SAR image (greyscale)
        display_img = img[:, :, 0] if img.ndim == 3 else img
        axes[row][0].imshow(display_img, cmap="gray")
        axes[row][1].imshow(mask,  cmap="gray", vmin=0, vmax=1)
        axes[row][2].imshow(pred,  cmap="hot",  vmin=0, vmax=1)
        axes[row][3].imshow(bin_,  cmap="gray", vmin=0, vmax=1)

        for col in range(4):
            axes[row][col].axis("off")

    plt.suptitle("Phase 1 — Sample Predictions (val set)", fontsize=14, y=1.01)
    plt.tight_layout()

    out_path = out_dir / "sample_predictions.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sample predictions → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def main():
    out_dir  = Path(config.OUTPUTS_DIR)
    ckpt_dir = Path(config.CHECKPOINT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Dataset inspection ─────────────────────────────────────────────────
    if not args.no_inspect or args.inspect_only:
        logger.info("Running dataset inspection …")
        inspection = run_inspection(save_json=True)
    else:
        inspection = {}

    if args.inspect_only:
        logger.info("--inspect-only flag set. Exiting after inspection.")
        return

    # ── 2. Build tf.data pipeline ─────────────────────────────────────────────
    logger.info("Building tf.data datasets …")
    train_ds, val_ds, test_ds, meta = build_datasets(
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Save pipeline metadata
    with open(out_dir / "pipeline_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    n_channels = meta["n_channels"]

    # ── 3. Build model ────────────────────────────────────────────────────────
    logger.info(
        "Building U-Net | base_filters=%d | dropout=%.2f | channels=%d",
        args.base_filters, args.dropout, n_channels,
    )
    model = build_unet(
        img_height   = config.IMG_SIZE[0],
        img_width    = config.IMG_SIZE[1],
        n_channels   = n_channels,
        base_filters = args.base_filters,
        dropout_rate = args.dropout,
    )

    # Save model summary to file
    summary_path = out_dir / "model_summary.txt"
    summary_path.write_text(model_summary_str(model), encoding="utf-8")
    logger.info("Model summary saved → %s", summary_path)

    total_params = model.count_params()
    logger.info("Total model parameters: %s", f"{total_params:,}")

    # ── 4. Compile ────────────────────────────────────────────────────────────
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer = optimizer,
        loss      = bce_dice_loss,
        metrics   = [DiceMetric(name="dice"), IoUMetric(name="iou")],
    )

    # ── 5. Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor              = "val_loss",
            patience             = 10,
            restore_best_weights = True,
            verbose              = 1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath              = config.BEST_MODEL_PATH,
            monitor               = "val_loss",
            save_best_only        = True,
            save_weights_only     = False,
            verbose               = 1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor   = "val_loss",
            factor    = 0.5,
            patience  = 5,
            min_lr    = 1e-7,
            verbose   = 1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename  = str(out_dir / "training_log.csv"),
            separator = ",",
            append    = False,
        ),
    ]

    # ── 6. Compute steps ──────────────────────────────────────────────────────
    n_train      = meta["n_train"]
    n_val        = meta["n_val"]
    steps_train  = math.ceil(n_train / args.batch_size)
    steps_val    = math.ceil(n_val   / args.batch_size)

    logger.info(
        "Training: %d pairs | %d steps/epoch | %d epochs max",
        n_train, steps_train, args.epochs,
    )
    logger.info(
        "Validation: %d pairs | %d steps/epoch", n_val, steps_val,
    )

    # ── 7. Train ──────────────────────────────────────────────────────────────
    logger.info("Starting training …")
    history = model.fit(
        train_ds,
        validation_data  = val_ds,
        epochs           = args.epochs,
        steps_per_epoch  = steps_train,
        validation_steps = steps_val,
        callbacks        = callbacks,
        verbose          = 1,
    )

    # ── 8. Save history ───────────────────────────────────────────────────────
    history_path = out_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f, indent=2,
        )
    logger.info("Training history saved → %s", history_path)

    # ── 9. Final validation metrics ───────────────────────────────────────────
    logger.info("Evaluating on validation set (best weights) …")
    val_results = model.evaluate(val_ds, verbose=1, steps=steps_val)
    metric_names = ["val_loss", "val_dice", "val_iou"]
    final_metrics = {name: float(val) for name, val in zip(metric_names, val_results)}

    logger.info("=" * 50)
    logger.info("Final validation metrics (best model):")
    for name, val in final_metrics.items():
        logger.info("  %-20s : %.6f", name, val)
    logger.info("=" * 50)

    metrics_path = out_dir / "final_val_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info("Final metrics saved → %s", metrics_path)

    # ── 10. Sample predictions ────────────────────────────────────────────────
    _save_sample_predictions(model, val_ds, out_dir, n_samples=4)

    # ── 11. Plot training curves ──────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h = history.history
        epochs_ran = range(1, len(h["loss"]) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        plot_cfg = [
            ("loss",  "Loss (BCE + Dice)",     "val_loss"),
            ("dice",  "Dice Coefficient",       "val_dice"),
            ("iou",   "IoU",                    "val_iou"),
        ]
        for ax, (train_key, ylabel, val_key) in zip(axes, plot_cfg):
            if train_key in h:
                ax.plot(epochs_ran, h[train_key], label="Train")
            if val_key in h:
                ax.plot(epochs_ran, h[val_key],   label="Val", linestyle="--")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle("Phase 1 — Training Curves", fontsize=14)
        plt.tight_layout()
        curve_path = out_dir / "training_curves.png"
        plt.savefig(curve_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Training curves saved → %s", curve_path)

    except ImportError:
        logger.warning("matplotlib not available — skipping training curve plots.")

    logger.info("Phase 1 training complete.")
    logger.info("Best model saved → %s", config.BEST_MODEL_PATH)


if __name__ == "__main__":
    main()
