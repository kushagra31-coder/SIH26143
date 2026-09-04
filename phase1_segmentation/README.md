# Phase 1 — Baseline SAR Oil-Spill Segmentation

> **Scope:** Dataset inspection → Preprocessing → U-Net → Training → Validation metrics  
> **Not included:** AIS, OpenDrift, vessel attribution, dashboard

---

## Project Layout

```
phase1_segmentation/
├── config.py             ← All hyperparameters and paths (edit here)
├── dataset_inspector.py  ← Step 1: Inspect data before training
├── data_pipeline.py      ← tf.data pipeline (pairs, split, normalisation)
├── model.py              ← U-Net (Keras Functional API)
├── losses_metrics.py     ← BCE+Dice loss, DiceMetric, IoUMetric
├── train.py              ← Main training script
├── evaluate.py           ← Post-training evaluation
├── setup_env.py          ← Environment & dataset readiness check
├── requirements.txt
├── checkpoints/
│   └── best_model.keras  ← Saved by ModelCheckpoint
└── outputs/
    ├── dataset_inspection.json
    ├── pipeline_meta.json
    ├── model_summary.txt
    ├── training_history.json
    ├── training_log.csv
    ├── training_curves.png
    ├── final_val_metrics.json
    └── sample_predictions.png
```

---

## Dataset Layout Required

Place the Kaggle dataset (nabilsherif/oil-spill) such that:

```
SIH ps/
└── dataset/
    ├── train/
    │   └── sentinel/
    │       ├── image/    ← Sentinel-1 SAR images
    │       └── label/    ← Binary segmentation masks
    └── test/
        └── sentinel/
            ├── image/
            └── label/
```

> **PALSAR images are ignored.** Only `sentinel/` folders are used.

---

## Quick Start

### 1. Install dependencies
```bash
cd "phase1_segmentation"
pip install -r requirements.txt
```

### 2. Check environment and dataset
```bash
python setup_env.py
```

### 3. Inspect dataset (before touching model)
```bash
python dataset_inspector.py
```

This prints:
- Number of images, masks, matched pairs
- Image dtype, shape, channels
- Pixel value range (min/max)
- Mask unique values
- Empty mask count and fraction
- Oil-pixel ratio (class imbalance)
- Normalisation guidance

### 4. Train
```bash
python train.py
```

Or with custom parameters:
```bash
python train.py --epochs 30 --batch-size 4 --base-filters 32 --lr 5e-5 --seed 42
```

Skip inspection if you already ran step 3:
```bash
python train.py --no-inspect
```

### 5. Evaluate (after training)
```bash
# Validation set (use during development)
python evaluate.py --split val

# Test set (run ONCE after all tuning is done)
python evaluate.py --split test
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **BCE + Dice combined loss** | BCE gives per-pixel calibration; Dice handles class imbalance (oil is rare) |
| **Scene-level train/val split** | Prevents spatial leakage — images from the same SAR acquisition must not appear in both splits |
| **Nearest-neighbour mask resize** | Preserves binary mask values — bilinear would introduce non-integer intermediate values |
| **Auto-detect normalisation** | Reads a sample of images first to determine correct divisor (avoids hardcoding 255) |
| **BASE_FILTERS=64** | Reasonable capacity for 256×256 SAR patches; reduces to 32 if GPU memory is limited |
| **EarlyStopping patience=10** | Avoids overfitting; stops if val_loss doesn't improve for 10 epochs |

---

## Configurable Parameters (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `DATASET_ROOT` | `../dataset` | Root of dataset relative to this folder |
| `IMG_SIZE` | `(256, 256)` | Resize target for images and masks |
| `BASE_FILTERS` | `64` | First encoder block filter count |
| `DROPOUT_RATE` | `0.2` | Dropout in encoder and bottleneck |
| `BATCH_SIZE` | `8` | Training batch size |
| `EPOCHS` | `50` | Maximum training epochs |
| `LEARNING_RATE` | `1e-4` | Initial Adam learning rate |
| `VAL_SPLIT` | `0.10` | Fraction of train set held for validation |
| `SEED` | `42` | Global random seed |

---

## Success Criteria (Phase 1)

- ✅ Training completes without NaN loss
- ✅ Validation Dice > 0.30 (model has learned something meaningful)
- ✅ `best_model.keras` saved successfully
- ✅ Training curves show decreasing loss
- ✅ Sample predictions show visible oil region detection

---

## Known Limitations

1. **Spatial leakage risk**: If filenames do not follow the Sentinel-1 naming convention, scene-level splitting is not possible and a random split is used. This is logged clearly.
2. **TIFF support**: Native GeoTIFF decoding requires `rasterio`. Without it, `tf.image.decode_image` handles PNG/JPG only.
3. **No augmentation yet**: Data augmentation (flips, rotations, speckle simulation) is reserved for Phase 2.
4. **No multi-class**: This baseline treats all non-zero mask pixels as oil. Look-alike suppression is Phase 2.
