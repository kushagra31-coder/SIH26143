"""
config.py — Phase 1 Centralised Configuration
=============================================
All hyper-parameters and paths are controlled from here.
Change values here; do not hard-code them inside model / training files.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# Root of the Kaggle / downloaded dataset.
# Adjust this if you place the dataset elsewhere.
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset")

TRAIN_IMG_DIR  = os.path.join(DATASET_ROOT, "train", "sentinel", "image")
TRAIN_MASK_DIR = os.path.join(DATASET_ROOT, "train", "sentinel", "label")
TEST_IMG_DIR   = os.path.join(DATASET_ROOT, "test",  "sentinel", "image")
TEST_MASK_DIR  = os.path.join(DATASET_ROOT, "test",  "sentinel", "label")

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
OUTPUTS_DIR    = os.path.join(os.path.dirname(__file__), "outputs")

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.keras")

# ── Image parameters ──────────────────────────────────────────────────────────
IMG_SIZE    = (256, 256)   # (height, width) after resize
IMG_CHANNELS = None        # auto-detected at runtime from actual data;
                           # override here if needed (e.g. 3 for RGB, 1 for grey)

# ── Model architecture ────────────────────────────────────────────────────────
BASE_FILTERS  = 64         # first encoder block filter count
DROPOUT_RATE  = 0.2        # dropout inside encoder / bottleneck blocks

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 8
EPOCHS        = 50
LEARNING_RATE = 1e-4
VAL_SPLIT     = 0.10       # 10 % of TRAIN set → validation

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── tf.data ───────────────────────────────────────────────────────────────────
PREFETCH_BUFFER = -1       # tf.data.AUTOTUNE at runtime

# ── Supported image extensions ────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
