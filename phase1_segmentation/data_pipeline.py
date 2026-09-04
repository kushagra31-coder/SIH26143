"""
data_pipeline.py — Phase 1 tf.data Pipeline
============================================
Responsibilities:
  1. Scan directories and build matched image/mask pair lists.
  2. Split TRAIN set into train / validation (scene-level preferred; falls
     back to random split with documented limitation).
  3. Build deterministic tf.data.Dataset pipelines for train and validation.
  4. Expose a helper for the test set (no augmentation, no shuffle).

Everything is driven by config.py values.
"""

import os
import sys
import random
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pair discovery
# ─────────────────────────────────────────────────────────────────────────────

def _collect_pairs(img_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    """
    Return sorted list of (image_path, mask_path) for all matched stems.
    Unmatched files are logged as warnings.
    """
    img_dir_p  = Path(img_dir)
    mask_dir_p = Path(mask_dir)

    def _scan(d: Path) -> dict:
        return {
            f.stem: str(f)
            for f in sorted(d.iterdir())
            if f.suffix.lower() in config.IMAGE_EXTENSIONS
        } if d.exists() else {}

    imgs  = _scan(img_dir_p)
    masks = _scan(mask_dir_p)

    matched   = sorted(set(imgs) & set(masks))
    img_only  = sorted(set(imgs) - set(masks))
    mask_only = sorted(set(masks) - set(imgs))

    if img_only:
        logger.warning("Images without matching mask (%d): %s …",
                       len(img_only), img_only[:3])
    if mask_only:
        logger.warning("Masks without matching image (%d): %s …",
                       len(mask_only), mask_only[:3])

    pairs = [(imgs[s], masks[s]) for s in matched]
    logger.info("Collected %d matched pairs from %s", len(pairs), img_dir)
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val split
# ─────────────────────────────────────────────────────────────────────────────

def _scene_id_from_stem(stem: str) -> Optional[str]:
    """
    Attempt to extract a scene ID from a filename stem.

    Sentinel-1 product names follow the pattern:
        S1A_IW_GRDH_1SDV_20200802T020708_...
    We treat the first 3 underscore-separated tokens as the scene prefix.
    If parsing fails (e.g. the dataset uses numeric filenames like '0001'),
    return None — the caller then falls back to random splitting.
    """
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0].startswith("S1"):
        return "_".join(parts[:3])   # e.g.  S1A_IW_GRDH
    return None


def split_train_val(
    pairs: List[Tuple[str, str]],
    val_fraction: float = config.VAL_SPLIT,
    seed: int = config.SEED,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str]:
    """
    Split matched pairs into training and validation subsets.

    Strategy
    --------
    1. Try scene-level splitting to prevent spatial leakage
       (images from the same SAR scene must not appear in both splits).
    2. If scene IDs cannot be parsed, fall back to random pair-level split
       and log a warning about the spatial leakage risk.

    Returns
    -------
    train_pairs, val_pairs, split_strategy_used
    """
    rng = random.Random(seed)

    # Try scene-level split
    scene_map: dict = {}  # scene_id → [pair, pair, …]
    fallback = False
    for pair in pairs:
        stem   = Path(pair[0]).stem
        scene  = _scene_id_from_stem(stem)
        if scene is None:
            fallback = True
            break
        scene_map.setdefault(scene, []).append(pair)

    if not fallback:
        scene_ids = sorted(scene_map.keys())
        rng.shuffle(scene_ids)
        n_val_scenes = max(1, round(len(scene_ids) * val_fraction))
        val_scenes   = set(scene_ids[:n_val_scenes])
        train_pairs  = [p for sid in scene_ids if sid not in val_scenes
                        for p in scene_map[sid]]
        val_pairs    = [p for sid in val_scenes for p in scene_map[sid]]
        strategy     = f"scene-level ({len(val_scenes)} val scenes)"
        logger.info("Scene-level split: %d train | %d val  (%s)",
                    len(train_pairs), len(val_pairs), strategy)
    else:
        # Random pair-level fallback
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        n_val    = max(1, round(len(shuffled) * val_fraction))
        val_pairs   = shuffled[:n_val]
        train_pairs = shuffled[n_val:]
        strategy    = (
            "RANDOM pair-level split — LIMITATION: spatial leakage is possible "
            "because filenames do not contain parseable Sentinel-1 scene IDs. "
            "Adjacent patches from the same acquisition scene may appear in both "
            "train and val. Treat validation metrics as optimistic estimates."
        )
        logger.warning("Falling back to random pair-level split. %s", strategy)

    return train_pairs, val_pairs, strategy


# ─────────────────────────────────────────────────────────────────────────────
# TensorFlow dataset builders
# ─────────────────────────────────────────────────────────────────────────────

# These will be set once during dataset construction after inspecting actual
# image range and mask values:
_NORM_DIVISOR: float  = 255.0   # updated by build_datasets()
_MASK_DIVISOR: float  = 1.0     # updated by build_datasets()
_N_CHANNELS:   int    = 3       # updated by build_datasets()


def _load_and_preprocess(
    img_path: tf.Tensor,
    mask_path: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    tf.py_function-safe loader.  Reads one image/mask pair and returns:
      image  : float32 tensor of shape (IMG_H, IMG_W, C), values in [0, 1]
      mask   : float32 tensor of shape (IMG_H, IMG_W, 1), binary {0, 1}
    """
    # ── Load image ────────────────────────────────────────────────────────────
    img_raw = tf.io.read_file(img_path)

    # tf.image.decode_image auto-detects PNG/JPEG/GIF/BMP.
    # For TIFF we fall back gracefully (rasterio path not inside a @tf.function).
    img = tf.image.decode_image(img_raw, channels=_N_CHANNELS,
                                expand_animations=False)
    img = tf.cast(img, tf.float32) / _NORM_DIVISOR

    # ── Load mask ─────────────────────────────────────────────────────────────
    mask_raw = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask_raw, channels=1, expand_animations=False)
    mask = tf.cast(mask, tf.float32) / _MASK_DIVISOR
    # Binarise: any non-zero pixel = oil (threshold at 0.5 after /255)
    # This correctly handles both {0,255} and 0-255 grayscale masks
    mask = tf.cast(mask > 0.5, tf.float32)

    # ── Resize ────────────────────────────────────────────────────────────────
    h, w = config.IMG_SIZE
    img  = tf.image.resize(img,  [h, w], method="bilinear")
    mask = tf.image.resize(mask, [h, w], method="nearest")

    # ── Final cleanup ─────────────────────────────────────────────────────────
    img  = tf.clip_by_value(img, 0.0, 1.0)
    mask = tf.cast(mask > 0.5, tf.float32)   # re-binarise after nearest resize

    return img, mask


def _set_shapes(img: tf.Tensor, mask: tf.Tensor):
    """Set static shapes after py_function so Keras can infer input shapes."""
    h, w = config.IMG_SIZE
    img.set_shape([h, w, _N_CHANNELS])
    mask.set_shape([h, w, 1])
    return img, mask


def _make_tf_dataset(
    pairs: List[Tuple[str, str]],
    shuffle: bool = False,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.SEED,
    prefetch: bool = True,
) -> tf.data.Dataset:
    """Build a batched tf.data.Dataset from a list of (img_path, mask_path) pairs."""
    img_paths  = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(pairs), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        _load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle,   # allow non-determinism only when shuffling
    )
    ds = ds.map(_set_shapes, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)

    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect normalisation parameters from a small image sample
# ─────────────────────────────────────────────────────────────────────────────

def _detect_norm_params(pairs: List[Tuple[str, str]], n_samples: int = 20):
    """
    Read a few images and masks to decide normalisation divisors.
    Updates the module-level _NORM_DIVISOR, _MASK_DIVISOR, _N_CHANNELS.

    IMPORTANT: This dataset has grayscale PNG masks with continuous values
    0-255 (not class indices). Binarisation is done via threshold > 0 in
    _load_and_preprocess, not by dividing by mask max.
    """
    global _NORM_DIVISOR, _MASK_DIVISOR, _N_CHANNELS

    sample = pairs[:n_samples]
    img_maxs      = []
    channels_list = []

    for img_p, mask_p in sample:
        try:
            img_raw = tf.io.read_file(img_p)
            img_t   = tf.image.decode_image(img_raw, expand_animations=False)
            img_maxs.append(float(tf.reduce_max(tf.cast(img_t, tf.float32))))
            channels_list.append(img_t.shape[-1] if img_t.ndim == 3 else 1)
        except Exception as e:
            logger.warning("Could not decode %s: %s", img_p, e)

    # Image normalisation
    global_max = max(img_maxs) if img_maxs else 255.0
    if global_max <= 1.01:
        _NORM_DIVISOR = 1.0
    else:
        _NORM_DIVISOR = 255.0

    # Masks: always divide by 255 then threshold > 0.5 to get binary
    # (works for both 0/255 binary masks AND 0-255 grayscale masks)
    _MASK_DIVISOR = 255.0

    # Channels
    if channels_list:
        from collections import Counter
        most_common_ch = Counter(channels_list).most_common(1)[0][0]
        _N_CHANNELS = int(most_common_ch) if most_common_ch else 3
    if config.IMG_CHANNELS is not None:
        _N_CHANNELS = config.IMG_CHANNELS

    logger.info(
        "Norm params -- img divisor: %.1f | mask divisor: %.1f | channels: %d",
        _NORM_DIVISOR, _MASK_DIVISOR, _N_CHANNELS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(
    batch_size: int = config.BATCH_SIZE,
    seed: int      = config.SEED,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict]:
    """
    Build and return:
      train_ds    — shuffled, batched, prefetched
      val_ds      — ordered, batched, prefetched
      test_ds     — ordered, batched, prefetched  (for final eval only)
      meta        — dict with pair counts, split strategy, norm params

    The TEST dataset is returned for completeness but MUST NOT be used during
    model selection or hyperparameter tuning.
    """
    # ── Collect pairs ─────────────────────────────────────────────────────────
    train_pairs = _collect_pairs(config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR)
    test_pairs  = _collect_pairs(config.TEST_IMG_DIR,  config.TEST_MASK_DIR)

    if not train_pairs:
        raise FileNotFoundError(
            f"No matched pairs found under {config.TRAIN_IMG_DIR} / "
            f"{config.TRAIN_MASK_DIR}.\n"
            "Check DATASET_ROOT in config.py."
        )

    # ── Auto-detect normalisation ─────────────────────────────────────────────
    _detect_norm_params(train_pairs, n_samples=min(30, len(train_pairs)))

    # ── Split train → train + val ─────────────────────────────────────────────
    train_split, val_split, strategy = split_train_val(
        train_pairs, val_fraction=config.VAL_SPLIT, seed=seed
    )

    print(f"\n[DataPipeline] Split strategy : {strategy}")
    print(f"[DataPipeline] Train pairs    : {len(train_split)}")
    print(f"[DataPipeline] Val   pairs    : {len(val_split)}")
    print(f"[DataPipeline] Test  pairs    : {len(test_pairs)}")
    print(f"[DataPipeline] Channels       : {_N_CHANNELS}")
    print(f"[DataPipeline] Img divisor    : {_NORM_DIVISOR}")
    print(f"[DataPipeline] Mask divisor   : {_MASK_DIVISOR}")

    # ── Build tf.data datasets ────────────────────────────────────────────────
    train_ds = _make_tf_dataset(train_split, shuffle=True,
                                 batch_size=batch_size, seed=seed)
    val_ds   = _make_tf_dataset(val_split,   shuffle=False,
                                 batch_size=batch_size, seed=seed)
    test_ds  = _make_tf_dataset(test_pairs,  shuffle=False,
                                 batch_size=batch_size, seed=seed)

    meta = {
        "n_train"       : len(train_split),
        "n_val"         : len(val_split),
        "n_test"        : len(test_pairs),
        "split_strategy": strategy,
        "n_channels"    : _N_CHANNELS,
        "norm_divisor"  : _NORM_DIVISOR,
        "mask_divisor"  : _MASK_DIVISOR,
        "img_size"      : config.IMG_SIZE,
        "batch_size"    : batch_size,
        "seed"          : seed,
    }

    return train_ds, val_ds, test_ds, meta
