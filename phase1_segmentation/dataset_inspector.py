"""
dataset_inspector.py -- Phase 1 Dataset Inspection
==================================================
Run this FIRST (before training) to understand the data.

Outputs a console report covering:
  - number of images / masks
  - matched / unmatched file pairs
  - dtype, shape, channels
  - pixel value range (min/max)
  - mask unique values
  - number/fraction of empty masks
  - oil-pixel ratio (positive class density)

Nothing is written to disk; this is a read-only diagnostic.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np

# -- Optional imports (handled gracefully if unavailable) ----------------------
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Add parent dir so config.py is importable whether run directly or from IDE
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ----------------------------------------------------------------------------─
# Helpers
# ----------------------------------------------------------------------------─

def _available_reader() -> str:
    if HAS_RASTERIO:
        return "rasterio"
    if HAS_CV2:
        return "opencv"
    if HAS_PIL:
        return "pillow"
    return "none"


def _load_image_as_array(path: Path, reader: str) -> np.ndarray:
    """
    Load any image (PNG, TIFF, JPG) as a numpy float32 array.
    Returns shape (H, W) for single-channel or (H, W, C) for multi-channel.
    """
    path = str(path)
    if reader == "rasterio":
        with rasterio.open(path) as src:
            data = src.read()           # (C, H, W)
            if data.shape[0] == 1:
                return data[0].astype(np.float32)
            return np.transpose(data, (1, 2, 0)).astype(np.float32)
    elif reader == "opencv":
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"cv2 could not read: {path}")
        return img.astype(np.float32)
    elif reader == "pillow":
        img = PILImage.open(path)
        return np.array(img, dtype=np.float32)
    else:
        raise RuntimeError(
            "No image reader available. Install at least one of: "
            "rasterio, opencv-python, Pillow"
        )


def _collect_files(directory: Path, extensions: set) -> dict:
    """
    Walk a directory and return {stem: Path} for all matching extensions.
    Stem is used as the pair-matching key.
    """
    if not directory.exists():
        return {}
    files = {}
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() in extensions:
            files[f.stem] = f
    return files


# ----------------------------------------------------------------------------─
# Core inspection
# ----------------------------------------------------------------------------─

def inspect_split(
    img_dir: Path,
    mask_dir: Path,
    split_name: str,
    reader: str,
    max_sample: int = 200,
    verbose: bool = True,
) -> dict:
    """
    Inspect one data split (train or test).

    Parameters
    ----------
    img_dir    : directory containing satellite images
    mask_dir   : directory containing segmentation masks
    split_name : label for console output (e.g. "TRAIN")
    reader     : one of "rasterio", "opencv", "pillow"
    max_sample : inspect at most this many pairs (prevents long runtime)
    verbose    : print results to console

    Returns
    -------
    dict with all computed statistics.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  SPLIT: {split_name}")
    print(sep)

    # -- File discovery --------------------------------------------------------
    img_files  = _collect_files(img_dir,  config.IMAGE_EXTENSIONS)
    mask_files = _collect_files(mask_dir, config.IMAGE_EXTENSIONS)

    n_images = len(img_files)
    n_masks  = len(mask_files)

    matched   = sorted(set(img_files.keys()) & set(mask_files.keys()))
    img_only  = sorted(set(img_files.keys()) - set(mask_files.keys()))
    mask_only = sorted(set(mask_files.keys()) - set(img_files.keys()))

    print(f"\n  Image directory : {img_dir}")
    print(f"  Mask  directory : {mask_dir}")
    print(f"\n  Total images    : {n_images}")
    print(f"  Total masks     : {n_masks}")
    print(f"  Matched pairs   : {len(matched)}")
    print(f"  Images w/o mask : {len(img_only)}")
    print(f"  Masks  w/o image: {len(mask_only)}")

    if img_only:
        print(f"\n  [WARN] Images without a matching mask ({len(img_only)} total):")
        for s in img_only[:10]:
            print(f"         {s}")
        if len(img_only) > 10:
            print(f"         … and {len(img_only) - 10} more")

    if mask_only:
        print(f"\n  [WARN] Masks without a matching image ({len(mask_only)} total):")
        for s in mask_only[:10]:
            print(f"         {s}")
        if len(mask_only) > 10:
            print(f"         … and {len(mask_only) - 10} more")

    if not matched:
        print("\n  [ERROR] No matched pairs found. Check directory paths.")
        return {
            "split": split_name,
            "n_images": n_images,
            "n_masks": n_masks,
            "n_matched": 0,
        }

    # -- Sample a subset for pixel-level stats --------------------------------─
    sample_stems = matched[:max_sample]
    print(f"\n  Sampling {len(sample_stems)} pairs for pixel statistics …")

    img_shapes   = []
    img_dtypes   = set()
    img_mins     = []
    img_maxs     = []
    mask_dtypes  = set()
    mask_uniques = set()
    empty_mask_count = 0
    oil_pixel_count  = 0
    total_pixel_count = 0
    load_errors  = []

    for stem in sample_stems:
        try:
            img  = _load_image_as_array(img_files[stem],  reader)
            mask = _load_image_as_array(mask_files[stem], reader)

            img_shapes.append(img.shape)
            img_dtypes.add(str(img.dtype))
            img_mins.append(float(img.min()))
            img_maxs.append(float(img.max()))

            mask_dtypes.add(str(mask.dtype))
            uniq = np.unique(mask)
            mask_uniques.update(uniq.tolist())

            # Empty mask = all pixels are background (0)
            if np.all(mask == 0):
                empty_mask_count += 1

            # Count oil pixels -- assume oil label ≠ 0
            n_pixels = mask.size
            total_pixel_count += n_pixels
            oil_pixels = int(np.sum(mask > 0))
            oil_pixel_count += oil_pixels

        except Exception as e:
            load_errors.append((stem, str(e)))

    # -- Image statistics ------------------------------------------------------
    unique_shapes = list({s for s in img_shapes})
    print(f"\n  -- Image statistics (over {len(sample_stems)} samples) --")
    print(f"  dtypes seen         : {img_dtypes}")
    print(f"  unique shapes       : {unique_shapes[:5]}")
    if len(unique_shapes) > 5:
        print(f"                        … and {len(unique_shapes)-5} more")

    global_min = min(img_mins) if img_mins else float("nan")
    global_max = max(img_maxs) if img_maxs else float("nan")
    print(f"  pixel value range   : [{global_min:.4f}, {global_max:.4f}]")

    # Infer channels from most common shape
    from collections import Counter
    shape_counter = Counter(img_shapes)
    most_common_shape = shape_counter.most_common(1)[0][0] if img_shapes else None
    if most_common_shape is not None:
        if len(most_common_shape) == 2:
            inferred_channels = 1
        else:
            inferred_channels = most_common_shape[-1]
    else:
        inferred_channels = "unknown"

    print(f"  inferred channels   : {inferred_channels}")

    # -- Mask statistics ------------------------------------------------------─
    print(f"\n  -- Mask statistics (over {len(sample_stems)} samples) --")
    print(f"  dtypes seen         : {mask_dtypes}")
    print(f"  unique pixel values : {sorted(mask_uniques)}")
    print(f"  empty masks         : {empty_mask_count} / {len(sample_stems)}"
          f"  ({100*empty_mask_count/max(len(sample_stems),1):.1f} %)")
    if total_pixel_count > 0:
        oil_frac = oil_pixel_count / total_pixel_count
        print(f"  oil-pixel ratio     : {oil_frac:.6f}"
              f"  ({oil_frac*100:.4f} %  of all pixels)")
        print(f"  class imbalance     : 1 : {1/oil_frac:.1f}"
              f"  (background : oil)" if oil_frac > 0 else "  (no oil pixels found in sample)")

    # -- Load errors ----------------------------------------------------------─
    if load_errors:
        print(f"\n  [WARN] {len(load_errors)} files could not be loaded:")
        for stem, err in load_errors[:5]:
            print(f"         {stem}: {err}")

    # Normalisation guidance
    print(f"\n  -- Normalisation guidance --")
    if global_max <= 1.0 and global_min >= 0.0:
        print("  Values already in [0, 1] -- no scaling needed.")
    elif global_max <= 255.0 and global_min >= 0.0:
        print("  Values in [0, 255] -- divide by 255 to normalise to [0, 1].")
    else:
        print(f"  Non-standard range [{global_min:.2f}, {global_max:.2f}].")
        print("  Clip and normalise manually. Consider SAR-specific dB scaling.")

    # Mask binarisation guidance
    print(f"\n  -- Mask binarisation guidance --")
    sv = sorted(mask_uniques)
    if set(sv) == {0.0, 1.0} or set(sv) == {0, 1}:
        print("  Masks are already binary {0, 1}.")
    elif set(sv) == {0.0, 255.0} or set(sv) == {0, 255}:
        print("  Masks are {0, 255} -- divide by 255 to binarise.")
    elif len(sv) <= 2:
        print(f"  Masks have 2 values: {sv}. Map max->1, min->0.")
    else:
        print(f"  Masks have >2 values: {sv}. Multi-class case -- check labels.")
        print("  For binary segmentation: treat all non-zero as oil (value > 0 -> 1).")

    result = {
        "split"              : split_name,
        "n_images"           : n_images,
        "n_masks"            : n_masks,
        "n_matched"          : len(matched),
        "n_unmatched_imgs"   : len(img_only),
        "n_unmatched_masks"  : len(mask_only),
        "sample_size"        : len(sample_stems),
        "img_dtypes"         : list(img_dtypes),
        "img_shapes_sample"  : unique_shapes[:5],
        "inferred_channels"  : inferred_channels,
        "img_global_min"     : global_min,
        "img_global_max"     : global_max,
        "mask_dtypes"        : list(mask_dtypes),
        "mask_unique_values" : sorted(mask_uniques),
        "empty_mask_count"   : empty_mask_count,
        "empty_mask_fraction": empty_mask_count / max(len(sample_stems), 1),
        "oil_pixel_ratio"    : oil_pixel_count / max(total_pixel_count, 1),
        "load_errors"        : len(load_errors),
    }

    return result


def run_inspection(save_json: bool = True) -> dict:
    """
    Inspect both TRAIN and TEST splits and optionally save a JSON summary.
    """
    reader = _available_reader()
    print(f"\n  Image reader: {reader}")
    if reader == "none":
        print(
            "[FATAL] No image reader found.\n"
            "Install at least one of:\n"
            "  pip install rasterio\n"
            "  pip install opencv-python\n"
            "  pip install Pillow"
        )
        sys.exit(1)

    # Check directories exist
    missing_dirs = []
    for d in [config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR,
              config.TEST_IMG_DIR,  config.TEST_MASK_DIR]:
        if not Path(d).exists():
            missing_dirs.append(d)

    if missing_dirs:
        print("\n[ERROR] The following directories were NOT FOUND:")
        for d in missing_dirs:
            print(f"  {d}")
        print(
            "\nEnsure the dataset is placed at:\n"
            f"  {config.DATASET_ROOT}\n"
            "Expected layout:\n"
            "  dataset/\n"
            "    train/sentinel/image/\n"
            "    train/sentinel/label/\n"
            "    test/sentinel/image/\n"
            "    test/sentinel/label/"
        )
        sys.exit(1)

    results = {}

    train_stats = inspect_split(
        img_dir    = Path(config.TRAIN_IMG_DIR),
        mask_dir   = Path(config.TRAIN_MASK_DIR),
        split_name = "TRAIN",
        reader     = reader,
    )
    results["train"] = train_stats

    test_stats = inspect_split(
        img_dir    = Path(config.TEST_IMG_DIR),
        mask_dir   = Path(config.TEST_MASK_DIR),
        split_name = "TEST",
        reader     = reader,
    )
    results["test"] = test_stats

    if save_json:
        out_path = Path(config.OUTPUTS_DIR) / "dataset_inspection.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Inspection summary saved -> {out_path}")

    return results


if __name__ == "__main__":
    run_inspection()
