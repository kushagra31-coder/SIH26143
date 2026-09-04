"""
setup_env.py - Environment Setup Checker
=========================================
Run this before train.py to verify all required packages are installed.
Prints a clear report of what is available and what needs installing.
"""

import sys
import os
from pathlib import Path

# Silence TF C++ and oneDNN noise before TF is imported anywhere
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

REQUIRED = [
    ("tensorflow",   "tensorflow>=2.12",  True),
    ("numpy",        "numpy",             True),
    ("matplotlib",   "matplotlib",        True),
    ("PIL",          "Pillow",            False),
    ("cv2",          "opencv-python",     False),
    ("rasterio",     "rasterio",          False),
    ("sklearn",      "scikit-learn",      False),
    ("tqdm",         "tqdm",              False),
]


def check():
    print("=" * 60)
    print("  Phase 1 - Environment Readiness Check")
    print("=" * 60)
    print(f"  Python : {sys.version}")
    print()

    all_required_ok = True
    install_missing = []

    for import_name, pip_name, required in REQUIRED:
        try:
            mod     = __import__(import_name)
            version = getattr(mod, "__version__", "?")
            status  = "OK "
            label   = ""
        except ImportError:
            status  = "XX " if required else "-- "
            version = "NOT INSTALLED"
            label   = ("REQUIRED - pip install " + pip_name) if required \
                      else ("optional  - pip install " + pip_name)
            if required:
                all_required_ok = False
            install_missing.append(pip_name)

        print(f"  [{status}] {import_name:<15} {version:<20}  {label}")

    print()
    if all_required_ok:
        print("  [READY] All required packages installed.")
    else:
        print("  [ACTION REQUIRED] Install missing packages:")
        print()
        print(f"    py -3.12 -m pip install {' '.join(install_missing)}")
        print()

    # GPU check
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"  GPU : {len(gpus)} device(s) found - will use GPU.")
            for g in gpus:
                print(f"        {g.name}")
        else:
            print("  GPU : None detected (CPU-only training).")
            print("        TF >= 2.11 does not support native Windows GPU.")
            print("        Options: use WSL2, or install tensorflow-directml.")
    except Exception as exc:
        print(f"  GPU check failed: {exc}")

    print()

    # Dataset directory check
    print("  Dataset directory check:")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import config
        checks = [
            ("TRAIN images", config.TRAIN_IMG_DIR),
            ("TRAIN masks",  config.TRAIN_MASK_DIR),
            ("TEST  images", config.TEST_IMG_DIR),
            ("TEST  masks",  config.TEST_MASK_DIR),
        ]
        for label, d in checks:
            p       = Path(d)
            exists  = p.exists()
            n_files = len(list(p.iterdir())) if exists else 0
            symbol  = "OK" if (exists and n_files > 0) else "XX"
            print(f"  [{symbol}] {label:<20} {n_files:>5} files   {d}")
    except Exception as exc:
        print(f"  Could not import config: {exc}")

    print("=" * 60)


if __name__ == "__main__":
    check()
