"""
model.py — Phase 1 U-Net (Keras Functional API)
================================================
Implements a standard U-Net with:
  - Configurable BASE_FILTERS
  - Encoder: 4 downsampling stages
  - Bottleneck
  - Decoder: 4 upsampling stages with skip connections
  - BatchNormalisation after every convolution
  - Dropout in encoder blocks and bottleneck
  - ReLU activations
  - Sigmoid output for binary segmentation

All architecture parameters are driven from config.py unless overridden.
"""

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _conv_block(
    x: tf.Tensor,
    filters: int,
    dropout_rate: float = 0.0,
    name_prefix: str    = "cb",
) -> tf.Tensor:
    """
    Two Conv2D → BN → ReLU layers, optionally followed by Dropout.

    Architecture:
        Conv2D(3×3, same) → BN → ReLU
        Conv2D(3×3, same) → BN → ReLU
        [Dropout]
    """
    x = layers.Conv2D(
        filters, (3, 3), padding="same", use_bias=False,
        name=f"{name_prefix}_conv1"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu1")(x)

    x = layers.Conv2D(
        filters, (3, 3), padding="same", use_bias=False,
        name=f"{name_prefix}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu2")(x)

    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop")(x)

    return x


def _encoder_block(
    x: tf.Tensor,
    filters: int,
    dropout_rate: float = 0.0,
    name_prefix: str    = "enc",
) -> tuple:
    """
    Returns (skip, pooled):
        skip   — feature map before pooling (used in decoder skip connections)
        pooled — downsampled by 2×2 MaxPool
    """
    skip   = _conv_block(x, filters, dropout_rate=dropout_rate, name_prefix=name_prefix)
    pooled = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool")(skip)
    return skip, pooled


def _decoder_block(
    x: tf.Tensor,
    skip: tf.Tensor,
    filters: int,
    name_prefix: str = "dec",
) -> tf.Tensor:
    """
    Upsample → Concatenate skip connection → Conv block.
    Uses bilinear upsampling + Conv2D (no transposed conv artefacts).
    """
    x = layers.UpSampling2D((2, 2), interpolation="bilinear",
                             name=f"{name_prefix}_up")(x)
    x = layers.Concatenate(name=f"{name_prefix}_cat")([x, skip])
    x = _conv_block(x, filters, dropout_rate=0.0, name_prefix=name_prefix)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────────────────────

def build_unet(
    img_height:   int   = config.IMG_SIZE[0],
    img_width:    int   = config.IMG_SIZE[1],
    n_channels:   int   = 3,
    base_filters: int   = config.BASE_FILTERS,
    dropout_rate: float = config.DROPOUT_RATE,
) -> keras.Model:
    """
    Build and return the U-Net model.

    Parameters
    ----------
    img_height   : image height (pixels) after preprocessing resize
    img_width    : image width  (pixels) after preprocessing resize
    n_channels   : number of input channels (detected from data)
    base_filters : number of filters in the first encoder block;
                   subsequent blocks double this value
    dropout_rate : dropout fraction applied after each encoder conv block

    Returns
    -------
    keras.Model with:
        input  shape: (None, img_height, img_width, n_channels)
        output shape: (None, img_height, img_width, 1)   ← sigmoid probability
    """
    inputs = keras.Input(
        shape=(img_height, img_width, n_channels), name="sar_image"
    )

    f = base_filters   # shorthand

    # ── Encoder ───────────────────────────────────────────────────────────────
    # Each encoder block halves spatial dims and doubles filters.
    s1, p1 = _encoder_block(inputs, f,     dropout_rate, name_prefix="enc1")
    s2, p2 = _encoder_block(p1,     f*2,   dropout_rate, name_prefix="enc2")
    s3, p3 = _encoder_block(p2,     f*4,   dropout_rate, name_prefix="enc3")
    s4, p4 = _encoder_block(p3,     f*8,   dropout_rate, name_prefix="enc4")

    # ── Bottleneck ────────────────────────────────────────────────────────────
    bridge = _conv_block(p4, f*16, dropout_rate=dropout_rate, name_prefix="bridge")

    # ── Decoder ───────────────────────────────────────────────────────────────
    # Each decoder block doubles spatial dims and halves filters.
    d4 = _decoder_block(bridge, s4, f*8,  name_prefix="dec4")
    d3 = _decoder_block(d4,     s3, f*4,  name_prefix="dec3")
    d2 = _decoder_block(d3,     s2, f*2,  name_prefix="dec2")
    d1 = _decoder_block(d2,     s1, f,    name_prefix="dec1")

    # ── Output ────────────────────────────────────────────────────────────────
    outputs = layers.Conv2D(
        1, (1, 1), activation="sigmoid", name="oil_mask"
    )(d1)

    model = keras.Model(inputs, outputs, name="UNet_OilSpill")
    return model


def model_summary_str(model: keras.Model) -> str:
    """Return model summary as a string (useful for saving to file)."""
    lines = []
    model.summary(print_fn=lambda s: lines.append(s))
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick sanity check: build model and print summary
    m = build_unet(n_channels=3)
    m.summary()
    print(f"\nInput  shape: {m.input_shape}")
    print(f"Output shape: {m.output_shape}")
    print(f"\nTotal parameters: {m.count_params():,}")
