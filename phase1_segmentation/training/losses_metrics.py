"""
losses_metrics.py — Phase 1 Custom Losses and Metrics
======================================================
Implements:
  - Dice coefficient (metric + loss component)
  - Binary Cross-Entropy + Dice combined loss  (original baseline)
  - Tversky index/loss
  - BCE + Tversky loss
  - BCE + Dice + Tversky hybrid loss (experimental)
  - IoU (Intersection over Union) metric

All functions follow the Keras metric / loss interface conventions and
are registered as serializable so ModelCheckpoint can reload them.
"""

import tensorflow as tf
from tensorflow import keras


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_SMOOTH = 1e-6   # numerical smoothing to prevent division by zero


# ─────────────────────────────────────────────────────────────────────────────
# Dice coefficient (stateless, per-batch)
# ─────────────────────────────────────────────────────────────────────────────

def dice_coeff(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Per-batch Dice coefficient.

    Formula: 2 * |A ∩ B| / (|A| + |B| + ε)

    Parameters
    ----------
    y_true : ground-truth mask, float32, values in {0, 1}
    y_pred : predicted probability map, float32, values in [0, 1]

    Returns
    -------
    Scalar float32 tensor, Dice score ∈ (0, 1].
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + _SMOOTH) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + _SMOOTH
    )


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Dice loss = 1 − Dice coefficient."""
    return 1.0 - dice_coeff(y_true, y_pred)


def tversky_index(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    alpha: float = 0.4,
    beta: float = 0.6,
) -> tf.Tensor:
    """
    Tversky index for binary segmentation.

    alpha weights false positives and beta weights false negatives.
    beta > alpha therefore penalizes false negatives more strongly.
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    true_pos = tf.reduce_sum(y_true_f * y_pred_f)
    false_pos = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    false_neg = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))

    return (
        true_pos + _SMOOTH
    ) / (
        true_pos
        + alpha * false_pos
        + beta * false_neg
        + _SMOOTH
    )


def tversky_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    alpha: float = 0.4,
    beta: float = 0.6,
) -> tf.Tensor:
    """Tversky loss = 1 − Tversky index."""
    return 1.0 - tversky_index(
        y_true,
        y_pred,
        alpha=alpha,
        beta=beta,
    )


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Original baseline loss = Binary Cross-Entropy + Dice Loss.

    Kept unchanged for comparison with previous experiments.
    """
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    return bce + dice_loss(y_true, y_pred)


def bce_tversky_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:
    """
    BCE + Tversky loss.

    alpha=0.4, beta=0.6 gives slightly more emphasis to false negatives.
    """
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)

    tv = tversky_loss(
        y_true,
        y_pred,
        alpha=0.4,
        beta=0.6,
    )

    return bce + tv


def bce_dice_tversky_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:
    """
    Experimental hybrid loss:

        0.5 * BCE + 0.25 * Dice Loss + 0.25 * Tversky Loss

    Keeps the stable BCE/Dice behavior while adding extra pressure against
    false negatives.
    """
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)

    dl = dice_loss(y_true, y_pred)

    tl = tversky_loss(
        y_true,
        y_pred,
        alpha=0.4,
        beta=0.6,
    )

    return 0.5 * bce + 0.25 * dl + 0.25 * tl


# ─────────────────────────────────────────────────────────────────────────────
# Keras Metric wrappers
# ─────────────────────────────────────────────────────────────────────────────

class DiceMetric(keras.metrics.Metric):
    """
    Stateful Keras metric that accumulates Dice coefficient across batches.
    Tracks sum_dice and count separately for a correct epoch-level mean.
    """

    def __init__(self, threshold: float = 0.5, name: str = "dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold  = threshold
        self.sum_dice   = self.add_weight(name="sum_dice",  initializer="zeros")
        self.count      = self.add_weight(name="count",     initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(y_pred >= self.threshold, tf.float32)
        d = dice_coeff(y_true, y_pred_bin)
        self.sum_dice.assign_add(d)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.sum_dice, self.count)

    def reset_state(self):
        self.sum_dice.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        base = super().get_config()
        base["threshold"] = self.threshold
        return base


class IoUMetric(keras.metrics.Metric):
    """
    Stateful Keras metric that accumulates IoU (Jaccard index) across batches.

    IoU = |A ∩ B| / |A ∪ B|
        = |A ∩ B| / (|A| + |B| - |A ∩ B| + ε)
    """

    def __init__(self, threshold: float = 0.5, name: str = "iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold  = threshold
        self.sum_iou    = self.add_weight(name="sum_iou", initializer="zeros")
        self.count      = self.add_weight(name="count",   initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true_f   = tf.cast(tf.reshape(y_true,    [-1]), tf.float32)
        y_pred_f   = tf.cast(tf.reshape(y_pred_bin,[-1]), tf.float32)

        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union        = (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
                        - intersection + _SMOOTH)
        iou = (intersection + _SMOOTH) / union

        self.sum_iou.assign_add(iou)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.sum_iou, self.count)

    def reset_state(self):
        self.sum_iou.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        base = super().get_config()
        base["threshold"] = self.threshold
        return base
