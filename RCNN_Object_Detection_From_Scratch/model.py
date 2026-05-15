"""
model.py
========
Proposed architecture:
  Semantic CNN backbone  →  Multi-Head Transformer Attention
                        →  RBM feature refinement
                        →  GBM classification + bbox regression head
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLAHE Preprocessing layer (differentiable approximation via contrast stretch)
# ─────────────────────────────────────────────────────────────────────────────
class ContrastEnhancement(layers.Layer):
    """
    Learnable contrast enhancement placed before the CNN backbone.
    Mimics CLAHE by stretching per-channel histograms.
    """
    def call(self, x):
        x_min = tf.reduce_min(x, axis=[1, 2], keepdims=True)
        x_max = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        return (x - x_min) / (x_max - x_min + 1e-7)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Semantic CNN backbone (lightweight)
# ─────────────────────────────────────────────────────────────────────────────
def semantic_cnn_backbone(input_tensor):
    """Three conv blocks with batch norm and max-pool."""
    x = layers.Conv2D(64,  3, padding="same", activation="relu")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    return x   # (B, H/8, W/8, 256)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-Head Transformer Attention block
# ─────────────────────────────────────────────────────────────────────────────
def transformer_attention_block(x, num_heads=8, key_dim=32, ff_dim=512):
    """
    One Transformer encoder block:
        LayerNorm → MultiHeadAttention (residual) →
        LayerNorm → Feed-Forward (residual)
    """
    B, H, W, C = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
    seq_len = H * W

    # Flatten spatial dims for attention
    tokens = layers.Reshape((seq_len, C))(x)

    # Self-attention with residual
    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(tokens, tokens)
    tokens = layers.LayerNormalization()(tokens + attn_out)

    # Feed-forward with residual
    ff = layers.Dense(ff_dim, activation="relu")(tokens)
    ff = layers.Dense(C)(ff)
    tokens = layers.LayerNormalization()(tokens + ff)

    # Restore spatial shape
    out = layers.Reshape((H, W, C))(tokens)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. RBM-inspired feature refinement (implemented as a trainable dense layer
#    with Gibbs-sampling-like stochastic dropout during training)
# ─────────────────────────────────────────────────────────────────────────────
def rbm_feature_refinement(x, units=512):
    """
    Approximate RBM visible→hidden→visible reconstruction as a
    bottleneck auto-encoder with sigmoid activations.
    """
    flat   = layers.GlobalAveragePooling2D()(x)           # (B, C)
    hidden = layers.Dense(units, activation="sigmoid")(flat)
    hidden = layers.Dropout(0.3)(hidden)                   # stochastic during training
    recon  = layers.Dense(flat.shape[-1], activation="sigmoid")(hidden)
    fused  = layers.Concatenate()([flat, recon])           # (B, 2C)
    return fused, hidden


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full proposed model
# ─────────────────────────────────────────────────────────────────────────────
def build_proposed_model(input_shape=(224, 224, 3), num_classes=80):
    """
    Build and return the full Keras model.

    Parameters
    ----------
    input_shape : tuple  (H, W, C)
    num_classes : int    number of object categories

    Returns
    -------
    keras.Model
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # ── Stage 1: contrast enhancement
    x = ContrastEnhancement()(inputs)

    # ── Stage 2: semantic CNN backbone
    x = semantic_cnn_backbone(x)

    # ── Stage 3: multi-head transformer attention
    x = transformer_attention_block(x, num_heads=8, key_dim=32, ff_dim=512)

    # ── Stage 4: RBM-style feature refinement
    fused, _ = rbm_feature_refinement(x, units=512)

    # ── Stage 5: classification head (GBM replaced by dense stack here;
    #             the external GBM in train.py wraps predictions from this head)
    dense = layers.Dense(256, activation="relu")(fused)
    dense = layers.Dropout(0.4)(dense)
    dense = layers.Dense(128, activation="relu")(dense)

    class_out = layers.Dense(num_classes, activation="softmax", name="class_output")(dense)
    bbox_out  = layers.Dense(4,           activation="sigmoid",  name="bbox_output")(dense)

    model = keras.Model(inputs=inputs, outputs=[class_out, bbox_out],
                        name="RBM_MultiHead_Transformer_GBM")
    return model


if __name__ == "__main__":
    model = build_proposed_model()
    model.summary()
