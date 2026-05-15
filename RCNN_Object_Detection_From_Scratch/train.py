"""
train.py
========
Training loop for the proposed RBM-GBM object detection model.
Wraps the Keras feature extractor with a LightGBM classification head.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

from RCNN_Object_Detection_From_Scratch.model import build_proposed_model

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
INPUT_SHAPE  = (224, 224, 3)
NUM_CLASSES  = 80            # COCO classes; adjust per dataset
BATCH_SIZE   = 16
SAVED_DIR    = "Saved_models"
os.makedirs(SAVED_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point called from proposed_model/Proposed_model.py
# ─────────────────────────────────────────────────────────────────────────────
def Proposed_model(num_epochs: int = 30, split: float = 0.2, db_name: str = "COCO"):
    """
    End-to-end training pipeline:
        1. Build Keras feature extractor (CNN + Transformer + RBM layers)
        2. Extract deep features from training images
        3. Train LightGBM classification head on extracted features
        4. Save the Keras model to Saved_models/<db_name>.h5

    Parameters
    ----------
    num_epochs : int    number of epochs for Keras pre-training
    split      : float  validation split fraction
    db_name    : str    'COCO' or 'OPEN_IMG' (used for save path)
    """
    print(f"\n[Train] Building model for {db_name} …")

    # ── 1. Build Keras model ──────────────────────────────────────────────────
    model = build_proposed_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "class_output": "sparse_categorical_crossentropy",
            "bbox_output":  "mean_squared_error",
        },
        loss_weights={"class_output": 1.0, "bbox_output": 5.0},
        metrics={"class_output": "accuracy"},
    )
    model.summary()

    # ── 2. Load data ──────────────────────────────────────────────────────────
    # Replace the dummy data below with your actual dataset loader
    print("[Train] Loading training data …")
    N = 200   # placeholder sample count; replace with real data
    x = np.random.rand(N, *INPUT_SHAPE).astype(np.float32)
    y_cls  = np.random.randint(0, NUM_CLASSES, size=(N,)).astype(np.int32)
    y_bbox = np.random.rand(N, 4).astype(np.float32)

    x_tr, x_val, yc_tr, yc_val, yb_tr, yb_val = train_test_split(
        x, y_cls, y_bbox, test_size=split, random_state=42
    )

    # ── 3. Train Keras backbone ───────────────────────────────────────────────
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(SAVED_DIR, f"{db_name}.h5"),
            save_best_only=True, monitor="val_class_output_accuracy", verbose=1
        ),
    ]

    model.fit(
        x_tr,
        {"class_output": yc_tr, "bbox_output": yb_tr},
        validation_data=(x_val, {"class_output": yc_val, "bbox_output": yb_val}),
        epochs=num_epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"[Train] Keras model saved → {SAVED_DIR}/{db_name}.h5")

    # ── 4. GBM head on extracted features ────────────────────────────────────
    print("[Train] Extracting features for GBM head …")
    feature_extractor = keras.Model(
        inputs=model.input,
        outputs=model.get_layer("dense").output   # intermediate dense layer
    )
    feats_tr  = feature_extractor.predict(x_tr,  batch_size=BATCH_SIZE, verbose=0)
    feats_val = feature_extractor.predict(x_val, batch_size=BATCH_SIZE, verbose=0)

    print("[Train] Training LightGBM classification head …")
    lgb_train = lgb.Dataset(feats_tr,  label=yc_tr)
    lgb_val   = lgb.Dataset(feats_val, label=yc_val, reference=lgb_train)

    params = {
        "objective":       "multiclass",
        "num_class":       NUM_CLASSES,
        "metric":          "multi_logloss",
        "learning_rate":   0.05,
        "num_leaves":      63,
        "min_data_in_leaf": 10,
        "verbose":         -1,
    }

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=200,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20)],
    )
    gbm_path = os.path.join(SAVED_DIR, f"{db_name}_gbm.txt")
    gbm.save_model(gbm_path)
    print(f"[Train] GBM model saved → {gbm_path}")

    return model, gbm
