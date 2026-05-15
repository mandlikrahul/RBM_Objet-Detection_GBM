import os
import json
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO


# ─────────────────────────────────────────────────────────────────────────────
# Dataset paths
# ─────────────────────────────────────────────────────────────────────────────
COCO_IMG_DIR   = "Dataset/COCO/images/val2017"
COCO_ANN_FILE  = "Dataset/COCO/annotations_trainval2017/annotations/instances_val2017.json"

OID_IMG_DIR    = "Dataset/Open_image_v7/images"
OID_BBOX_FILE  = "Dataset/Open_image_v7/oidv6-train-annotations-bbox.csv"
OID_LABEL_FILE = "Dataset/Open_image_v7/annotations/oidv7-class-descriptions.csv"

IMG_SIZE = (224, 224)   # resize target for model input


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _clahe_preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement in LAB colour space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _load_and_preprocess(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = _clahe_preprocess(img)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────
def Read_data(db_name: str, max_samples: int = 500):
    """
    Load, preprocess, and cache images + labels for the chosen dataset.

    Parameters
    ----------
    db_name     : 'COCO' or 'OPEN_IMG'
    max_samples : number of samples to load (set lower for quick tests)

    Returns
    -------
    x_data : np.ndarray  shape (N, H, W, 3)
    y_data : list of annotation dicts
    """
    print(f"\n[Read_data] Loading {db_name} dataset …")

    if db_name == "COCO":
        return _read_coco(max_samples)
    elif db_name == "OPEN_IMG":
        return _read_openimages(max_samples)
    else:
        raise ValueError(f"Unknown dataset: {db_name}. Choose 'COCO' or 'OPEN_IMG'.")


# ─────────────────────────────────────────────────────────────────────────────
# COCO
# ─────────────────────────────────────────────────────────────────────────────
def _read_coco(max_samples: int):
    coco   = COCO(COCO_ANN_FILE)
    img_ids = coco.getImgIds()[:max_samples]

    x_data, y_data = [], []
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        path = os.path.join(COCO_IMG_DIR, info["file_name"])
        try:
            img = _load_and_preprocess(path)
        except FileNotFoundError:
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns    = coco.loadAnns(ann_ids)
        x_data.append(img)
        y_data.append(anns)

    print(f"[Read_data] COCO: loaded {len(x_data)} images.")
    return np.array(x_data, dtype=np.float32), y_data


# ─────────────────────────────────────────────────────────────────────────────
# Open Images V7
# ─────────────────────────────────────────────────────────────────────────────
def _read_openimages(max_samples: int):
    boxes_df = pd.read_csv(
        OID_BBOX_FILE,
        usecols=["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"]
    )
    label_df  = pd.read_csv(OID_LABEL_FILE)
    label_map = dict(zip(label_df["LabelName"], label_df["DisplayName"]))

    img_ids = boxes_df["ImageID"].unique()[:max_samples]

    x_data, y_data = [], []
    for img_id in img_ids:
        path = os.path.join(OID_IMG_DIR, img_id + ".jpg")
        try:
            img = _load_and_preprocess(path)
        except FileNotFoundError:
            continue

        rows = boxes_df[boxes_df["ImageID"] == img_id]
        anns = []
        for _, row in rows.iterrows():
            anns.append({
                "label":  label_map.get(row["LabelName"], row["LabelName"]),
                "XMin":   row["XMin"],
                "XMax":   row["XMax"],
                "YMin":   row["YMin"],
                "YMax":   row["YMax"],
            })
        x_data.append(img)
        y_data.append(anns)

    print(f"[Read_data] Open Images V7: loaded {len(x_data)} images.")
    return np.array(x_data, dtype=np.float32), y_data
