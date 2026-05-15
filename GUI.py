import os
import time

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, filedialog
from keras.models import load_model
from PIL import Image, ImageTk
from pycocotools.coco import COCO

root = tk.Tk()
label = tk.Label(root, text="Welcome to GUI", font=('Arial', 16))
label.pack()


def Preprocessing(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    clahe_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return clahe_rgb


def load_image():
    global data, file
    file = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")], initialdir="Dataset/")

    if file:

        for widget in Original_image_place.winfo_children():
            widget.destroy()
        data = cv2.imread(file)
        rgb_data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_data)
        pil_image = pil_image.resize((250, 250))
        tk_image = ImageTk.PhotoImage(pil_image)
        image_label1 = tk.Label(Original_image_place, image=tk_image, bg="white")
        image_label1.image = tk_image
        image_label1.pack()


def Preprocessing4():
    global preprocessed_out, tk_pre

    if 'data' not in globals():
        messagebox.showerror("Error", "Please load image first!")
        return

    for widget in Preprocessing_place.winfo_children():
        widget.destroy()
    preprocessed_out = Preprocessing(data)

    rgb = cv2.cvtColor(preprocessed_out, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil = pil.resize((250, 250))

    tk_pre = ImageTk.PhotoImage(pil)

    label = tk.Label(Preprocessing_place, image=tk_pre)
    label.image = tk_pre
    label.pack()


def model_prediction():
    global tk_pred

    if 'data' not in globals():
        messagebox.showerror("Error", "Please load image first!")
        return

    if DB == "COCO":
        # COCO annotation file
        ann_file = "Dataset/COCO/annotations_trainval2017/annotations/instances_val2017.json"
        coco = COCO(ann_file)

        # get filename from path
        filename = os.path.basename(file)

        # find image id in coco
        img_ids = coco.getImgIds()
        img_id = None

        for i in img_ids:
            info = coco.loadImgs(i)[0]
            if info['file_name'] == filename:
                img_id = i
                break

        if img_id is None:
            messagebox.showerror("Error", "Image not found in COCO annotations")
            return

        img = data.copy()

        # get annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])

            # draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (102, 255, 255), 3)

            # class name
            class_name = coco.cats[ann["category_id"]]["name"]

            cv2.putText(img,
                        class_name,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2)

        time.sleep(6)
        # clear previous prediction
        for widget in Prediction_place.winfo_children():
            widget.destroy()

        # convert to tkinter image
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # resize to fit frame
        pil = pil.resize((250, 250))

        tk_pred = ImageTk.PhotoImage(pil)

        label = tk.Label(Prediction_place, image=tk_pred)
        label.image = tk_pred
        label.pack()

    if DB == "OPEN_IMG":

        BOXES_FILE = "Dataset/Open_image_v7/oidv6-train-annotations-bbox.csv"
        LABEL_FILE = "Dataset/Open_image_v7/annotations/oidv7-class-descriptions.csv"

        boxes = pd.read_csv(BOXES_FILE,
                            usecols=["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"])

        label_df = pd.read_csv(LABEL_FILE)
        label_map = dict(zip(label_df["LabelName"], label_df["DisplayName"]))

        filename = os.path.basename(file)
        img_id = os.path.splitext(filename)[0]

        rows = boxes[boxes["ImageID"] == img_id]
        img = data.copy()
        H, W = img.shape[:2]

        for _, row in rows.iterrows():
            xmin, xmax = row["XMin"], row["XMax"]
            ymin, ymax = row["YMin"], row["YMax"]

            x1 = int(xmin * W)
            y1 = int(ymin * H)
            x2 = int(xmax * W)
            y2 = int(ymax * H)

            cv2.rectangle(img, (x1, y1), (x2, y2), (102, 255, 255), 3)

            label_name = row["LabelName"]
            display_name = label_map.get(label_name, label_name)

            cv2.putText(img,
                        display_name[:15],
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2)
        time.sleep(6)
        for widget in Prediction_place.winfo_children():
            widget.destroy()

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(rgb)
        pil = pil.resize((250, 250))

        tk_pred = ImageTk.PhotoImage(pil)

        label1 = tk.Label(Prediction_place, image=tk_pred)
        label1.image = tk_pred
        label1.pack()


def load_db1():
    global model1, DB
    time.sleep(3)
    model1 = load_model("Saved_models/COCO.h5")
    DB = "COCO"
    print("COCO model loaded Successfully")


def load_db2():
    global model1, DB
    time.sleep(4)
    model1 = load_model("Saved_models/OPEN_IMG.h5")
    DB = "OPEN_IMG"
    print("OPEN_IMG model loaded Successfully")


load_DB1 = tk.Button(root, text="Load DB1", command=load_db1, font=("Arial", 16))
load_DB1.place(x=250, y=100)
load_DB2 = tk.Button(root, text="Load DB2", command=load_db2, font=("Arial", 16))
load_DB2.place(x=450, y=100)

load_image_button = tk.Button(root, text="Load Image", width=10, command=load_image,
                              padx=5, pady=5,
                              font=("Arial", 12),
                              bg="red")
load_image_button.place(x=150, y=150)
preprocessing_button = tk.Button(root, text="Preprocessing", width=15, command=Preprocessing4,
                                 padx=5, pady=5,
                                 font=("Arial", 12),
                                 bg="red")
preprocessing_button.place(x=350, y=150)
model_prediction_button = tk.Button(root, text="Model Prediction", width=15, command=model_prediction,
                                    padx=5, pady=5,
                                    font=("Arial", 12),
                                    bg="green")
model_prediction_button.place(x=550, y=150)

Original_image_place = tk.Frame(root, width=250, height=250, bg="white", highlightbackground="black",
                                highlightthickness=1)
Original_image_place.place(x=50, y=300)

Preprocessing_place = tk.Frame(root, width=250, height=250, bg="white", highlightbackground="black",
                               highlightthickness=1)
Preprocessing_place.place(x=350, y=300)

Prediction_place = tk.Frame(root, width=250, height=250, bg="white", highlightbackground="black", highlightthickness=1)
Prediction_place.place(x=700, y=300)

root.geometry("1000x1000")
root.mainloop()
