import sys, os
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import traceback
import time
from pathlib import Path
from glob import glob
import shutil
from datetime import datetime as dt
import datetime

import darknet.python.darknet as dn

from src.label import Label, Shape, writeShapes, readShapes
from src.utils import image_files_from_folder, im2single, nms
from src.keras_utils import load_model

def unwarp_wpodnet_dataset_from_label(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = image_files_from_folder(str(input_dir))
    img_paths.sort()

    for _, img_path in enumerate(img_paths):
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))

        label_path = input_dir / (img_path.stem + ".txt")
        labels = readShapes(str(label_path))

        for i, label in enumerate(labels):
            w, h = img.shape[1], img.shape[0]
            pts0 = np.multiply(label.pts.T, (w, h)).astype(np.float32)
            margin = 8.
            dl, dr = 0. + margin, 288. - margin
            dt, db = 0. + margin, 96. - margin
            pts1 = np.array([ [dl, dt], [dr, dt], [dr, db], [dl, db] ]).astype(np.float32)

            #print(pts0.dtype, pts1.dtype)
            mat = cv2.getPerspectiveTransform(pts0, pts1)
            img_unwarp = cv2.warpPerspective(img, mat, (288, 96))

            output_file_name = img_path.stem + "_" + str(i) + ".jpg"
            img_unwarp_path = output_dir / output_file_name
            cv2.imwrite(str(img_unwarp_path), img_unwarp, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            print(output_file_name)
            print(pts0, pts1)

def detect_lp_bb(img_path, model_net, model_meta, 
                      classes_on_interest, threshold):
    ret, _, _ = dn.detect(
        model_net,
        model_meta,
        str(img_path).encode("ascii"),
        thresh=threshold,
    )
    if len(classes_on_interest) > 0:
        ret = [r for r in ret if r[0] in classes_on_interest]

    lp_labels = []
    if len(ret):
        ret
        for i, r in enumerate(ret):
            cx, cy, w, h = np.array(r[2][:4]).astype(int)
            lp_labels.append((cx, cy, w, h, r[0]))

    lp_labels.sort(key = lambda v : v[0])
    return lp_labels

def pre_annotate_ocr(input_dir):
    input_dir = Path(input_dir)
    # output_dir = Path(output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = image_files_from_folder(str(input_dir))
    img_paths.sort()

    # Load ocr
    ocr_weights = "data/ocr-kor/yolov4-tiny-obj_best.weights"
    ocr_netcfg = "data/ocr-kor/yolov4-tiny-obj.cfg"
    # ocr_weights = "data/ocr-kor/yolo-obj_best.weights"
    # ocr_netcfg = "data/ocr-kor/yolo-obj.cfg"
    ocr_dataset = "data/ocr-kor/obj.data"
    classes_on_interest = []
    net_threshold = 0.5

    ocr_net = dn.load_net(ocr_netcfg.encode("ascii"),
                              ocr_weights.encode("ascii"), 0)
    ocr_meta = dn.load_meta(ocr_dataset.encode("ascii"))

    for _, img_path in enumerate(img_paths):
        img_path = Path(img_path)

        img = cv2.imread(str(img_path))
        img_show = img.copy()
        img_w, img_h = np.array(img.shape[1::-1], dtype=int)

        labels = detect_lp_bb(img_path, ocr_net, ocr_meta,
                              classes_on_interest, net_threshold)

        char_list = []
        for label in labels:
            cx, cy, w, h, class_name = label
            char_list.append(class_name.decode('utf-8'))

            line_color = (255, 0, 255)
            pts = np.array([
                [ cx - w // 2, cx + w // 2, cx + w // 2, cx - w // 2 ],
                [ cy - h // 2, cy - h // 2, cy + h // 2, cy + h // 2 ],
            ])

            for i in range(4):
                cv2.line(
                    img_show,
                    (int(pts[0][i]), int(pts[1][i])), 
                    (int(pts[0][(i+1)%4]), int(pts[1][(i+1)%4])),
                    line_color, thickness=1)

        print("".join(char_list))
        
        wname = "img"
        cv2.imshow(wname, img_show)
        key = cv2.waitKey(0) & 0xEFFFFF
        cv2.destroyWindow(wname)
        if key == 27:
            break

if __name__ == "__main__":
    # unwarp_wpodnet_dataset_from_label(
    #     "_train_wpod/dataset/data_kor_v1_done",
    #     "_train_wpod/dataset/data_kor_v1_unwarp")

    #pre_annotate_ocr("_train_ocr/dataset/synth/val")
    pre_annotate_ocr("_train_wpod/dataset/data_kor_v1_unwarp")