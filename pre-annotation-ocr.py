import sys, os
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from glob import glob
import shutil

import darknet.python.darknet as dn

from src.label import Label, Shape, readShapes
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

def detect_char_labels(img_path, model_net, model_meta, threshold):
    ret, _, _ = dn.detect(
        model_net,
        model_meta,
        str(img_path).encode("ascii"),
        thresh=threshold,
    )

    def _find_class_idx(class_name):
        ret = -1
        for i in range(model_meta.classes):
            if model_meta.names[i] == class_name:
                ret = i
                break
        return ret

    char_labels = []
    if len(ret):
        for _, r in enumerate(ret):
            class_name = r[0]
            class_idx = _find_class_idx(class_name)
            cx, cy, w, h = np.array(r[2][:4]).astype(int)
            char_labels.append((class_idx, class_name, cx, cy, w, h))

    char_labels.sort(key = lambda v : v[0])
    return char_labels

def write_char_labels(path, char_labels):
    if len(char_labels):
        with open(path, 'w') as fp:
            for char_label in char_labels:
                class_idx, _, cx, cy, w, h = char_label
                fp.write(f'{class_idx} {cx} {cy} {w} {h}\n')

def pre_annotate_ocr(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = image_files_from_folder(str(input_dir))
    img_paths.sort()

    # Load ocr
    ocr_weights = "data/ocr-kor/yolov4-tiny-obj_best.weights"
    ocr_netcfg = "data/ocr-kor/yolov4-tiny-obj.cfg"
    # ocr_weights = "data/ocr-kor/yolo-obj_best.weights"
    # ocr_netcfg = "data/ocr-kor/yolo-obj.cfg"
    ocr_dataset = "data/ocr-kor/obj.data"
    net_threshold = 0.5

    ocr_net = dn.load_net(ocr_netcfg.encode("ascii"), ocr_weights.encode("ascii"), 0)
    ocr_meta = dn.load_meta(ocr_dataset.encode("ascii"))

    for _, img_path in enumerate(img_paths):
        img_path = Path(img_path)
        output_img_path = output_dir / img_path.name
        output_txt_path = output_dir / (img_path.stem + "_pre.txt")

        print(" ** annotating", str(img_path))

        img = cv2.imread(str(img_path))
        img_show = img.copy()
        img_w, img_h = np.array(img.shape[1::-1], dtype=int)

        char_labels_raw = detect_char_labels(img_path, ocr_net, ocr_meta, net_threshold)

        #char_list = []
        char_labels = []
        for char_label in char_labels_raw:
            class_idx, class_name, cx, cy, w, h = char_label

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

            class_name = class_name.decode('utf-8')
            cx, cy, w, h = float(cx) / img_w, float(cy) / img_h, float(w) / img_w, float(h) / img_h
            char_labels.append((class_idx, class_name, cx, cy, w, h))\

        wname = "img"
        cv2.imshow(wname, img_show)
        key = cv2.waitKey(0) & 0xEFFFFF
        cv2.destroyWindow(wname)
        if key == 27:
            break
        elif key == 32:
            shutil.copy(str(img_path), str(output_img_path))
            write_char_labels(str(output_txt_path), char_labels)
        else:
            print("    -> ignored")

if __name__ == "__main__":
    # unwarp_wpodnet_dataset_from_label(
    #     "_train_wpod/dataset/data_kor_v1_done",
    #     "_train_wpod/dataset/data_kor_v1_unwarp")

    #pre_annotate_ocr("_train_ocr/dataset/synth/val")
    pre_annotate_ocr("_train_wpod/dataset/data_kor_v1_unwarp", "_train_ocr/dataset/data_kor_v1")