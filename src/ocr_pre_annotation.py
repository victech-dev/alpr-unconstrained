import sys, os
import cv2
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from glob import glob
import shutil
from PIL import Image, ImageDraw, ImageFont

import darknet.darknet as dn

from base.label import read_shapes
from base.utils import image_files_from_folder, show
from base.darknet_utils import write_labels, load_ocr_network, detect_bb
from base.wpod_utils import ocr_unwarp_margin, ocr_input_wh

def unwarp_wpodnet_dataset_from_label(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = image_files_from_folder(str(input_dir))
    image_paths.sort()

    for _, image_path in enumerate(image_paths):
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))

        label_path = input_dir / (image_path.stem + ".txt")
        labels = read_shapes(str(label_path))

        for i, label in enumerate(labels):
            w, h = image.shape[1], image.shape[0]
            pts0 = np.multiply(label.pts.T, (w, h)).astype(np.float32)
            dl, dr = 0. + ocr_unwarp_margin, ocr_input_wh[0] - ocr_unwarp_margin
            dt, db = 0. + ocr_unwarp_margin, ocr_input_wh[1] - ocr_unwarp_margin
            pts1 = np.array([ [dl, dt], [dr, dt], [dr, db], [dl, db] ]).astype(np.float32)

            #print(pts0.dtype, pts1.dtype)
            mat = cv2.getPerspectiveTransform(pts0, pts1)
            img_unwarp = cv2.warpPerspective(image, mat, ocr_input_wh)

            output_file_name = image_path.stem + "_" + str(i) + ".jpg"
            img_unwarp_path = output_dir / output_file_name
            cv2.imwrite(str(img_unwarp_path), img_unwarp, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            print(output_file_name)
            print(pts0, pts1)

def pre_annotate_ocr(input_dir, output_dir=None, preview=False):
    input_dir = Path(input_dir)
    if not preview:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = image_files_from_folder(str(input_dir))
    image_paths.sort()

    net, meta = load_ocr_network()
    detect_threshold = 0.5

    # font for show
    if preview:
        show_font = ImageFont.truetype('data/ocr-kor/font_kor/BlackHanSans-Regular.ttf', size=30)

    for _, image_path in enumerate(image_paths):
        image_path = Path(image_path)
        if not preview:
            output_img_path = output_dir / image_path.name
            output_txt_path = output_dir / (image_path.stem + "_pre.txt")

        print(" ** annotating", str(image_path))

        image = cv2.imread(str(image_path))
        image_w, image_h = np.array(image.shape[1::-1], dtype=int)

        show_factor = 3
        image_show = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_show = image_show.resize((image_w * show_factor, image_h * show_factor))
        img_show_draw = ImageDraw.Draw(image_show)

        bb_list = detect_bb(net, meta, image, detect_threshold, use_cls=True)
        labels = []
        for bb in bb_list:
            class_name, class_idx, _, l, t, r, b = bb
            l, r, t, b = tuple([show_factor * x for x in (l, r, t, b)])
            img_show_draw.line([l, t, r, t, r, b, l, b, l, t], fill = (255, 0, 255))
            img_show_draw.text((l + 2, t + 2), class_name, fill=(0, 0, 255, 64), font=show_font)

            cx, cy, w, h = (
                float((l + r) * 0.5) / image_w,
                float((t + b) * 0.5) / image_h,
                float(r - l) / image_w, 
                float(b - t) / image_h
            )
            labels.append((class_name, class_idx, cx, cy, w, h))

        key = show(cv2.cvtColor(np.asarray(image_show), cv2.COLOR_RGB2BGR))
        if key == 27:
            break
        elif key == 32:
            if not preview:
                shutil.copy(str(image_path), str(output_img_path))
                write_labels(str(output_txt_path), labels)
        else:
            print("    -> ignored")

if __name__ == "__main__":
    # unwarp_wpodnet_dataset_from_label(
    #     "_train_wpod/dataset/data_kor_v1_done",
    #     "_train_wpod/dataset/data_kor_v1_unwarp")

    #pre_annotate_ocr("_train_wpod/dataset/data_kor_v1_unwarp", "_train_ocr/dataset/data_kor_v1")
    pre_annotate_ocr("_train_ocr/dataset/data_kor_v1/val", preview=True)
    