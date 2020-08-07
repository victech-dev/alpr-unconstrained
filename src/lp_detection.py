import sys
import cv2
import numpy as np
import traceback
import time
from pathlib import Path
from os.path import isdir
from os import makedirs

import darknet.darknet as dn
from base.label import Label, lwrite
from base.utils import crop_region, image_files_from_folder

def load_lp_network():
    config = "data/lp/yolo-obj.cfg"
    weight = "data/lp/yolo-obj_best.weights"
    meta = "data/lp/obj.data"
    return dn.load_network(config, weight, meta)

# for simple detection
def detect_lp_bb(net, meta, image, threshold):
    ret, image_wh = dn.detect_cv2image(net, meta, image, thresh=threshold)
    bb_list = []
    for r in ret:
        cx, cy, w, h = r[3][:4]
        l, r, t, b = cx - w * 0.5, cx + w * 0.5, cy - h * 0.5, cy + h * 0.5
        l, r, t, b = max(0, l), min(image_wh[0], r), max(0, t), min(image_wh[1], b)
        bb_list.append((l, t, r, b))
    return bb_list

def detect_lp_labels(net, meta, image, threshold, preserve_ratio=True):
    ret, image_wh = dn.detect_cv2image(net, meta, image, thresh=threshold)

    labels = []
    # TODO np.array로 한번에 처리?
    # TODO margin to detected bb?
    for _, r in enumerate(ret):
        # <r sample>
        # (b'LP', 0, 0.9673437476158142, (951.8226412259615, 351.51587500939, 94.64744215745193, 54.00713700514573))

        cx, cy, w, h = r[3][:4]
        if not preserve_ratio:
            w, h = max(w, h), max(w, h)
        cx, cy, w, h = (np.array([cx, cy, w, h]) /
                        np.concatenate((image_wh, image_wh))).tolist()

        l, r, t, b = cx - w * 0.5, cx + w * 0.5, cy - h * 0.5, cy + h * 0.5
        if t < 0.0:
            b = b - t
            t = 0.0
        if b > 1.0:
            t = t - (b - 1.0)
            b = 1.0
        if l < 0.0:
            r = r - l
            l = 0.0
        if r > 1.0:
            l = l - (r - 1.0)
            r = 1.0

        label = Label(0, np.array([t, l]), br=np.array([b, r]))
        labels.append(label)
    return labels

def detect_lp_from_folder(input_path, output_path):
    net, meta = load_lp_network()
    detect_threshold = 0.5

    image_paths = image_files_from_folder(input_path)
    image_paths.sort()

    if not isdir(output_path):
        makedirs(output_path)

    for _, image_path in enumerate(image_paths):
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        hw = np.array(image.shape[:2])

        labels = detect_lp_labels(net, meta, image, detect_threshold)
        for i, label in enumerate(labels):
            tl = np.floor(label.tl() * hw).astype(int)
            br = np.ceil(label.br() * hw).astype(int)
            image_lp = image[tl[0]:br[0], tl[1]:br[1],:]
            cv2.imwrite("%s/%s_%d_lp.png" % (output_path, image_path.stem, i), image_lp)
        if len(labels):
            lwrite("%s/%s_lps.txt" % (output_path, image_path.stem), labels)

    print(f"* {len(image_paths)} image processed")

if __name__ == "__main__":
    # input_path = sys.argv[1]
    # output_path = sys.argv[2]
    input_path = "_train_wpod/dataset/samples"
    output_path = "_train_wpod/dataset/samples_"
    detect_lp_from_folder(input_path, output_path)
