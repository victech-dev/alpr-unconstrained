import cv2
import time
import tensorflow as tf
import numpy as np
from pathlib import Path

from base.label import Shape, write_shapes
from base.darknet_utils import load_lp_network, load_ocr_network, detect_bb
from base.wpod_utils import load_wpod, detect_wpod
from base.utils import image_files_from_folder, draw_label, show

lp_net, lp_meta = None, None
lp_threshold = 0.5

wpod_net, wpod_net_fn = None, None
wpod_threshold = 0.5

ocr_net, orc_meta = None, None
ocr_threshold = 0.5

def prepare_networks():
    global lp_net, lp_meta, wpod_net, wpod_net_fn, ocr_net, orc_meta

    lp_net, lp_meta = load_lp_network()
    wpod_net, wpod_net_fn = load_wpod()
    ocr_net, orc_meta = load_ocr_network()
    wpod_net.summary()

def detect(image):
    w, h = np.array(image.shape[1::-1], dtype=np.int)

    # first, detect bouding-box of lp
    margin = 2**4 # same with net_stride of wpod-net
    lp_bb_list = detect_bb(lp_net, lp_meta, image, lp_threshold, margin)

    chars_list = []
    for lp_bb in lp_bb_list:
        # absolute points with margin
        l, t, r, b = lp_bb
        l, t, r, b = np.floor(l), np.floor(t), np.ceil(r), np.ceil(b)
        l, t, r, b = max(0, int(l)), max(0, int(t)), min(int(w), int(r)), min(int(h), int(b))

        # get proportional points from wpod
        image_lp = image[t:b, l:r, :]
        _, wpod_image, _ = detect_wpod(wpod_net_fn, image_lp, wpod_threshold)

        # detect char using ocr
        char_bb_list = detect_bb(ocr_net, orc_meta, wpod_image, ocr_threshold, use_cls=True)
        char_bb_list.sort(key=lambda v: (v[3] + v[5]) * 0.5) # sort by cx
        chars_list.append("".join([v[0] for v in char_bb_list]))

    return chars_list

def test_detect(input_dir):
    image_paths = image_files_from_folder(input_dir)
    image_paths.sort()

    for _, image_path in enumerate(image_paths):
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))

        t0 = time.time()
        chars_list = detect(image)
        t1 = time.time()

        chars_all = " / ".join(chars_list)
        print(f"{image_path.name} : chars='{chars_all}, elap={t1-t0}")
        key = show(image)
        if key == 27:
            break

if __name__ == "__main__":
    print("aaa")
    prepare_networks()
    test_detect("_train_wpod/dataset/data_kor_v1_done")
