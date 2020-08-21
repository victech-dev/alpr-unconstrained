import cv2
import time
import tensorflow as tf
import numpy as np
from pathlib import Path

from base.label import Shape, write_shapes
from base.darknet_utils import load_lp_network, load_ocr_network, detect_bb
from base.wpod_utils import load_wpod, detect_wpod, import_saved_model
from base.utils import image_files_from_folder, draw_label, draw_text, draw_bb_from_ltrb, get_precent_color, show

lp_net, lp_meta = None, None
lp_threshold = 0.5

wpod_net, wpod_net_fn = None, None
wpod_threshold = 0.5

ocr_net, orc_meta = None, None
ocr_threshold = 0.3

def prepare_networks(wpod_saved_model_path=None):
    global lp_net, lp_meta, wpod_net, wpod_net_fn, ocr_net, orc_meta

    lp_net, lp_meta = load_lp_network()
    if wpod_saved_model_path is None:
        wpod_net, wpod_net_fn = load_wpod()
        wpod_net.summary()
    else:
        wpod_net_fn = import_saved_model(wpod_saved_model_path)
        print(f"*** wpod_net loaded from {str(wpod_saved_model_path)}")
    ocr_net, orc_meta = load_ocr_network()

def detect(image):
    e0, e1, e2 = 0., 0., 0.

    t0 = time.time()

    w, h = np.array(image.shape[1::-1], dtype=np.int)

    # first, detect bouding-box of lp
    margin = 2**4 # same with net_stride of wpod-net
    lp_bb_list = detect_bb(lp_net, lp_meta, image, lp_threshold, margin)

    e0 = time.time() - t0

    results = []
    for lp_bb in lp_bb_list:
        t0 = time.time()

        # absolute points with margin
        l, t, r, b = lp_bb
        l, t, r, b = np.floor(l), np.floor(t), np.ceil(r), np.ceil(b)
        l, t, r, b = max(0, int(l)), max(0, int(t)), min(int(w), int(r)), min(int(h), int(b))

        # get proportional points from wpod
        image_lp = image[t:b, l:r, :]
        _, image_lp_unwarp, _ = detect_wpod(wpod_net_fn, image_lp, wpod_threshold)

        e1 = e1 + (time.time() - t0)

        t0 = time.time()

        # detect char using ocr
        char_bb_list = detect_bb(ocr_net, orc_meta, image_lp_unwarp, ocr_threshold, use_class=True, nms=0.7)
        lp_txt = "".join([v[0] for v in char_bb_list])
        
        e2 = e2 + (time.time() - t0)

        results.append((lp_bb, lp_txt, char_bb_list, image_lp_unwarp))

    return results, e0, e1, e2

def test_detect(input_dir):
    image_paths = image_files_from_folder(input_dir)
    image_paths.sort()

    for _, image_path in enumerate(image_paths):
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))

        t0 = time.time()
        detect_results, e0, e1, e2 = detect(image)
        t1 = time.time()

        show_ratio = 0.5
        image = cv2.resize(image, dsize=(0, 0), fx=show_ratio, fy=show_ratio, interpolation=cv2.INTER_LINEAR)

        # show result in text and image
        print(f"{image_path.name} : detect {len(detect_results)} plates. elap={t1-t0} ({e0} / {e1} / {e2})")
        for ret in detect_results:
            lp_bb, lp_txt, char_bb_list, _ = ret

            print(f"  plate={lp_txt}")
            for char_bb in char_bb_list:
                print(f"    {char_bb}")
            l, t, r, b = lp_bb
            draw_bb_from_ltrb(image, l * show_ratio, t * show_ratio, r * show_ratio, b * show_ratio)

        draw_w = 10
        for ret in detect_results:
            _, lp_txt, char_bb_list, image_lp_unwarp = ret

            for char_bb in char_bb_list:
                l, t, r, b = char_bb[3:7]
                draw_bb_from_ltrb(image_lp_unwarp, l, t, r, b, line_color=get_precent_color(char_bb[2]))

            unwarp_w, unwarp_h = (image_lp_unwarp.shape[1], image_lp_unwarp.shape[0])
            image[10:10+unwarp_h, draw_w:draw_w+unwarp_w, :] = image_lp_unwarp
            image = draw_text(image, lp_txt, (draw_w, unwarp_h + 10))
            draw_w = draw_w + unwarp_w + 10

        key = show(image)
        if key == 27:
            break

if __name__ == "__main__":
    #prepare_networks()
    prepare_networks("data/wpod/frozen_model_fp16")
    test_detect("_train_wpod/dataset/data_kor_v1_done")
    #test_detect("_0821_site")
     