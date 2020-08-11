import cv2
import tensorflow as tf
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime as dt

from base.label import Shape, write_shapes
from base.darknet_utils import load_lp_network, detect_bb
from base.wpod_utils import load_wpod, detect_wpod
from base.utils import image_files_from_folder, draw_label, show

# resize all images in input_dir to (max_side=1024)
def resize_imgs(input_dir, output_dir):
    img_paths = image_files_from_folder(input_dir)
    img_paths.sort()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = dt.today().strftime("%Y%m%d%H%M%S")

    for i, img_path in enumerate(img_paths):
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))

        w, h = np.array(img.shape[1::-1], dtype=int)
        max_side = max(w, h)
        ratio = float(max_side) / 1024.0
        if max_side == w:
            nw = 1024
            nh = int(h / ratio)
        else:
            nw = int(w / ratio)
            nh = 1024
        interpolation = cv2.INTER_AREA if ratio > 1.0 else cv2.INTER_CUBIC
        img_resized = cv2.resize(img, (nw, nh), interpolation=interpolation)

        img_resized_path = Path(output_dir) / (output_prefix + str(i).zfill(4) + img_path.suffix)
        cv2.imwrite(str(img_resized_path), img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def pre_annotate(input_dir, output_dir):
    lp_net, lp_meta = load_lp_network()
    lp_threshold = 0.5

    wpod_net, wpod_net_fn = load_wpod("data/wpod/weights-200.h5")
    wpod_net.summary()
    wpod_threshold = 0.5

    image_paths = image_files_from_folder(input_dir)
    image_paths.sort()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_img_paths = image_files_from_folder(str(output_dir))
    existing_img_paths = sorted(existing_img_paths)
    last_img_path = Path(existing_img_paths[-1]) if len(existing_img_paths) > 0 else ""
    print(f"** previously done till {str(last_img_path)}")

    for _, image_path in enumerate(image_paths):
        image_path = Path(image_path)
        output_image_path = output_dir / image_path.name
        output_txt_path = output_dir / (image_path.stem + "_pre.txt")

        # skip to last done
        if last_img_path != "" and image_path.name <= last_img_path.name:
            continue

        if (output_dir / image_path.name).exists():
            print(" ** skipping  ", str(image_path))
            continue
        else:
            print(" ** annotating", str(image_path))

        image = cv2.imread(str(image_path))
        image_show = image.copy()
        w, h = np.array(image.shape[1::-1], dtype=np.float32)

        # first, detect bouding-box of lp
        margin = 2**4 # same with net_stride of wpod-net
        lp_bb_list = detect_bb(lp_net, lp_meta, image, lp_threshold)

        shapes = []
        for lp_bb in lp_bb_list:
            # absolute points with margin
            l, t, r, b = lp_bb
            l, t, r, b = np.floor(l - margin), np.floor(t - margin), np.ceil(r + margin), np.ceil(b + margin)
            l, t, r, b = max(0, int(l)), max(0, int(t)), min(int(w), int(r)), min(int(h), int(b))

            # get proportional points from wpod
            image_lp = image[t:b, l:r, :]
            wpod_pts, _, _ = detect_wpod(wpod_net_fn, image_lp, wpod_threshold)

            if wpod_pts is None:
                # if wpod-net result invalid, use boundingbox points
                line_color = (255, 0, 255)
                l, t, r, b = lp_bb
                pts = np.array([[l, r, r, l], [t, t, b, b]])
            else:
                # if wpod-net result valid, use warped points
                line_color = (0, 0, 255)
                pts = wpod_pts * np.float32([r-l, b-t]).reshape((2, 1)) + np.float32([l, t]).reshape((2, 1))

            pts_prop = pts / np.array([w, h]).reshape((2, 1))
            pts_prop = np.clip(pts_prop, 0.0, 1.0)
            shapes.append(Shape(pts=pts_prop))
            draw_label(image_show, pts_prop, line_color)

        key = show(image_show)
        if key == 27:
            break
        elif key == 32:
            shutil.copy(str(image_path), str(output_image_path))
            write_shapes(str(output_txt_path), shapes)
        else:
            print("    -> ignored")

if __name__ == "__main__":
    # for kor_data annotaion
    #resize_imgs("_train_wpod/dataset/data_kor_v1_raw", "_train_wpod/dataset/data_kor_v1_resized")
    pre_annotate("_train_wpod/dataset/data_kor_preanno_test", "_train_wpod/dataset/data_kor_preanno_test_pre")


