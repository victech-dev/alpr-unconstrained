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

import src.darknet.darknet as dn

from src.label import Label, Shape, writeShapes
from src.utils import image_files_from_folder, im2single, nms
from src.keras_utils import load_model

net_stride = 2**4

def resize_imgs(input_dir, output_dir):
    img_paths = image_files_from_folder(input_dir)
    img_paths.sort()

    output_dir = Path(output_dir)
    if not output_dir.exists():
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

class LabelEx(Label):
    def __init__(self, cl, pts, prob, pts_mn):
        self.pts = pts
        self.pts_mn = pts_mn
        tl = np.amin(pts, 1)
        br = np.amax(pts, 1)
        Label.__init__(self, cl, tl, br, prob)

def detect_lp_bb(img_path, model_net, model_meta, classes_on_interest, threshold):
    ret, _, _ = dn.detect(
        model_net,
        model_meta,
        str(img_path).encode("ascii"),
        thresh=threshold,
    )
    ret = [r for r in ret if r[0] in classes_on_interest]

    lp_labels = []
    if len(ret):
        for i, r in enumerate(ret):
            cx, cy, w, h = np.array(r[2][:4]).astype(int)
            lp_labels.append((cx, cy, w, h))

    return lp_labels

def detect_lp_pts(img, wpod_net_ret, threshold=.9):
    side = ((208. + 40.) / 2.) / net_stride  # 7.75

    Probs = wpod_net_ret[..., 0]
    Affines = wpod_net_ret[..., 2:]

    yy, xx = np.where(Probs > threshold)

    WH = np.array(img.shape[1::-1]).astype(float)
    MN = WH / net_stride

    vxx = vyy = 0.5  #alpha
    base = lambda vx, vy: np.matrix([[-vx, -vy, 1.], [vx, -vy, 1.],
                                     [vx, vy, 1.], [-vx, vy, 1.]]).T

    labels = []
    for i in range(len(yy)):
        y, x = yy[i], xx[i]
        affine = Affines[y, x]
        prob = Probs[y, x]

        mn = np.array([float(x) + .5, float(y) + .5])

        A = np.reshape(affine, (2, 3))
        A[0, 0] = max(A[0, 0], 0.)
        A[1, 1] = max(A[1, 1], 0.)

        pts = np.array(A * base(vxx, vyy))  #*alpha
        pts_MN_center_mn = pts * side
        pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
        pts_prop = pts_MN / MN.reshape((2, 1))

        labels.append(LabelEx(0, pts_prop, prob, pts_MN))

    final_labels = nms(labels, .1)
    final_labels.sort(key=lambda x: x.prob(), reverse=True)

    ret = []
    if len(final_labels):
        print(" ** prob =", final_labels[0].prob())
        label0 = final_labels[0]
        ret.append(label0.pts_mn)

    return ret

def pre_annotate(input_dir, output_dir):
    # Load lp detector
    vehicle_weights = "data/vehicle-detector/yolo-obj_best.weights"
    vehicle_netcfg = "data/vehicle-detector/yolo-obj.cfg"
    vehicle_dataset = "data/vehicle-detector/obj.data"
    classes_on_interest = [b"LP"]
    vehicle_threshold = 0.5

    vehicle_net = dn.load_net(vehicle_netcfg.encode("ascii"),
                              vehicle_weights.encode("ascii"), 0)
    vehicle_meta = dn.load_meta(vehicle_dataset.encode("ascii"))

    # Load wpod-net
    #wpod_net_path = "data/lp-detector/wpod-net_update1.h5"
    wpod_net_path = "data/lp-detector/weights-200.h5"
    wpod_net = load_model(wpod_net_path)
    wpod_net.summary()
    wpod_net_threshold = .3

    # for tf 2.0
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32)
    ])
    def wpod_net_fn(img):
        return wpod_net.call(img)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = image_files_from_folder(input_dir)
    img_paths.sort()

    existing_img_paths = image_files_from_folder(str(output_dir))
    existing_img_paths = sorted(existing_img_paths)
    last_img_path = Path(existing_img_paths[-1]) if len(existing_img_paths) > 0 else ""
    print(str(last_img_path))

    print("** Searching for license-plate using yolo(v4)...")

    for _, img_path in enumerate(img_paths):
        img_path = Path(img_path)
        output_img_path = output_dir / img_path.name
        output_txt_path = output_dir / (img_path.stem + "_pre.txt")

        # skip to last done
        if last_img_path != "" and img_path.name <= last_img_path.name:
            continue

        if (output_dir / img_path.name).exists():
            print(" ** skipping  ", str(img_path))
            continue
        else:
            print(" ** annotating", str(img_path))

        img = cv2.imread(str(img_path))
        img_show = img.copy()
        img_w, img_h = np.array(img.shape[1::-1], dtype=int)

        labels = detect_lp_bb(img_path, vehicle_net, vehicle_meta,
                              classes_on_interest, vehicle_threshold)

        margin = net_stride
        max_side = 208
        shapes = []
        for label in labels:
            cx, cy, w, h = label

            pw, ph = w + margin * 2, h + margin * 2
            pl, pr = max(cx - pw // 2, 0), min(cx + pw // 2, img_w)
            pt, pb = max(cy - ph // 2, 0), min(cy + ph // 2, img_h)
            pw, ph = pr - pl, pb - pt

            max_ratio = float(max(pw, ph)) / float(max_side)
            max_ratio = max(max_ratio, 1.0)
            nw, nh = int(pw / max_ratio), int(ph / max_ratio)
            nw = ((nw + net_stride - 1) // net_stride) * net_stride
            nh = ((nh + net_stride - 1) // net_stride) * net_stride

            img_lp_prev = img[pt:pb, pl:pr, :]
            img_lp = cv2.resize(img_lp_prev, (nw, nh))
            ratio = np.array((float(pw) / float(nw), float(ph) / float(nh)))
            ratio = np.reshape(ratio, (2, 1))

            wname = "img"
            cv2.imshow(wname, img_lp)
            cv2.waitKey(0)
            cv2.destroyWindow(wname)

            tensor_img_lp = im2single(img_lp)
            tensor_img_lp = np.expand_dims(tensor_img_lp, 0)
            tensor_img_lp = tf.convert_to_tensor(tensor_img_lp)
    
            wpod_net_fn_concreate = wpod_net_fn.get_concrete_function(tensor_img_lp)
            wpod_net_ret = wpod_net_fn_concreate(tensor_img_lp).numpy()
            wpod_net_ret = np.squeeze(wpod_net_ret)

            wpod_net_pts = detect_lp_pts(img_lp, wpod_net_ret, wpod_net_threshold)
            if len(wpod_net_pts) > 0:
                # if wpod-net result valid, use warped points
                line_color = (0, 0, 255)
                pts = wpod_net_pts[0] * net_stride * ratio + np.reshape(np.array((pl, pt)), (2, 1))
            else:
                # if wpod-net result invalid, use boundingbox points
                line_color = (255, 0, 255)
                pts = np.array([
                    [ cx - w // 2, cx + w // 2, cx + w // 2, cx - w // 2 ],
                    [ cy - h // 2, cy - h // 2, cy + h // 2, cy + h // 2 ],
                ])

            pts_prop = pts / np.array([float(img_w), float(img_h)]).reshape((2, 1))
            pts_prop = np.clip(pts_prop, 0.0, 1.0)
            shapes.append(Shape(pts=pts_prop))

            for i in range(4):
                cv2.line(
                    img_show,
                    (int(pts[0][i]), int(pts[1][i])), 
                    (int(pts[0][(i+1)%4]), int(pts[1][(i+1)%4])),
                    line_color, thickness=1)

        wname = "img"
        cv2.imshow(wname, img_show)
        key = cv2.waitKey(0) & 0xEFFFFF
        cv2.destroyWindow(wname)
        if key == 27:
            break
        elif key == 32:
            shutil.copy(str(img_path), str(output_img_path))
            writeShapes(str(output_txt_path), shapes)
        else:
            print("    -> ignored")

if __name__ == "__main__":
    # for unit test
    #pre_annotate("samples/test_oid", "tmp/test_oid")

    # for oid_data annotation
    #pre_annotate("/workspace/darknet/_train_lp/data/obj", "./_train_wpod/data/data_oid")

    # for kor_data annotaion
    #resize_imgs("_train_wpod/dataset/data_kor_v1_raw", "_train_wpod/dataset/data_kor_v1_resized")
    #pre_annotate("_train_wpod/dataset/data_kor_v1_resized", "_train_wpod/dataset/data_kor_v1")

    # for test wpod-net custom trained
    #pre_annotate("_train_wpod/dataset/data_oid_preanno_test", "_train_wpod/dataset/data_oid_preanno_test_pre")
    pre_annotate("_train_wpod/dataset/data_kor_preanno_test", "_train_wpod/dataset/data_kor_preanno_test_pre")


