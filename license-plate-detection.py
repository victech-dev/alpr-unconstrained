import sys, os
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import traceback
import time

from src.keras_utils import load_model
from glob import glob
from os.path import splitext, basename
from src.utils import im2single
from src.keras_utils import load_model, detect_lp
from src.label import Shape, writeShapes

if __name__ == '__main__':

    try:
        input_dir = sys.argv[1]
        output_dir = input_dir

        lp_threshold = .1

        wpod_net_path = sys.argv[2]

        wpod_net = load_model(wpod_net_path)
        wpod_net.summary()

        # for tf 2.0
        @tf.function(input_signature=[
            tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32)
        ])
        def wpod_net_fn(img):
            return wpod_net.call(img)

        # wpod_net_fn = tf.function(wpod_net.call)
        # wpod_net_fn_concreate = wpod_net_fn.get_concrete_function(
        # 	(tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32, name="img")))

        imgs_paths = glob('%s/*_lp.png' % input_dir)
        imgs_paths = sorted(imgs_paths)

        print('Searching for license plates using WPOD-NET')

        for i, img_path in enumerate(imgs_paths):
            bname = splitext(basename(img_path))[0]
            Ivehicle = cv2.imread(img_path)

            ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
            side = int(ratio * 288.)
            bound_dim = min(side + (side % (2**4)), 608)

            t0 = time.time()
            Llp, LlpImgs, _, t_shape, yr_shape, yr_conf_max = detect_lp(
                wpod_net_fn, im2single(Ivehicle), bound_dim, 2**4,
                (240, 80), lp_threshold)
            t1 = time.time()

            print("**", img_path, ", w=", Ivehicle.shape[1], ", h=", Ivehicle.shape[0],
                ", Bound dim=", bound_dim, ", ratio=", ratio,
                ", t=", t1 - t0, ", t.shape=", t_shape,
                  ", yr.shape=", yr_shape, ", conf.max=", yr_conf_max)

            if len(LlpImgs):
                Ilp = LlpImgs[0]
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

                s = Shape(Llp[0].pts)

                cv2.imwrite('%s/%s_unwarp.png' % (output_dir, bname), Ilp * 255.)
                writeShapes('%s/%s_unwarp.txt' % (output_dir, bname), [s])

    except:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)
