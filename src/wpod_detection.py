import sys, os
import time
from pathlib import Path
from os.path import isdir
from os import makedirs
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from base.wpod_utils import load_wpod, detect_wpod
from base.label import Shape, write_shapes
from base.utils import image_files_from_folder

def detect_wpod_from_from_folder(input_path, output_path):
    wpod_net, wpod_net_fn = load_wpod("data/wpod/weights-200.h5")
    wpod_net.summary()

    image_paths = image_files_from_folder(input_path)
    image_paths.sort()

    if not isdir(output_path):
        makedirs(output_path)

    for _, image_path in enumerate(image_paths):
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        threshold = 0.5
        out_label, out_image, confidence = detect_wpod(wpod_net_fn, image, threshold)
        print(f"* {str(image_path)} processing. confidence={confidence}")

        cv2.imwrite("%s/%s_unwarp.png" % (output_path, image_path.stem), out_image)
        write_shapes("%s/%s_unwarp.txt" % (output_path, image_path.stem), [Shape(pts=out_label)])

    print(f"* {len(image_paths)} image processed")

if __name__ == "__main__":
    # input_path = sys.argv[1]
    # output_path = sys.argv[2]
    input_path = "_train_wpod/dataset/samples_cropped"
    output_path = "_train_wpod/dataset/samples_cropped_"
    detect_wpod_from_from_folder(input_path, output_path)
