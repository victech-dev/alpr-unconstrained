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
from base.wpod_utils import export_to_frozen_saved_model, export_to_trt_fp16_model, import_saved_model
from base.label import Shape, write_shapes
from base.utils import image_files_from_folder

def export_to_saved_model(path, trt_fp16_path):
    wpod_net, _ = load_wpod()
    export_to_frozen_saved_model(wpod_net, path, verbose=1)
    export_to_trt_fp16_model(path, trt_fp16_path, verbose=1)

def detect_folder(wpod_net_fn, input_path, output_path):
    wpod_threshold = 0.5

    image_paths = image_files_from_folder(input_path)
    image_paths.sort()

    if not isdir(output_path):
        makedirs(output_path)

    for _, image_path in enumerate(image_paths):
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        
        out_label, out_image, confidence = detect_wpod(wpod_net_fn, image, wpod_threshold)
        print(f"* {str(image_path)} processing. confidence={confidence}")

        cv2.imwrite("%s/%s_unwarp.png" % (output_path, image_path.stem), out_image)
        write_shapes("%s/%s_unwarp.txt" % (output_path, image_path.stem), [Shape(pts=out_label)])

    print(f"* {len(image_paths)} image processed")

def detect_folder_from_model(input_path, output_path):
    wpod_net, wpod_net_fn = load_wpod()
    wpod_net.summary()
    detect_folder(wpod_net_fn, input_path, output_path)

def detect_folder_from_frozen_model(model_path, input_path, output_path):
    wpod_net_fn = import_saved_model(model_path)
    detect_folder(wpod_net_fn, input_path, output_path)

if __name__ == "__main__":
    saved_model_path = "data/wpod/frozen_model"
    saved_model_fp16_path = "data/wpod/frozen_model_fp16"
    # export_to_saved_model(saved_model_path, saved_model_fp16_path)

    input_path = "_train_wpod/dataset/samples_cropped"
    output_path = "_train_wpod/dataset/samples_cropped_"
    # detect_folder_from_model(input_path, output_path)
    detect_folder_from_frozen_model("data/wpod/frozen_model", input_path, output_path)

    
