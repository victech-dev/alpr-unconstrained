import cv2
import numpy as np
from pathlib import Path
from os.path import isdir
from os import makedirs

from base.utils import image_files_from_folder
from base.darknet_utils import load_lp_network, detect_lp_labels
from base.label import Label, lwrite

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
    input_path = "_train_wpod/dataset/samples"
    output_path = "_train_wpod/dataset/samples_"
    detect_lp_from_folder(input_path, output_path)
