import sys
import cv2
import numpy as np
import traceback
import time
from pathlib import Path

import darknet.python.darknet as dn

from src.label import Label, lwrite
from os.path import splitext, basename, isdir
from os import makedirs
from src.utils import crop_region, image_files_from_folder
from darknet.python.darknet import detect


def detect_lp(input_dir, output_dir):
    vehicle_threshold = 0.5

    # for yolo4.cfg
    vehicle_weights = "data/vehicle-detector/yolo-obj_best.weights"
    vehicle_netcfg = "data/vehicle-detector/yolo-obj.cfg"
    vehicle_dataset = "data/vehicle-detector/obj.data"
    classes_on_interest = [b"LP"]

    vehicle_net = dn.load_net(vehicle_netcfg.encode("ascii"),
                              vehicle_weights.encode("ascii"), 0)
    vehicle_meta = dn.load_meta(vehicle_dataset.encode("ascii"))

    imgs_paths = image_files_from_folder(input_dir)
    imgs_paths.sort()

    if not isdir(output_dir):
        makedirs(output_dir)

    print("** Searching for vehicles using yolo(v4)...")

    for i, img_path in enumerate(imgs_paths):
        img_path = Path(img_path)

        ret, _, elapsed = detect(
            vehicle_net,
            vehicle_meta,
            str(img_path).encode("ascii"),
            thresh=vehicle_threshold,
        )
        ret = [r for r in ret if r[0] in classes_on_interest]
        print(
            "    ",
            img_path,
            "detect cnt =",
            len(ret),
            ", t =",
            elapsed[-1] - elapsed[0],
            "sec",
        )

        if len(ret):
            img_org = cv2.imread(str(img_path))
            wh = np.array(img_org.shape[1::-1], dtype=float)
            car_labels = []
            output_side = 288

            for i, r in enumerate(ret):
                # r sample : (
                #   b'LP',
                #   0.7768856287002563,
                #   (667.3297119140625, 410.9588928222656, 52.92448806762695, 16.414148330688477)
                # )

                # TODO margin to detected bb

                # preserve 1:1 ratio
                cx, cy, w, h = r[2][:4]
                side = max(output_side, max(w, h))
                cx, cy, w, h = (np.array([cx, cy, side, side]) /
                                np.concatenate((wh, wh))).tolist()
                t, l, b, r = (
                    cx - w / 2.0,
                    cy - h / 2.0,
                    cx + w / 2.0,
                    cy + h / 2.0,
                )
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
                img_car = crop_region(img_org, label)
                if img_car.shape[:2] != [output_side, output_side]:
                    img_car = cv2.resize(img_car, (output_side, output_side))
                car_labels.append(label)

                cv2.imwrite("%s/%s_%dcar.png" % (output_dir, img_path.stem, i),
                            img_car)

            lwrite("%s/%s_cars.txt" % (output_dir, img_path.stem), car_labels)


if __name__ == "__main__":
    try:
        detect_lp(sys.argv[1], sys.argv[2])
    except:
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)
