import numpy as np
import cv2
import sys

from glob import glob

def im2single(I):
    assert I.dtype == "uint8"
    return I.astype("float32") / 255.0

def get_wh(shape):
    return np.array(shape[1::-1]).astype(float)

def IOU(tl1, br1, tl2, br2):
    wh1, wh2 = br1 - tl1, br2 - tl2
    assert (wh1 >= 0.0).all() and (wh2 >= 0.0).all()

    intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0.0)
    intersection_area = np.prod(intersection_wh)
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area

def IOU_labels(l1, l2):
    return IOU(l1.tl(), l1.br(), l2.tl(), l2.br())

def IOU_centre_and_dims(cc1, wh1, cc2, wh2):
    return IOU(cc1 - wh1 / 2.0, cc1 + wh1 / 2.0, cc2 - wh2 / 2.0, cc2 + wh2 / 2.0)

def nms(Labels, iou_threshold=0.5):
    SelectedLabels = []
    Labels.sort(key=lambda l: l.prob(), reverse=True)

    for label in Labels:
        non_overlap = True
        for sel_label in SelectedLabels:
            if IOU_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break

        if non_overlap:
            SelectedLabels.append(label)

    return SelectedLabels

def image_files_from_folder(folder, upper=True):
    extensions = ["jpg", "jpeg", "png"]
    img_files = []
    for ext in extensions:
        img_files += glob("%s/*.%s" % (folder, ext))
        if upper:
            img_files += glob("%s/*.%s" % (folder, ext.upper()))
    return img_files

def is_inside(ltest, lref):
    return (ltest.tl() >= lref.tl()).all() and (ltest.br() <= lref.br()).all()

def hsv_transform(I, hsv_modifier):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    I = I + hsv_modifier
    return cv2.cvtColor(I, cv2.COLOR_HSV2BGR)

def show(img):
    cv2.imshow("img", img)
    key = cv2.waitKey(0) & 0xEFFFFF
    cv2.destroyWindow("img")
    return key == 27

def draw_label(img, pts, line_color=(0, 0, 255)):
    w, h = img.shape[1], img.shape[0]
    pts = np.asarray(pts)
    for i in range(4):
        cv2.line(
            img,
            (int(pts[0][i] * w), int(pts[1][i] * h)), 
            (int(pts[0][(i+1)%4] * w), int(pts[1][(i+1)%4] * h)),
            line_color, thickness=1)
