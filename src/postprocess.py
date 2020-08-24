import cv2
import numpy as np
from imgaug.augmentables.bbs import BoundingBox
from base.label import Label
from base.utils import IOU_labels

def _unravel(labels):
    ys = np.float32([l.cc()[1] for l in labels])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    _, gidx, _ = cv2.kmeans(ys, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    gidx = gidx.flatten()
    g0 = [labels[i] for i, g in enumerate(gidx) if g == 0]
    g1 = [labels[i] for i, g in enumerate(gidx) if g == 1]
    lt0, rb0 = np.min([l.tl() for l in g0], 0), np.max([l.br() for l in g0], 0)
    lt1, rb1 = np.min([l.tl() for l in g1], 0), np.max([l.br() for l in g1], 0)
    bb0 = Label(-1, np.array([0, lt0[1]]), np.array([1, rb0[1]]))
    bb1 = Label(-1, np.array([0, lt1[1]]), np.array([1, rb1[1]]))
    if IOU_labels(bb0, bb1) > 0.33:
        return 1, sorted(labels, key=lambda l: l.cc()[0])
    else:
        if bb0.cc()[1] > bb1.cc()[1]:
            g0, g1 = g1, g0
        g0.sort(key=lambda l: l.cc()[0])
        g1.sort(key=lambda l: l.cc()[0])
        return 2, g0 + g1

def _get_htext(labels, all_chars):
    return ''.join([all_chars[l.cl()] for l in labels])

def solve_to_text(labels, all_chars):
    labels.sort(key=lambda l: l.cc()[0])
    if len(labels) < 2:
        return _get_htext(labels, all_chars)

    # 세로 지역2 글자 확인
    n, vregion = _unravel(labels[:2])
    if n == 2:
        n, flat = _unravel(labels[2:])
        if n == 1:
            return _get_htext(vregion, all_chars) + _get_htext(flat, all_chars)

    _, flat = _unravel(labels)
    return _get_htext(flat, all_chars)

