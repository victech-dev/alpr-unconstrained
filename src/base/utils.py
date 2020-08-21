import numpy as np
import cv2
import sys
import shutil
from glob import glob
import colorsys
from PIL import ImageFont, ImageDraw, Image

def im2single(I):
    assert I.dtype == "uint8"
    return I.astype("float32") / 255.0

def get_wh(shape):
    return np.array(shape[1::-1]).astype(float)

def iou(tl1, br1, tl2, br2):
    wh1, wh2 = br1 - tl1, br2 - tl2
    assert (wh1 >= 0.0).all() and (wh2 >= 0.0).all()

    intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0.0)
    intersection_area = np.prod(intersection_wh)
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area

def iou_labels(l1, l2):
    return iou(l1.tl(), l1.br(), l2.tl(), l2.br())

def iou_centre_and_dims(cc1, wh1, cc2, wh2):
    return iou(cc1 - wh1 / 2.0, cc1 + wh1 / 2.0, cc2 - wh2 / 2.0, cc2 + wh2 / 2.0)

def iou_ltrb(l1, t1, r1, b1, l2, t2, r2, b2):
    tl1, br1 = np.array([t1, l1]), np.array([b1, r1])
    tl2, br2 = np.array([t2, l2]), np.array([b2, r2])
    return iou(tl1, br1, tl2, br2)

def nms(Labels, iou_threshold=0.5):
    SelectedLabels = []
    Labels.sort(key=lambda l: l.prob(), reverse=True)

    for label in Labels:
        non_overlap = True
        for sel_label in SelectedLabels:
            if iou_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break

        if non_overlap:
            SelectedLabels.append(label)

    return SelectedLabels

def image_files_from_folder(folder, upper=True):
    extensions = ["jpg", "jpeg", "png"]
    image_files = []
    for ext in extensions:
        image_files += glob("%s/*.%s" % (folder, ext))
        if upper:
            image_files += glob("%s/*.%s" % (folder, ext.upper()))
    return image_files

def is_inside(ltest, lref):
    return (ltest.tl() >= lref.tl()).all() and (ltest.br() <= lref.br()).all()

def hsv_transform(I, hsv_modifier):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    I = I + hsv_modifier
    return cv2.cvtColor(I, cv2.COLOR_HSV2BGR)

def show(image, px=-1, py=-1):
    winname = "image"
    cv2.namedWindow(winname)
    if px >= 0 and py >= 0:
        cv2.moveWindow(winname, px, py)
    cv2.imshow(winname, image)
    key = cv2.waitKey(0) & 0xEFFFFF
    cv2.destroyWindow(winname)
    return key

def draw_label(image, pts_prop, line_color=(0, 0, 255)):
    w, h = image.shape[1], image.shape[0]
    pts_prop = np.asarray(pts_prop)
    for i in range(4):
        cv2.line(
            image,
            (int(pts_prop[0][i] * w), int(pts_prop[1][i] * h)), 
            (int(pts_prop[0][(i+1)%4] * w), int(pts_prop[1][(i+1)%4] * h)),
            line_color, thickness=1)

def draw_bb_from_ltrb(image, l, t, r, b, line_color=(0, 0, 255)):
    w, h = image.shape[1], image.shape[0]
    l, t, r, b = max(int(l), 0), max(int(t), 0), min(int(r), w), min(int(b), h)
    cv2.rectangle(image, (l, t), (r, b), line_color, thickness=1)

def draw_text(image, text, pos=(0, 0), color_bgra=(0,0,255,0)):
    fontpath = "data/ocr-kor/font_kor/MapoDPP.ttf"
    font = ImageFont.truetype(fontpath, 40)
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    draw.text(pos, text, font=font, fill=color_bgra)
    return np.array(image_pil)

def get_precent_color(percent):
    # 0=green, 1=red
    percent = min(1., max(0., percent))
    hue = ((1. - percent) * 120.) / 360.
    r, g, b = tuple(v * 255. for v in colorsys.hls_to_rgb(hue, 0.5, 1.0))
    return (b, g, r)

def split_train_val(path):
    path = Path(path)
    train_path = (path / 'train')
    val_path = (path / 'val')
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    image_paths = [str(f) for f in path.glob('**/*.jpg')]
    random.shuffle(image_paths)
    
    split_len = int(len(image_paths) * 0.7)
    for i, image_path in enumerate(image_paths):
        image_path = Path(image_path)
        txt_path = image_path.parent / (image_path.stem + '.txt')
        dst_path = train_path if i < split_len else val_path
        shutil.move(str(image_path), str(dst_path / image_path.name))
        shutil.move(str(txt_path), str(dst_path / txt_path.name))
