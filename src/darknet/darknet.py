#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Both the GPU and no-GPU version should be compiled; the no-GPU version should be renamed "yolo_cpp_dll_nogpu.dll".

On a GPU system, you can force CPU evaluation by any of:

- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "FORCE_CPU" to "true"


To use, either run performDetect() after import, or modify the end of this file.

See the docstring of performDetect() for parameters.

Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)


Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn
@date: 20180503
"""
#pylint: disable=R, W0401, W0614, W0703
from ctypes import *
import math
import random
import os
import cv2
import numpy as np

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
hasGPU = True
lib = CDLL("darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

def network_width(net):
    return lib.network_width(net)

def network_height(net):
    return lib.network_height(net)

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)

def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        nameTag = meta.names[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect_image(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45, ratio=(1.0, 1.0)):
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    letter_box = 0
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letter_box)
    #predict_image_letterbox(net, im)
    #letter_box = 1
    #dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV
    num = pnum[0]
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                nameTag = meta.names[i]
                res.append((nameTag, i, dets[j].prob[i], (b.x * ratio[0], b.y * ratio[1], b.w * ratio[0], b.h * ratio[1])))
    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    wh = (im.w, im.h)
    return res, wh

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    """
    Performs the meat of the detection
    """
    #pylint: disable= C0321
    # image should be converted by .encode("ascii")
    im = load_image(image, 0, 0)
    res, wh = detect_image(net, meta, im, thresh, hier_thresh, nms)
    free_image(im)
    return ret, wh

def detect_cv2image(net, meta, cv2image, thresh=.5, hier_thresh=.5, nms=.45):
    cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
    nwh = np.array((lib.network_width(net), lib.network_height(net)))
    iwh = np.array(cv2image.shape[1::-1])
    ratio = iwh / nwh
    cv2image = cv2.resize(cv2image, tuple(nwh), interpolation=cv2.INTER_LINEAR)
    im, arr = array_to_image(cv2image) # need to assign arr to avoid free data
    res, _ = detect_image(net, meta, im, thresh, hier_thresh, nms, ratio)
    # do not free_image(im)
    return res, iwh

def load_network(configPath, weightPath, metaPath, batch_size=1):
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `"+os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `"+os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `"+os.path.abspath(metaPath)+"`")
    net = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, batch_size)
    meta = load_meta(metaPath.encode("ascii"))
    return net, meta

# TODO batch prediction 코드 정리

# def performBatchDetect(thresh= 0.25, configPath = "./cfg/yolov4.cfg", weightPath = "yolov4.weights", metaPath= "./cfg/coco.data", hier_thresh=.5, nms=.45, batch_size=3):
#     import cv2
#     import numpy as np
#     # NB! Image sizes should be the same
#     # You can change the images, yet, be sure that they have the same width and height
#     img_samples = ['data/person.jpg', 'data/person.jpg', 'data/person.jpg']
#     image_list = [cv2.imread(k) for k in img_samples]

#     net = load_net_custom(configPath.encode('utf-8'), weightPath.encode('utf-8'), 0, batch_size)
#     meta = load_meta(metaPath.encode('utf-8'))
#     pred_height, pred_width, c = image_list[0].shape
#     net_width, net_height = (network_width(net), network_height(net))
#     img_list = []
#     for custom_image_bgr in image_list:
#         custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
#         custom_image = cv2.resize(
#             custom_image, (net_width, net_height), interpolation=cv2.INTER_NEAREST)
#         custom_image = custom_image.transpose(2, 0, 1)
#         img_list.append(custom_image)

#     arr = np.concatenate(img_list, axis=0)
#     arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
#     data = arr.ctypes.data_as(POINTER(c_float))
#     im = IMAGE(net_width, net_height, c, data)

#     batch_dets = network_predict_batch(net, im, batch_size, pred_width,
#                                                 pred_height, thresh, hier_thresh, None, 0, 0)
#     batch_boxes = []
#     batch_scores = []
#     batch_classes = []
#     for b in range(batch_size):
#         num = batch_dets[b].num
#         dets = batch_dets[b].dets
#         if nms:
#             do_nms_obj(dets, num, meta.classes, nms)
#         boxes = []
#         scores = []
#         classes = []
#         for i in range(num):
#             det = dets[i]
#             score = -1
#             label = None
#             for c in range(det.classes):
#                 p = det.prob[c]
#                 if p > score:
#                     score = p
#                     label = c
#             if score > thresh:
#                 box = det.bbox
#                 left, top, right, bottom = map(int,(box.x - box.w / 2, box.y - box.h / 2,
#                                             box.x + box.w / 2, box.y + box.h / 2))
#                 boxes.append((top, left, bottom, right))
#                 scores.append(score)
#                 classes.append(label)
#                 boxColor = (int(255 * (1 - (score ** 2))), int(255 * (score ** 2)), 0)
#                 cv2.rectangle(image_list[b], (left, top),
#                           (right, bottom), boxColor, 2)
#         cv2.imwrite(os.path.basename(img_samples[b]),image_list[b])

#         batch_boxes.append(boxes)
#         batch_scores.append(scores)
#         batch_classes.append(classes)
#     free_batch_detections(batch_dets, batch_size)
#     return batch_boxes, batch_scores, batch_classes

if __name__ == "__main__":
    print(performDetect(imagePath="samples/test/03009.jpg", thresh= 0.25, 
        configPath = "data/vehicle-detector/yolo-obj.cfg", 
        weightPath = "data/vehicle-detector/yolo-obj_best.weights", 
        metaPath= "data/vehicle-detector/obj.data"))
    #print(performDetect())
    #Uncomment the following line to see batch inference working 
    #print(performBatchDetect())