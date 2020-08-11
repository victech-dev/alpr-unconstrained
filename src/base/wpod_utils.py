import numpy as np
import cv2
import time
from os.path import splitext
import tensorflow as tf
from tensorflow.keras.models import model_from_json

from base.label import Label
from base.utils import get_wh, nms, show, draw_label
from base.projection_utils import get_rect_ptsh, find_T_matrix

wpod_input_dim = 208 # same as training input dim
wpod_net_stride = 2**4 # decided by pooling layer count
ocr_unwarp_margin = 8.
ocr_input_wh = (288, 96)

def save_model(model, path, verbose=0):
    path = splitext(path)[0]
    model_json = model.to_json()
    with open('%s.json' % path, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('%s.h5' % path)
    if verbose: print('Saved to %s' % path)

def load_model(path, custom_objects={}, verbose=0):
    path = splitext(path)[0]
    with open('%s.json' % path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json, custom_objects=custom_objects)
    model.load_weights('%s.h5' % path)
    if verbose: print('Loaded from %s' % path)
    return model

def reconstruct(image, infer_image_wh, net_stride, infer_ret, out_wh, out_margin, threshold=.9):
    side = ((208. + 40.) / 2.) / net_stride  # 7.75

    confidences = infer_ret[..., 0] 
    y, x = np.unravel_index(confidences.argmax(), confidences.shape)

    out_label = None
    out_image = None
    if confidences[y, x] > threshold:
        affine = infer_ret[y, x, 2:]
        affine = np.reshape(affine, (2, 3))
        affine[0, 0] = max(affine[0, 0], 0.) # prevent flip
        affine[1, 1] = max(affine[1, 1], 0.) # prevent flip

        vx = vy = 0.5 
        base = np.matrix(
            [[-vx, -vy, 1.],
             [ vx, -vy, 1.],
             [ vx,  vy, 1.],
             [-vx,  vy, 1.]]).T

        pts_mn_centered = np.array(affine * base) * side
        mn = np.array([float(x) + .5, float(y) + .5])
        pts = pts_mn_centered + mn.reshape((2, 1))
        # infer_image may have black letterbox
        out_label = pts / (infer_image_wh / net_stride).reshape((2, 1)).astype(np.float32)

        # image_to_show = image.copy()
        # draw_label(image_to_show, out_label)
        # show(image_to_show)

        def _get_rect_pts(l, t, r, b):
            return np.array([[l, t], [r, t], [r, b], [l, b]]).astype(np.float32)

        pts_on_image = (out_label * get_wh(image.shape).reshape((2, 1))).T.astype(np.float32)
        pts_on_out = _get_rect_pts(out_margin, out_margin, out_wh[0] - out_margin, out_wh[1] - out_margin)
        trasform_mat = cv2.getPerspectiveTransform(pts_on_image, pts_on_out)
        out_image = cv2.warpPerspective(image, trasform_mat, out_wh, borderValue=.0)
        
    return out_label, out_image

# cnn으로만 구성되어 있어서 input_dim을 고정할 필요는 없지만
# train 과정에서 사용된 input으로 고정하여
# 새로운 input_dim의 그래프를 생성하는 tf의 cost를 없애야 한다
def detect_lp(wpod_net_fn, image, input_dim, net_stride, out_wh, threshold):
    scale_factor = float(input_dim) / np.amax(image.shape[:2])
    w, h = (np.array(image.shape[1::-1]) * scale_factor).astype(int).tolist()

    infer_image = np.zeros((input_dim, input_dim, 3), dtype=np.uint8)
    infer_image[:h, :w, :] = cv2.resize(image, (w, h))
    infer_image = np.expand_dims(infer_image, axis=0)
    infer_image = infer_image.astype(np.float32) / 255.0
    infer_image = tf.convert_to_tensor(infer_image)

    infer_func = wpod_net_fn.get_concrete_function(infer_image)
    infer_ret = infer_func(infer_image).numpy()
    infer_ret = np.squeeze(infer_ret) # infer_ret.shape=(h/net_stride, w/net_stride, 8)

    out_label, out_image = reconstruct(
        image, np.array([w, h]), net_stride, infer_ret,
        out_wh, ocr_unwarp_margin, threshold)
    confidence = np.max(infer_ret[:,:,:1])
    return out_label, out_image, confidence

def detect_wpod(wpod_net_fn, image, threshold):
    out_label, out_image, confidence = detect_lp(
        wpod_net_fn, image, wpod_input_dim, wpod_net_stride,
        ocr_input_wh, threshold)
    # out_label : np.array, shape=(2, 4)
    return out_label, out_image, confidence

def load_wpod(wpod_net_path):
    wpod_net = load_model(wpod_net_path)

    # for tf 2.0
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32)
    ])
    def wpod_net_fn(img):
        return wpod_net.call(img)
    
    return wpod_net, wpod_net_fn