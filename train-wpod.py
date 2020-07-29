import os
import math
import functools
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
import cv2

from src.label import readShapes
from src.keras_utils import save_model, load_model
from src.sampler import augment_sample, labels2output_map
from src.loss import loss

data_dir_list = [
    "_train_wpod/data/data_kor_v1_done/*.txt",
    "_train_wpod/data/data_oid_v1_done/*.txt"
]
#data_dir_list = ["_train_wpod/data/samples/*.txt"]
model_path = "_train_wpod/data/data/lp-detector/wpod-net_update1.h5"
output_dir = "_train_wpod/output"
input_dim = 208

BATCH_SIZE = 32
NUM_ITERATIONS = 3000000
STEPS_PER_EPOCH = 1000
EPOCHS = NUM_ITERATIONS // STEPS_PER_EPOCH
LEARNING_RATE = 0.05

def load_network(model_path, input_dim):
    model = load_model(model_path)
    input_shape = (input_dim, input_dim, 3)

    # Fixed input size for training
    inputs = keras.layers.Input(shape=(input_dim, input_dim, 3))
    outputs = model(inputs)

    output_shape = tuple([s for s in outputs.shape[1:]])
    output_dim = output_shape[1]
    model_stride = int(input_dim / output_dim)

    assert input_dim % output_dim == 0, \
     'The output resolution must be divisible by the input resolution'

    assert model_stride == 2**4, \
     'Make sure your model generates a feature map with resolution ' \
     '16x smaller than the input'

    return model, model_stride, input_shape, output_shape

def load_labels(label_file):
    label_file = Path(bytes.decode(label_file.numpy()))
    image_file = label_file.parent / (label_file.stem + ".jpg")
    shapes = readShapes(str(label_file))
    # TODO flatten to multiple shape 
    return tf.constant(str(image_file)), shapes[0].pts # pts.shape=(2,4)

def process_data_item(image_path, pts, input_dim, model_stride):
    img = cv2.imread(bytes.decode(image_path.numpy()))
    xx, lp_label, pts = augment_sample(img, pts.numpy(), float(input_dim))
    yy = labels2output_map(lp_label, pts, input_dim, model_stride)
    return xx, yy

def batch_from_dataset(input_dim, model_stride):
    ds = tf.data.Dataset.list_files(data_dir_list)

    # load labels
    def _load_labels(label_path):
        image_path, pts = tf.py_function(load_labels, [label_path], [tf.string, tf.float32])
        return image_path, pts
    ds = ds.map(_load_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # shuffle whole dataset
    num_labels = tf.data.experimental.cardinality(ds).numpy()
    ds = ds.cache().shuffle(num_labels)

    # load image
    def _process_data_item(image_path, pts):
        x_data, y_data = tf.py_function(
            process_data_item,
            [image_path, pts, input_dim, model_stride], 
            [tf.float32, tf.float32])
        return x_data, y_data
    ds = ds.map(_process_data_item, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # repeat, batch
    ds = ds.repeat(-1).batch(BATCH_SIZE)

    # prefetch
    ds = ds.prefetch(BATCH_SIZE)
    return ds

def step_decay(epoch):
    initial_lr = LEARNING_RATE
    min_lr = 0.0001
    drop = 0.5
    epochs_drop = 8.0
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    lr = max(lr, min_lr)
    return lr

if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    print('* Number of devices: ', strategy.num_replicas_in_sync)
    with strategy.scope():
        model, model_stride, xshape, yshape = load_network(model_path, input_dim)
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
        model.summary()

    callbacks = [
        ModelCheckpoint(
            os.path.join(output_dir, 'weights-{epoch:03d}.h5'), 
            save_weights_only=True, save_best_only=False),
        TensorBoard(
            log_dir=output_dir, update_freq=100),
        LearningRateScheduler(step_decay)]

    ds_train = batch_from_dataset(input_dim, model_stride)
    model.fit(ds_train, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, 
        verbose=1, callbacks=callbacks)










