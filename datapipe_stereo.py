"""
This file is used for construction of the data input pipeline. It takes care of
batching and preprocessing, and can be used to repeatedly draw a fresh batch
for use in training. It utilizes TFRecords format, so data must be converted to
this beforehand. tfrecords_writer.py handles this.

File author: Grant Watson
Date: Jan 2017
"""

import tensorflow as tf

# TODO: better modularize all this.
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
# from testrandpipe import data_augment

# TODO: for disparity image, do the shift + scaling as defined in sintel_io.py.
#       then convert it into the appropriate warp map.
# TODO: match the interface of this with the tfrecords_writer_stereo.py


def data_augment(example):
    """Performs flipping (with eye's swapped if vertical flip).

    :param example:
    processed_example = [processed_example_l, processed_example_r,
                         processed_example_d, processed_example_occ,
                         processed_example_oof]
    """
    ex_tensor = tf.stack(example, axis=2)
    flipped = tf.image.random_flip_up_down(ex_tensor)
    example = tf.unstack(flipped, axis=2)

    # # Also do a random left-right flip.
    # seed = None
    # uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
    # mirror_cond = math_ops.less(uniform_random, .5)

    return example

def preprocessing(image, resize_shape):
    """Simply resizes the image.

    :param image:
        image tensor
    :param resize_shape:
        list of dimensions
    """
    if resize_shape is None:
        return image
    else:
        image = tf.image.resize_images(image, size=resize_shape, method=0)
        return image


# Global tensor used in preprocessing disparity, convention defined by Sintel.


def preprocessing_disparity(image, resize_shape):
    """Resizes + rescales the disparity along with the resizing.

    :param image:
        image tensor
    :param resize_shape:
        list of dimensions
    """
    orig_shape = [436,1024]
    rescale = tf.constant([4.0, 1./2**6, 1./2**14])
    image = tf.reduce_sum(tf.to_float(image)*rescale, 2, keep_dims=True)
    if resize_shape is not None:
        # TODO: this is hacked in for the time being.
        scale = resize_shape[1]*1./1024 # ex: 0.5 = 218/436
        image = tf.image.resize_images(image, size=resize_shape, method=0)
        image = scale*image

    # Transform into a warp
    # TODO: creating duplicate ops here. Make more efficient if needed.
    # perhaps just precompute.
    # TODO: dont know dynamic size...precompute these offline?
    image = tf.concat([image, tf.zeros(image.shape)], axis=2)
    h, w, _ = image.get_shape().as_list()
    X, Y = tf.meshgrid(range(w), range(h))
    warp_y = tf.to_float(Y) - image[:, :, 1]
    warp_x = tf.to_float(X) - image[:, :, 0]
    warp = tf.stack([warp_x, warp_y], axis=2)

    return warp


def preprocessing_masks(image, resize_shape):
    """Makes masks binary. Also resizes.

    :param image:
        image tensor
    :param resize_shape:
        list of dimensions
    """
    #image = tf.to_float(image)
    if resize_shape is None:
        return image
    else:
        image = tf.image.resize_images(image, size=resize_shape, method=1)
        return image


def read_my_file_format(filename_queue, resize_shape=None):
    """Sets up part of the pipeline that takes elements from the filename queue
    and turns it into a tf.Tensor of a batch of images.

    :param filename_queue:
        tf.train.string_input_producer object
    :param resize_shape:
        2 element list defining the shape to resize images to.
    """
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example, features={
            'image/encoded_l': tf.FixedLenFeature([], tf.string),
            'image/encoded_r': tf.FixedLenFeature([], tf.string),
            'image/encoded_d': tf.FixedLenFeature([], tf.string),
            'image/encoded_occ': tf.FixedLenFeature([], tf.string),
            'image/encoded_oof': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64)})
    example_l = tf.image.decode_png(features['image/encoded_l'], 3)
    example_r = tf.image.decode_png(features['image/encoded_r'], 3)
    example_d = tf.image.decode_png(features['image/encoded_d'], 3)
    example_occ = tf.image.decode_png(features['image/encoded_occ'], 1)
    example_oof = tf.image.decode_png(features['image/encoded_oof'], 1)
    processed_example_l = preprocessing(example_l, resize_shape)
    processed_example_r = preprocessing(example_r, resize_shape)
    processed_example_d = preprocessing_disparity(example_d, resize_shape)
    processed_example_occ = preprocessing_masks(example_occ, resize_shape)
    processed_example_oof = preprocessing_masks(example_oof, resize_shape)
    processed_example = [processed_example_l, processed_example_r,
                         processed_example_d, processed_example_occ,
                         processed_example_oof]

    processed_example = data_augment(processed_example)

    return processed_example


def batcher(filenames, batch_size, resize_shape=None, num_epochs=None,
            min_after_dequeue=4000):
    """Creates the batching part of the pipeline.

    :param filenames:
        list of filenames
    :param batch_size:
        size of batches that get output upon each access.
    :param resize_shape:
        for preprocessing. What to resize images to.
    :param num_epochs:
        number of epochs that define end of training set.
    :param min_after_dequeue:
        min_after_dequeue defines how big a buffer we will randomly sample
        from -- bigger means better shuffling but slower start up and more
        memory used.
        capacity must be larger than min_after_dequeue and the amount larger
        determines the maximum we will prefetch.  Recommendation:
        min_after_dequeue + (num_threads + a small safety margin) * batch_size
    """
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    example = read_my_file_format(filename_queue, resize_shape)
    capacity = min_after_dequeue + 3 * batch_size
    example_batch = tf.train.shuffle_batch(
        example, batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch
