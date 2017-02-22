"""
This file is used for construction of the data input pipeline. It takes care of
batching and preprocessing, and can be used to repeatedly draw a fresh batch
for use in training. It utilizes TFRecords format, so data must be converted to
this beforehand. tfrecords_writer.py handles this.

File author: Grant Watson
Date: Jan 2017
"""

import tensorflow as tf


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
        image = tf.image.resize_images(image, size=resize_shape, method=2)
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
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64)})
    example = tf.image.decode_jpeg(features['image/encoded'], 3)
    processed_example = preprocessing(example, resize_shape)
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
        [example], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch
