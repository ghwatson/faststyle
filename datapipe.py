"""
This file is used for construction of the data input pipeline. It takes care of
batching and preprocessing, and can be used to repeatedly draw a fresh batch
for use in training.
"""

import tensorflow as tf


def preprocessing(image, resize_shape):
    image = tf.image.resize_images(image, size=resize_shape)

    return image


def read_my_file_format(filename_queue, resize_shape):
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


def batcher(filenames, batch_size, resize_shape, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    example = read_my_file_format(filename_queue, resize_shape)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch = tf.train.shuffle_batch(
        [example], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch

if __name__ == '__main__':
    # Simple execution test (specific to my setup)
    # TODO: replace with something more informative + generic.

    train_dir = '/home/ghwatson/workspace/faststyle/mockshards/'
    mscoco_shape = [256, 256]

    files = tf.train.match_filenames_once(train_dir + 'train-*')

    batch_op = batcher(files, 4, mscoco_shape, 2)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    yo = 1
    with tf.Session() as sess:

        # Initialize the pipeline
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                out = sess.run(batch_op)
                yo += 1
                print yo
        except tf.errors.OutOfRangeError:
            print('Done training.')
        finally:
            coord.request_stop()

        coord.join(threads)

        print out.shape
        print out[1, :, :, 0]
