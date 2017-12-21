# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# MODIFIED:
#   -Dec 2017 Further modifications to work with Sintel dataset structure.
#   -Feb 2017 by Grant Watson for independent use.

# TODO: add in the sintel_io.py file. provides means by which to access the
# disparity data.
"""
Editor's Note: As mentioned above, I edited this. In particular, I made it to
work with an unlabelled set of images. Original file here:
    https://github.com/tensorflow/models/blob/master/inception/inception/data/build-image-data.py

This works with the Sintel stereo dataset.

--------------------

Converts image data to TFRecords file format with Example protos.

The image data set is expected to reside in JPEG files located in a single,
flat directory.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files

  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-00127-of-01024

where we have selected 1024 shards for the data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:

  image/encoded_l: string containing JPEG encoded left image in RGB colorspace
  image/encoded_r: string containing JPEG encoded right image in RGB colorspace
  image/encoded_d: string containing JPEG encoded disparity img in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always'JPEG'
  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('train_directory', '/tmp/',
                           'Training data directory')
tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')
tf.app.flags.DEFINE_integer('train_shards', 2,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 2,
                            'Number of threads to preprocess the images.')


FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer_l, image_buffer_r,
                        image_buffer_d, image_buffer_occ,
                        image_buffer_oof, height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer_l: string, JPEG encoding of RGB image
    image_buffer_r: string, JPEG encoding of RGB image
    image_buffer_d: string, JPEG encoding of RGB image
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
      'image/channels': _int64_feature(channels),
      'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
      'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
      'image/encoded_l': _bytes_feature(tf.compat.as_bytes(image_buffer_l)),
      'image/encoded_r': _bytes_feature(tf.compat.as_bytes(image_buffer_r)),
      'image/encoded_d': _bytes_feature(tf.compat.as_bytes(image_buffer_d)),
      'image/encoded_occ': _bytes_feature(tf.compat.as_bytes(image_buffer_occ)),
      'image/encoded_oof': _bytes_feature(tf.compat.as_bytes(image_buffer_oof))}))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._png_data, channels=3)
    # self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    # self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    # self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  # def png_to_jpeg(self, image_data):
    # return self._sess.run(self._png_to_jpeg,
                          # feed_dict={self._png_data: image_data})

  def decode_png(self, image_data):
    image = self._sess.run(self._decode_png,
                           feed_dict={self._png_data: image_data})
    assert image.shape[2] == 3
    return image
  # def decode_jpeg(self, image_data):
    # image = self._sess.run(self._decode_jpeg,
                           # feed_dict={self._decode_jpeg_data: image_data})

    # assert image.shape[2] == 3
    # return image

class ImageCoderGs(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._png_data, channels=1)
    # self._png_to_jpeg = tf.image.encode_jpeg(image, format='grayscale', quality=100)


    # Initializes function that decodes greyscale JPEG data.
    # self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    # self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=1)

  # def png_to_jpeg(self, image_data):
    # # returns output of _png_to_jpeg given image data.
    # return self._sess.run(self._png_to_jpeg,
                          # feed_dict={self._png_data: image_data})
  
  def decode_png(self, image_data):
    image = self._sess.run(self._decode_png,
                          feed_dict={self._png_data: image_data})
    assert image.shape[2] == 1
    return image

  # def decode_jpeg(self, image_data):
    # image = self._sess.run(self._decode_jpeg,
                           # feed_dict={self._decode_jpeg_data: image_data})
    # assert image.shape[2] == 1
    # return image

def _is_png(filename):
  """Determine if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
  return '.png' in filename


def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'r') as f:
    image_data = f.read()

  # Convert any PNG to JPEG's for consistency.
  # if _is_png(filename):
    # print('Converting PNG to JPEG for %s' % filename)
    # image_data = coder.png_to_jpeg(image_data)


  # Decode the RGB JPEG.
  # image = coder.decode_jpeg(image_data)
  image = coder.decode_png(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width

def _process_image_gs(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'r') as f:
    image_data = f.read()

  # Convert any PNG to JPEG's for consistency.
  # if _is_png(filename):
    # print('Converting PNG to JPEG for %s' % filename)
    # image_data = coder.png_to_jpeg(image_data)

  # Decode the GS PNG.
  # image = coder.decode_jpeg(image_data)
  image = coder.decode_png(image_data)

  # Check that image converted to GS
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 1

  return image_data, height, width


def _process_image_files_batch(coder, coder_gs, thread_index, ranges, name, filenames,
                               num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads 
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename_l, filename_r, filename_d, filename_occ, filename_oof = \
              filenames[i]
      basename = os.path.splitext(filename_l)[0]
      # scene = scenes[i]

      image_buffer_l, height, width = _process_image(filename_l, coder)
      image_buffer_r,_,_ = _process_image(filename_r, coder)
      image_buffer_d,_,_ = _process_image(filename_d, coder)
      # TODO: pass in greyscale encoders here.
      # 
      image_buffer_occ,_,_ = _process_image_gs(filename_occ, coder_gs)
      image_buffer_oof,_,_ = _process_image_gs(filename_oof, coder_gs)

      # TODO: here we will want to feed in left, right, disparity buffers.
      example = _convert_to_example(basename, image_buffer_l, image_buffer_r,
                                    image_buffer_d, image_buffer_occ,
                                    image_buffer_oof, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, num_shards):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    num_shards: integer number of shards for this data set.
  """

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()
  coder_gs = ImageCoderGs()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, coder_gs, thread_index, ranges, name, filenames, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(data_dir, scenes):
  """Build a list of all images files in the data set.

  Args:
    data_dir: string, path to the root directory of images.  Assumes the
              directory is flat with no subdirectories.

  Returns:
    filenames: list of strings; each string is a path to an image file.
  """
  print('Determining list of input files from %s.' % data_dir)

  # List of JPEG files.
  filenames = []
  for scene in scenes:
      file_l_path = '%s/clean_left/%s/*' % (data_dir, scene)
      filenames_l = tf.gfile.Glob(file_l_path)
      file_r_path = '%s/clean_right/%s/*' % (data_dir, scene)
      filenames_r = tf.gfile.Glob(file_r_path)
      file_d_path = '%s/disparities/%s/*' % (data_dir, scene)
      filenames_d = tf.gfile.Glob(file_d_path)
      file_occ_path = '%s/occlusions/%s/*' % (data_dir, scene)
      filenames_occ = tf.gfile.Glob(file_occ_path)
      file_oof_path = '%s/outofframe/%s/*' % (data_dir, scene)
      filenames_oof = tf.gfile.Glob(file_oof_path)
      to_add = zip(filenames_l,
                   filenames_r,
                   filenames_d,
                   filenames_occ,
                   filenames_oof)
      filenames.extend(to_add)
  # filenames = tf.gfile.Glob(data_dir + '/*')

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  # Note (Grant): Not sure if this is needed anymore. Left in to be safe. A
  # little extra shuffling of unlabelled data never hurt anyone!
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]

  print('Found %d JPEG files inside %s.' %
        (len(filenames), data_dir))
  return filenames


def _process_dataset(name, directory, num_shards):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
  """
  # Get the scene directory names.
  scenes = os.listdir(directory + '/clean_left/')


  filenames = _find_image_files(directory, scenes)
  _process_image_files(name, filenames, num_shards)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  print('Saving results to %s' % FLAGS.output_directory)

  # Run it!
  _process_dataset('train', FLAGS.train_directory,
                   FLAGS.train_shards)

if __name__ == '__main__':
  tf.app.run()
