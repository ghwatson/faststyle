import tensorflow as tf
import numpy as np
from libs import vgg16
from im_transf_net import create_net
import datapipe
import datapipe_stereo
import os
import argparse
import utils
import losses


# TODO: debugging
from PIL import Image
import cv2
from tensorflow.contrib.resampler import resampler



train_stereo_dir = '/mnt/77D2D4A40B3D7BC7/tf_stereo'
batch_stereo_size = 2
# TODO: see why aspect ratio introduces negative occlusion values.
preprocess_stereo_size = [256,512]
# preprocess_stereo_size = [436,1024]
# preprocess_stereo_size = [218,512]
preprocess_stereo_size = [220,512]

n_epochs_stereo = 2
num_pipe_buffer = 5
files_stereo = tf.train.match_filenames_once(train_stereo_dir + '/train-*')
with tf.variable_scope('input_pipe'), tf.device('/cpu:0'):
    batch_stereo_op = datapipe_stereo.batcher(files_stereo,
                                              batch_stereo_size,
                                              preprocess_stereo_size,
                                              n_epochs_stereo,
                                              num_pipe_buffer)


def random_flip_up_down(image, seed=None):
  """Randomly flips an image vertically (upside down).
  With a 1 in 2 chance, outputs the contents of `image` flipped along the first
  dimension, which is `height`.  Otherwise output the image as-is.
  Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    seed: A Python integer. Used to create a random seed. See
      @{tf.set_random_seed}
      for behavior.
  Returns:
    A 3-D tensor of the same type and shape as `image`.
  Raises:
    ValueError: if the shape of `image` not supported.
  """
  image = ops.convert_to_tensor(image, name='image')
  image = control_flow_ops.with_dependencies(
      _Check3DImage(image, require_static=False), image)
  uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
  mirror_cond = math_ops.less(uniform_random, .5)
  result = control_flow_ops.cond(mirror_cond,
                                 lambda: array_ops.reverse(image, [0]),
                                 lambda: image)
  return fix_image_flip_shape(image, result)



init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)
    print 'yo'
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print 'yo2'
    batch = sess.run(batch_stereo_op)
    batch_l, batch_r, batch_d, batch_occ, batch_oof = batch

    print batch_occ.shape
    print np.unique(batch_occ)

    viz = np.concatenate([batch_l, batch_r], axis=1)
    viz_bmp = np.concatenate([batch_d[:,:,:,0:1], batch_occ, batch_oof], axis=1)
    utils.imwrite('./viz.png', viz[0])
    utils.imwrite_greyscale('./viz_bmp.png',viz_bmp[0])
    img = Image.fromarray(viz[0].astype('uint8'))
    img.show()
    print (viz_bmp[0].astype('uint8')).dtype
    print (viz_bmp[0].astype('uint8')).shape
    img = Image.fromarray(np.squeeze(viz_bmp[0]))
    img.show()
