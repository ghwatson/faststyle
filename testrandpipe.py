
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Implementation of image ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# from tensorflow.python.framework import constant_op
# from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
# from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
# from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.ops import gen_image_ops
# from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
import tensorflow as tf
# from tensorflow.python.ops import string_ops
# from tensorflow.python.ops import variables


# ops.NotDifferentiable('RandomCrop')
# # TODO(b/31222613): This op may be differentiable, and there may be
# # latent bugs here.
# ops.NotDifferentiable('RGBToHSV')
# # TODO(b/31222613): This op may be differentiable, and there may be
# # latent bugs here.
# ops.NotDifferentiable('HSVToRGB')
# ops.NotDifferentiable('DrawBoundingBoxes')
# ops.NotDifferentiable('SampleDistortedBoundingBox')
# ops.NotDifferentiable('SampleDistortedBoundingBoxV2')
# # TODO(bsteiner): Implement the gradient function for extract_glimpse
# # TODO(b/31222613): This op may be differentiable, and there may be
# # latent bugs here.
# ops.NotDifferentiable('ExtractGlimpse')
# ops.NotDifferentiable('NonMaxSuppression')
# ops.NotDifferentiable('NonMaxSuppressionV2')


def _assert(cond, ex_type, msg):
  """A polymorphic assert, works with tensors and boolean expressions.

  If `cond` is not a tensor, behave like an ordinary assert statement, except
  that a empty list is returned. If `cond` is a tensor, return a list
  containing a single TensorFlow assert op.

  Args:
    cond: Something evaluates to a boolean value. May be a tensor.
    ex_type: The exception class to use.
    msg: The error message.

  Returns:
    A list, containing at most one assert op.
  """
  if _is_tensor(cond):
    return [control_flow_ops.Assert(cond, [msg])]
  else:
    if not cond:
      raise ex_type(msg)
    else:
      return []


def _is_tensor(x):
  """Returns `True` if `x` is a symbolic tensor-like object.

  Args:
    x: A python object to check.

  Returns:
    `True` if `x` is a `tf.Tensor` or `tf.Variable`, otherwise `False`.
  """
  return isinstance(x, (ops.Tensor, variables.Variable))


def _Check3DImage(image, require_static=True):
  """Assert that we are working with properly shaped image.

  Args:
    image: 3-D Tensor of shape [height, width, channels]
    require_static: If `True`, requires that all dimensions of `image` are
      known and non-zero.

  Raises:
    ValueError: if `image.shape` is not a 3-vector.

  Returns:
    An empty list, if `image` has fully defined dimensions. Otherwise, a list
    containing an assert op is returned.
  """
  try:
    image_shape = image.get_shape().with_rank(3)
  except ValueError:
    raise ValueError("'image' (shape %s) must be three-dimensional." %
                     image.shape)
  if require_static and not image_shape.is_fully_defined():
    raise ValueError("'image' (shape %s) must be fully defined." %
                     image_shape)
  if any(x == 0 for x in image_shape):
    raise ValueError("all dims of 'image.shape' must be > 0: %s" %
                     image_shape)
  if not image_shape.is_fully_defined():
    return [check_ops.assert_positive(array_ops.shape(image),
                                      ["all dims of 'image.shape' "
                                       "must be > 0."])]
  else:
    return []


def fix_image_flip_shape(image, result):
  """Set the shape to 3 dimensional if we don't know anything else.

  Args:
    image: original image size
    result: flipped or transformed image

  Returns:
    An image whose shape is at least None,None,None.
  """

  image_shape = image.get_shape()
  if image_shape == tensor_shape.unknown_shape():
    result.set_shape([None, None, None])
  else:
    result.set_shape(image_shape)
  return result


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


def random_flip_left_right(image, seed=None):
  """Randomly flip an image horizontally (left to right).

  With a 1 in 2 chance, outputs the contents of `image` flipped along the
  second dimension, which is `width`.  Otherwise output the image as-is.

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
                                 lambda: array_ops.reverse(image, [1]),
                                 lambda: image)
  return fix_image_flip_shape(image, result)


#TODO: this shouldn't be here and should be coded more neatly.
def data_augment(example):
    """Performs flipping (with eye's swapped if vertical flip).

    :param example:
    processed_example = [processed_example_l, processed_example_r,
                         processed_example_d, processed_example_occ,
                         processed_example_oof]
    """

    # Stack into a size ten tensor.
    ex_tensor = tf.stack(example, axis=2)
    flipped = tf.image.random_flip_up_down(ex_tensor)

    # Also do a random left-right flip.
    seed = None
    uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
    mirror_cond = math_ops.less(uniform_random, .5)


    result = control_flow_ops.cond(mirror_cond,
                                 lambda: array_ops.reverse(image, [1]),
                                 lambda: image)
    return fix_image_flip_shape(image, result)

    example = tf.unstack(flipped, axis=2)
    return example


def random_flip_left_right_stereo(image_l, image_r, seed=None):
  """Randomly flip an image horizontally (left to right).

  With a 1 in 2 chance, outputs the contents of `image` flipped along the
  second dimension, which is `width`.  Otherwise output the image as-is.

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
                                 lambda: array_ops.reverse(image, [1]),
                                 lambda: image)
  return fix_image_flip_shape(image, result)



def flip_left_right(image):
  """Flip an image horizontally (left to right).

  Outputs the contents of `image` flipped along the second dimension, which is
  `width`.

  See also `reverse()`.

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`

  Returns:
    A 3-D tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  image = ops.convert_to_tensor(image, name='image')
  image = control_flow_ops.with_dependencies(
      _Check3DImage(image, require_static=False), image)
  return fix_image_flip_shape(image, array_ops.reverse(image, [1]))


def flip_up_down(image):
  """Flip an image vertically (upside down).

  Outputs the contents of `image` flipped along the first dimension, which is
  `height`.

  See also `reverse()`.

  Args:
    image: A 3-D tensor of shape `[height, width, channels].`

  Returns:
    A 3-D tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
  image = ops.convert_to_tensor(image, name='image')
  image = control_flow_ops.with_dependencies(
      _Check3DImage(image, require_static=False), image)
  return fix_image_flip_shape(image, array_ops.reverse(image, [0]))

