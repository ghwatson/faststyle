"""
This file contains the different loss functions.

File author: Grant Watson
Date: Feb 2017
"""

import tensorflow as tf
import numpy as np


def content_loss(content_layers, target_content_layers,
                 content_weights):
    """Defines the content loss function.

    :param content_layers
        List of tensors for layers derived from training graph.
    :param target_content_layers
        List of placeholders to be filled with content layer data.
    :param content_weights
        List of floats to be used as weights for content layers.
    """
    assert(len(target_content_layers) == len(content_layers))
    num_content_layers = len(target_content_layers)

    # Content loss
    content_losses = []
    for i in xrange(num_content_layers):
        content_layer = content_layers[i]
        target_content_layer = target_content_layers[i]
        content_weight = content_weights[i]
        loss = tf.reduce_sum(tf.squared_difference(content_layer,
                                                   target_content_layer))
        loss = content_weight * loss
        _, h, w, c = content_layer.get_shape().as_list()
        num_elements = h * w * c
        loss = loss / tf.cast(num_elements, tf.float32)
        content_losses.append(loss)
    content_loss = tf.add_n(content_losses, name='content_loss')
    return content_loss


def style_loss(grams, target_grams, style_weights):
    """Defines the style loss function.

    :param grams
        List of tensors for Gram matrices derived from training graph.
    :param target_grams
        List of numpy arrays for Gram matrices precomputed from style image.
    :param style_weights
        List of floats to be used as weights for style layers.
    """
    assert(len(grams) == len(target_grams))
    num_style_layers = len(target_grams)

    # Style loss
    style_losses = []
    for i in xrange(num_style_layers):
        gram, target_gram = grams[i], target_grams[i]
        style_weight = style_weights[i]
        _, c1, c2 = gram.get_shape().as_list()
        size = c1*c2
        loss = tf.reduce_sum(tf.square(gram - tf.constant(target_gram)))
        loss = style_weight * loss / size
        style_losses.append(loss)
    style_loss = tf.add_n(style_losses, name='style_loss')
    return style_loss


def tv_loss(X):
    """Creates 2d TV loss using X as the input tensor. Acts on different colour
    channels individually, and uses convolution as a means of calculating the
    differences.

    :param X:
        4D Tensor
    """
    # These filters for the convolution will take the differences across the
    # spatial dimensions. Constructing these on paper has to be done carefully,
    # but can be easily understood  when one realizes that the sub-3x3 arrays
    # should have no mixing terms as the RGB channels should not interact
    # within this convolution. Thus, the 2 3x3 subarrays are identity and
    # -1*identity. The filters should look like:
    # v_filter = [ [(3x3)], [(3x3)] ]
    # h_filter = [ [(3x3), (3x3)] ]
    ident = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    v_array = np.array([[ident], [-1*ident]])
    h_array = np.array([[ident, -1*ident]])
    v_filter = tf.constant(v_array, tf.float32)
    h_filter = tf.constant(h_array, tf.float32)

    vdiff = tf.nn.conv2d(X, v_filter, strides=[1, 1, 1, 1], padding='VALID')
    hdiff = tf.nn.conv2d(X, h_filter, strides=[1, 1, 1, 1], padding='VALID')

    loss = tf.reduce_sum(tf.square(hdiff)) + tf.reduce_sum(tf.square(vdiff))

    return loss
