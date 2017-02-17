"""
Contains some functions that can be used with vgg to get tensors of layers and
compute gram matrices. Assumes that Davi Frossard's vgg is loaded into default
graph within 'vgg' namespace.

File author: Grant Watson
Date: Feb 2017
"""

import tensorflow as tf


def get_layers(layer_names):
    """Get the tensors corresponding to the layer names from VGG in the default
    graph.

    :param layer_names:
        list of strings corresponding to vgg layer names (ex: [conv1_1,
        conv2_1])
    """
    g = tf.get_default_graph()
    layers_names = ['vgg/' + i + ':0' for i in layer_names]
    layers = [g.get_tensor_by_name(name) for name in layers_names]
    return layers


def get_grams(layer_names):
    """Get the style layer tensors from the VGG graph (presumed to be loaded into
    default).

    :param layer_names
        Names of the layers in tf's default graph
    """
    grams = []
    style_layers = get_layers(layer_names)
    for i, layer in enumerate(style_layers):
        b, h, w, c = layer.get_shape().as_list()
        num_elements = h*w*c
        features_matrix = tf.reshape(layer, tf.stack([b, -1, c]))
        gram_matrix = tf.matmul(features_matrix, features_matrix,
                                transpose_a=True)
        gram_matrix = gram_matrix / tf.cast(num_elements, tf.float32)
        grams.append(gram_matrix)
    return grams
