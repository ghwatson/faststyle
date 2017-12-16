"""
Contains some functions that can be used with vgg to get tensors of layers and
compute gram matrices.  Also contains some wrappers around OpenCV's image I/O
functions.

File author: Grant Watson
Date: Feb 2017
"""

import tensorflow as tf
import cv2


def imread(path):
    """Wrapper around cv2.imread. Switches channels to keep everything in RGB.

    :param path:
        String indicating path to image.
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imresize(img, scale):
    """Depending on if we scale the image up or down, we use an interpolation
    technique as per OpenCV recommendation.

    :param img:
        3D numpy array of image.
    :param scale:
        float to scale image by in both axes.
    """
    if scale > 1.0:  # use cubic interpolation for upscale.
        img = cv2.resize(img, None, interpolation=cv2.INTER_CUBIC,
                         fx=scale, fy=scale)
    elif scale < 1.0:  # area relation sampling for downscale.
        img = cv2.resize(img, None, interpolation=cv2.INTER_AREA,
                         fx=scale, fy=scale)
    return img


def imwrite(path, img):
    """Wrapper around cv2.imwrite. Switches it to RGB input convention.

    :param path:
        String indicating path to save image to.
    :param img:
        3D RGB numpy array of image.
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def get_layers(layer_names):
    """Get tensors from default graph by name.

    :param layer_names:
        list of strings corresponding to names of tensors we want to extract.
    """
    g = tf.get_default_graph()
    layers = [g.get_tensor_by_name(name) for name in layer_names]
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
