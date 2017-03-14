"""
Contains some functions that can be used with vgg to get tensors of layers and
compute gram matrices.  Also contains some wrappers around OpenCV's image I/O
functions.

File author: Grant Watson
Date: Feb 2017
"""

import tensorflow as tf
import numpy as np
import cv2

# TODO: merge the 2 imresize util functions?


def imread(path, mode=-1):
    """Wrapper around cv2.imread. Switches channels to keep everything in RGB.

    :param path:
        String indicating path to image.
    :param mode:
        -1 for colour, 0 for grayscale, 1 for unchanged (for example, keep
        alpha channels)
    """
    img = cv2.imread(path, mode)
    if mode == -1:
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
        out = cv2.resize(img, None, interpolation=cv2.INTER_CUBIC,
                         fx=scale, fy=scale)
        return out
    elif scale < 1.0:  # area relation sampling for downscale.
        out = cv2.resize(img, None, interpolation=cv2.INTER_AREA,
                         fx=scale, fy=scale)
        return out
    else:
        return img


def imresize_shape(img, shape):
    """Depending on if we increase/decrease image area, we use an interpolation
    technique as per OpenCV recommendation.

    :param img:
        3D numpy array of image.
    :param shape:
        tuple to shape image to.
    """
    img_area = img.shape[0]*img.shape[1]
    area = shape[0]*shape[1]
    shape = (shape[1], shape[0])  # TODO: document this properly.

    if area > img_area:  # use cubic interpolation for upscale.
        out = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)
        return out
    elif area < img_area:  # area relation sampling for downscale.
        out = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
        return out
    else:
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


def get_grams(layer_names, guidance_channels=None):
    """Get the style layer tensors from the graph using the given layer_names.
    The option to pass in a region's guidance channels for the set of layers
    is available. Spatial control described in:
               https://arxiv.org/abs/1611.07865

    :param layer_names:
        Names of NxHxWxC tensors in tf's default graph
    :param guidance_channels:
        List of HxW arrays that represent the layers' guidance channels. This
        can be obtained from utils.propagate_mask.
    """
    if guidance_channels is not None:
        assert(len(layer_names) == len(guidance_channels))
    grams = []
    style_layers = get_layers(layer_names)
    for i, layer in enumerate(style_layers):
        b, h, w, c = layer.get_shape().as_list()
        num_elements = h*w*c

        # Perform spatial guidance, if applicable.
        if guidance_channels is not None:
            layer_guidance_channels = guidance_channels[i]
            layer = guide_layer(layer, layer_guidance_channels)

        features_matrix = tf.reshape(layer, tf.stack([b, -1, c]))

        gram_matrix = tf.matmul(features_matrix, features_matrix,
                                transpose_a=True)
        gram_matrix = gram_matrix / tf.cast(num_elements, tf.float32)
        grams.append(gram_matrix)
    return grams


def guide_layer(layer, layer_guidance_channels):
    """Applies guidance channels to a layer. Renormalizes the layer as well to
    account for the effect of the guidance channel.

    :param layer:
        tf.Tensor of dimension NxHxWxC.
    :param layer_guidance_channels:
        A layer's guidance channels, obtained from utils.propagate_mask.
    """
    layer_guidance_channels = layer_guidance_channels[np.newaxis, :, :]
    layer_guidance_channels = layer_guidance_channels[:, :, :, np.newaxis]
    # Guide the layer while preserving its norm.
    pre_norm = l2_normalization(layer, [1, 2])
    layer = layer_guidance_channels*layer
    post_norm = l2_normalization(layer, [1, 2])
    layer = (layer / post_norm) * pre_norm
    return layer


def l2_normalization(x, dim, epsilon=1e-12, name=None):
    square_sum = tf.reduce_sum(tf.square(x), dim, keep_dims=True)
    x_norm = tf.sqrt(tf.maximum(square_sum, epsilon))
    return x_norm


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def propagate_mask(mask, layer_names, strategy='Simple'):
    """Takes a mask and from it creates masks for each layer of style
    features.  Convolutional neural networks lose spatial precision, so
    there are multiple strategies to do this. Reference:
               https://arxiv.org/abs/1611.07865

    :param mask:
        HxW Numpy array. Mask that gets propagated so that it can be used at
        other layers.
    :param layer_names:
        A list of names of tensors. Represents the sequence of layers through
        which the image mask gets propagated.
    :param strategy:
        String = Simple, All, or Inside. Strategies as detailed in the above
        reference's supplement.
    """
    # Get the spatial dimensions of each layer.
    layers = get_layers(layer_names)
    layer_sizes = [tuple(layer.get_shape().as_list()[1:3]) for layer in layers]

    # Propagate
    guidance_channels = []
    if strategy is 'Simple':
        # We simply downsample without considering the receptive fields.
        for size in layer_sizes:
            layer_guidance_channels = imresize_shape(mask, size)
            guidance_channels.append(layer_guidance_channels)
    elif strategy is 'All':
        raise ValueError('Not yet implemented.')
    elif strategy is 'Inside':
        raise ValueError('Not yet implemented.')
    else:
        raise ValueError("""{} is not a valid strategy name. Options are:
                         Simple, Inside, or All.""")

    return guidance_channels
