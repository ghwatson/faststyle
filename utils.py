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
    is available. These channels are used for spatial control, described in:
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
        _, h, w, c = layer.get_shape().as_list()
        num_elements = h*w*c

        # Perform spatial guidance, if applicable.
        if guidance_channels is not None:
            layer_guidance_channels = guidance_channels[i]
            layer = guide_layer(layer, layer_guidance_channels)

        features_matrix = tf.reshape(layer, tf.stack([-1, h*w, c]))

        gram_matrix = tf.matmul(features_matrix, features_matrix,
                                transpose_a=True)
        gram_matrix = gram_matrix / tf.cast(num_elements, tf.float32)
        grams.append(gram_matrix)
    return grams


def guide_layer(layer, layer_guidance_channels):
    """Applies guidance channels to a layer. If desired, this is a good spot to
    encode normalization in the spatial control strategy.

    :param layer:
        tf.Tensor of dimension NxHxWxC.
    :param layer_guidance_channels:
        A layer's guidance channels, obtained from utils.propagate_mask.
    """
    layer = layer_guidance_channels*layer
    return layer

def propagate_mask(mask, layer_names, strategy='Simple', vggnet=None,
                   batchsize=2):
    """Takes a mask and from it creates masks for each layer of style
    features.  Convolutional neural networks lose spatial precision, so
    there are multiple strategies to do this. Reference:
               https://arxiv.org/abs/1611.07865
    Simple: Just resizes the mask to the layer's (H,W).
    All: Propagate the mask's open region to neurons in the layer that are
         connected to it.
    Inside: Only propagates open values in the mask to a neuron if that
    neuron's receptive field overlaps entirely with open values.

    :param mask:
        HxW Numpy array. Mask that gets propagated so that it can be used at
        other layers.
    :param layer_names:
        A list of names of tensors. Represents the sequence of layers through
        which the image mask gets propagated.
    :param strategy:
        String = Simple, All, or Inside. Strategies as detailed in the above
        reference's supplement.
    :param vggnet:
        Pass in the vggnet model. Used to compute values of layers for a probe
        image that is needed in the All and Inside strategies. Technically, all
        you really need is dimensions defining VGG, but it's simple to just do
        it empirically as done here by passing through a probe image.
    :param batchsize:
        Integer > 1. Only used for All and Inside methods. Used to
        stochastically determine how input neurons influence neurons at
        different layers.
    """
    # Get the spatial dimensions of each layer.
    layers = get_layers(layer_names)
    layer_sizes = [tuple(layer.get_shape().as_list()[1:3]) for layer in layers]

    # Check options.
    if strategy == 'All' or strategy == 'Inside':
        assert(vggnet is not None)
        assert(batchsize > 1)

    # Propagate
    guidance_channels = []
    if strategy == 'Simple':
        # We simply downsample without considering the receptive fields. Some
        # mixing of styles occurs at the boundary.
        for size in layer_sizes:
            layer_guidance_channels = imresize_shape(mask, size)
            layer_guidance_channels = layer_guidance_channels[np.newaxis, :, :,
                                                              np.newaxis]
            guidance_channels.append(layer_guidance_channels)
    elif strategy == 'All':
        # Create batch of probe images filled with noise where the mask is
        # open. We can then look for a noisy signal at later layers by looking
        # at the variance in the batch. We reduce out the channels dimension
        # via the mean. Lots of mixing at the boundary occurs.
        probe = np.zeros((batchsize,) + mask.shape + (3,))
        probe[:, mask.astype(bool), :] += \
                1e2*np.random.randn(*probe[:, mask.astype(bool), :].shape)
        with tf.Session() as sess:
            vggnet.load_weights('libs/vgg16_weights.npz', sess)
            layers_out = sess.run(layers, feed_dict={vggnet.imgs: probe})
        for layer in layers_out:
            layer_guidance_channels = (layer.var(0).mean(2) !=
                                       0.0).astype(np.float32)
            layer_guidance_channels = layer_guidance_channels[np.newaxis, :, :,
                                                              np.newaxis]
            guidance_channels.append(layer_guidance_channels)
    elif strategy == 'Inside':
        # Similar in implementation to 'All' strategy, but this time we invert
        # the mask, and probe which neurons are connected to the closed region
        # of the mask. The values in the guidance channel corresponding to
        # these neurons will be set to 0. This ensures that only neurons whose
        # receptive field lies entirely within the open region of the given
        # mask are included. No mixing occurs, but the boundary is
        # under-stylized.
        inv_mask = -1.0*mask + 1.0
        probe = np.zeros((batchsize,) + inv_mask.shape + (3,))
        probe[:, inv_mask.astype(bool), :] += \
                1e2*np.random.randn(*probe[:, inv_mask.astype(bool), :].shape)
        with tf.Session() as sess:
            vggnet.load_weights('libs/vgg16_weights.npz', sess)
            layers_out = sess.run(layers, feed_dict={vggnet.imgs: probe})
        for layer in layers_out:
            layer_guidance_channels = (layer.var(0).mean(2) ==
                                       0.0).astype(np.float32)
            layer_guidance_channels = layer_guidance_channels[np.newaxis, :, :,
                                                              np.newaxis]
            guidance_channels.append(layer_guidance_channels)
    elif strategy == 'Boundary':
        # In each layer's channels, we interpret the mask as defining the
        # midsection of a boundary, defined as the difference between the
        # All-type channels and the Inside-type channels. This might be useful
        # in getting better results than the Simple strategy. Less mixing
        # supposedly occurs if the boundary region is left unguided according
        # to the reference at the top of this function.
        raise ValueError("Boundary strategy not yet implemented.")
    else:
        raise ValueError("""{} is not a valid strategy name. Options are:
                         Simple, Inside, or All.""".format(strategy))

    return guidance_channels
