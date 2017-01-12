"""
Train image transformation network in conjunction with perceptual loss. Save
the image transformation network for later application.

File author: Grant Watson
Date: Jan 2017
"""

import tensorflow as tf
import numpy as np
from libs import vgg16
from im_transf_net import create_net
from matplotlib import pyplot as plt

# TODO: Move style image to file argument
# TODO: Ensure that vgg16 has trainable=False
# TODO: Refactor into functions for readability
# TODO: Get to the bottom of learning how to feedforward from the middle of a
# graph in a shape-dynamic fashion, without having to create multiple graphdefs
# or graphs or using reset_default_graph().
# TODO: Make code batch-proof.
# TODO: My dynamic method for calculating the number of elements seems kind of
# hacky.
# TODO: do we need biases on the terms where we removed instance normalization?


def create_perceptual_loss(grams, target_grams, content_layers,
                         target_content_layers,
                         style_weights, content_weights):
    """Defines the perceptual loss function.

    :param grams
        List of tensors for Gram matrices derived from training graph.
    :param target_grams
        List of numpy arrays for Gram matrices precomputed from style image.
    :param content_layers
        List of tensors for layers derived from training graph.
    :param target_content_layers
        List of placeholders to be filled with content layer data.
    :param style_weights
        List of floats to be used as weights for style layers.
    :param content_weights
        List of floats to be used as weights for content layers.
    """
    assert(len(target_content_layers) == len(content_layers))
    assert(len(grams) == len(target_grams))
    num_style_layers = len(target_grams)
    num_content_layers = len(target_content_layers)

    # Style loss
    style_losses = []
    for i in xrange(num_style_layers):
        gram, target_gram = grams[i], target_grams[i]
        style_weight = style_weights[i]
        loss = tf.reduce_sum(tf.square(gram - tf.constant(target_gram)))
        loss = style_weight * loss
        style_losses.append(loss)
    style_loss = tf.add_n(style_losses)

    # Content loss (identical to above code for style)
    content_losses = []
    for i in xrange(num_content_layers):
        gram, target_gram = grams[i], target_grams[i]
        content_weight = style_weights[i]
        loss = tf.reduce_sum(tf.square(gram - tf.constant(target_gram)))
        loss = content_weight * loss
        content_losses.append(loss)
    content_loss = tf.add_n(content_losses)

    total_loss = style_loss + content_loss
    return total_loss

def create_tv_loss(X):
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
    ident = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
    v_array = np.array([[ident],[-1*ident]])
    h_array = np.array([[ident, -1*ident]])
    v_filter = tf.constant(v_array, tf.float32)
    h_filter = tf.constant(h_array, tf.float32)

    vdiff = tf.nn.conv2d(X, v_filter, strides=[1,1,1,1], padding='VALID')
    hdiff = tf.nn.conv2d(X, h_filter, strides=[1,1,1,1], padding='VALID')

    loss = tf.reduce_sum(tf.square(hdiff)) + tf.reduce_sum(tf.square(vdiff))

    return loss


def get_style_layers(layer_names):
    """Get the style layer tensors from the VGG graph (presumed to be loaded into
    default).

    :param layer_names
        Names of the layers in tf's default graph
    """
    g = tf.get_default_graph()
    grams = []
    style_layers_names = ['vgg/' + i + ':0' for i in layer_names]
    style_layers = [g.get_tensor_by_name(name) for name
                    in style_layers_names]
    for i, layer in enumerate(style_layers):
        shape = tf.shape(layer)
        num_elements = tf.reduce_sum(tf.ones(shape[1::]))
        features_matrix = tf.reshape(layer, tf.pack([-1, shape[3]]))
        gram_matrix = tf.matmul(tf.transpose(features_matrix), features_matrix)
        gram_matrix = gram_matrix / num_elements
        grams.append(gram_matrix)
    return grams


if __name__ == "__main__":

    # Training hyperparameters
    mscoco_shape = [256, 256, 3]  # We'll compress to this
    batch_size = 2  # TODO: change back to 4
    n_iter = 40000
    learn_rate = 1e-3
    tv_reg_bounds = [1e-6, 1e-4]
    loss_content_layers = ['conv2_2']
    loss_style_layers = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']
    content_weights = [1.0]
    style_weights = [1.0, 1.0, 1.0, 1.0]

    # Load in style image that will define the model.
    style_img = plt.imread('style_images/starry_night_crop.jpg')
    style_img = style_img[np.newaxis, :].astype(np.float32)

    # Get target Gram matrices from the style image.
    with tf.variable_scope('vgg'):
        X_vgg = tf.placeholder(tf.float32, shape=[None, None, None, 3],
                               name='input')
        vggnet = vgg16.vgg16(X_vgg)
    with tf.Session() as sess:
        vggnet.load_weights('libs/vgg16_weights.npz', sess)
        print 'starting gram'
        target_grams = sess.run(get_style_layers(loss_style_layers),
                                feed_dict={'vgg/input:0': style_img})
        print 'done gram'

    # Clean up so we can re-create vgg connected to our image network.
    tf.reset_default_graph()

    # Load in image transformation network into default graph.
    shape = [batch_size] + mscoco_shape
    with tf.variable_scope('img_t_net'):
        img_t_out = create_net(shape)

    # Connect vgg directly to the image transformation network.
    with tf.variable_scope('vgg'):
        vggnet = vgg16.vgg16(img_t_out)
    with tf.Session() as sess:
        vggnet.load_weights('libs/vgg16_weights.npz', sess)

    # Get the input
    g = tf.get_default_graph()
    X = g.get_tensor_by_name('img_t_net/input:0')

    # Get the gram matrices' tensors for the style loss features.
    input_img_grams = get_style_layers(loss_style_layers)

    # Get the tensors for content loss features.
    content_layers_names = ['vgg/' + i + ':0' for i in loss_content_layers]
    content_layers = [g.get_tensor_by_name(name) for name
                      in content_layers_names]

    # Create loss function
    content_targets = [tf.placeholder(tf.float32,
                       shape=[None, None, None, None], name='input')
                       for _ in loss_content_layers]
    perc_loss = create_perceptual_loss(input_img_grams, target_grams, content_layers,
                         content_targets, style_weights, content_weights)
    tv_loss = create_tv_loss(g.get_tensor_by_name('img_t_net/output:0'))
    total_loss = tf.add(perc_loss, tv_loss, name='loss')

    # Prime MS-Coco dataqueuer.

    # Load in style image.

    # Train.

    # Save the model (the image transformation network) for later usage in
    # predict.py.
