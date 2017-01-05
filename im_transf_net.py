"""
Functions used for the creation of the transformation network.

Batch normalization functions are provided but not utilized.
"""

import numpy as np
import tensorflow as tf


def create_net(image):
    """Creates the transformation network, given dimensions acquired from an
    input image. Does this according to J.C. Johnson's specifications
    after utilizing instance normalization (i.e. halving dimensions given
    in the paper).

    :param image
        Input image in numpy array form with NxHxWxC dimensions.
    """
    shape = image.shape

    X = tf.placeholder(tf.float32, shape=shape, name="input")

    # Padding
    h = reflect_pad(X, 40)

    # Initial convolutional layers
    with tf.variable_scope('initconv_0'):
        h = relu(inst_norm(conv2d(h, 3, 16, 9, [1, 1, 1, 1])))
    with tf.variable_scope('initconv_1'):
        h = relu(inst_norm(conv2d(h, 16, 32, 3, [1, 2, 2, 1])))
    with tf.variable_scope('initconv_2'):
        h = relu(inst_norm(conv2d(h, 32, 64, 3, [1, 2, 2, 1])))

    # Residual layers
    with tf.variable_scope('resblock_0'):
        h = res_layer(h, 64, 3, [1, 1, 1, 1])
    with tf.variable_scope('resblock_1'):
        h = res_layer(h, 64, 3, [1, 1, 1, 1])
    with tf.variable_scope('resblock_2'):
        h = res_layer(h, 64, 3, [1, 1, 1, 1])

    # Deconvolutional layers
    with tf.variable_scope('deconv_0'):
        h = relu(inst_norm(deconv2d(h, 64, 32, 3, [1, 2, 2, 1])))
    with tf.variable_scope('deconv_1'):
        h = relu(inst_norm(deconv2d(h, 32, 16, 3, [1, 2, 2, 1])))
    with tf.variable_scope('deconv_2'):
        h = relu(inst_norm(deconv2d(h, 16, 3, 9, [1, 1, 1, 1])))

    # Output layers
    out = h

    return out


def reflect_pad(X, padsize):
    """Pre-net padding.

    :param X
        Input image tensor
    :param padsize
        Amount by which to pad the image tensor
    """
    h = tf.pad(X, paddings=[[0, 0], [padsize, padsize], [padsize, padsize],
                            [0, 0]], mode='REFLECT')
    return h


def conv2d(X, n_ch_in, n_ch_out, kernel_size, strides, name=None,
           padding='SAME'):
    """Creates the convolutional layer.

    :param X
        Input tensor
    :param n_ch_in
        Number of input channels
    :param n_ch_out
        Number of output channels
    :param kernel_size
        Dimension of the square-shaped convolutional kernel
    :param strides
        Length 4 vector of stride information
    :param name
        Optional name for the weight matrix
    """
    if name is None:
        name = 'W'
    shape = [kernel_size, kernel_size, n_ch_in, n_ch_out]
    W = tf.get_variable(name=name,
                        shape=shape,
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer())
    h = tf.nn.conv2d(X,
                     filter=W,
                     strides=strides,
                     padding=padding)
    return h


def deconv2d(X, n_ch_in, n_ch_out, kernel_size, strides):
    """Creates a transposed convolutional layer.

    :param X
        Input tensor
    :param n_ch_in
        Number of input channels
    :param n_ch_out
        Number of output channels
    :param kernel_size
        Size of square shaped deconvolutional kernel
    :param strides
        Stride information
    """
    # Note the in and out channels reversed for deconv shape
    shape = [kernel_size, kernel_size, n_ch_out, n_ch_in]

    # Construct output shape of the deconvolution
    new_h = X.get_shape().as_list()[1]*strides[1]
    new_w = X.get_shape().as_list()[2]*strides[2]
    output_shape = [X.get_shape().as_list()[0], new_h, new_w, n_ch_out]

    W = tf.get_variable(name='W',
                        shape=shape,
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer())
    h = tf.nn.conv2d_transpose(X,
                               output_shape=output_shape,
                               filter=W,
                               strides=strides,
                               padding="SAME")

    return h


def relu(X):
    """Performs relu on the tensor.

    :param X
        Input tensor
    """
    return tf.nn.relu(X, name='relu')


def inst_norm(inputs, epsilon=1e-3):
    """
    Assuming TxHxWxC dimensions on the tensor, will normalize over
    the H,W dimensions. Use this before the activation layer.
    This function borrows from:
        http://r2rt.com/implementing-batch-normalization-in-tensorflow.html

    Note this is similar to batch_normalization, which normalizes each
    neuron by looking at its statistics over the batch.

    :param input_:
        input tensor of NHWC format
    """
    # Create scale + shift. Exclude batch dimension.
    stat_shape = inputs.get_shape().as_list()
    scale = tf.Variable(tf.ones(stat_shape[1::]), name='INscale')
    shift = tf.Variable(tf.zeros(stat_shape[1::]), name='INshift')

    inst_means, inst_vars = tf.nn.moments(inputs, axes=[1, 2],
                                          keep_dims=True)

    # Normalization
    inputs_normed = (inputs - inst_means) / tf.sqrt(inst_vars + epsilon)

    # Perform trainable shift.
    output = scale * inputs_normed + shift

    return output


def res_layer(X, n_ch, kernel_size, strides):
    """Creates a residual block layer.

    :param X
        Input tensor
    :param n_ch
        Number of input channels
    :param kernel_size
        Size of square shaped convolutional kernel
    :param strides
        Stride information
    """
    h = conv2d(X, n_ch, n_ch, kernel_size, strides, name='W1', padding='VALID')
    h = relu(inst_norm(h))
    h = conv2d(h, n_ch, n_ch, kernel_size, strides, name='W2', padding='VALID')
    h = inst_norm(h)

    # Crop for skip connection
    in_shape = X.get_shape().as_list()
    begin = [0, 2, 2, 0]
    size = [-1, in_shape[1]-4, in_shape[2]-4, -1]
    X_crop = tf.slice(X, begin=begin, size=size)

    # Residual skip connection
    h = tf.add(h, X_crop, name='res_out')

    return h


if __name__ == "__main__":
    # Create the transformation network with some bogus image.
    img = np.zeros((256, 256, 3))
    img *= 0.0
    img[0:128, :, 0] = 1.0
    img = img[np.newaxis, :]
    img2 = np.zeros((256, 256, 3))
    img2[0:128, :, 0] = 0.5
    img2 = img2[np.newaxis, :]
    print img.shape
    img = np.append(img, img2, axis=0)
    print img.shape

    net = create_net(img)
    g = tf.get_default_graph()
    # Attempt to initialize the net and feedforward the bogus image.
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        X = g.get_tensor_by_name('input:0')
        scale = g.get_tensor_by_name('initconv_0/INscale:0')
        outt = g.get_tensor_by_name('deconv_2/relu:0')
        output = sess.run(outt, feed_dict={X: img})
