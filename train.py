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
import datapipe
import os
import argparse

# TODO: Refactor into functions for readability
# TODO: do we need biases on the terms where we removed instance normalization?
# TODO: implement cross-validation for TV reg. For now we'll just use a fixed
# beta.
# TODO: verify that the gradient isn't coming into play for the target content
# data.
# TODO: setup single beta input for if don't want to do CV.


def setup_parser():
    """Used to interface with the command-line."""
    # TODO: remove the defaults I put in for my own convenience.
    parser = argparse.ArgumentParser(
                description='Train a style transfer net.')
    parser.add_argument('--train_dir',
                        default='/home/ghwatson/workspace/faststyle/mockshards/',
                        help='Directory of training  data.')
    parser.add_argument('--style_img_path',
                        default='style_images/starry_night_crop.jpg',
                        help='Path to style template.')
    parser.add_argument('--model_name',
                        default='starry_night',
                        help='Name of model being trained.')
    parser.add_argument('--learn_rate',
                        help='Learning rate for optimizer.',
                        default=1e-3, type=float)
    parser.add_argument('--batch_size',
                        help='Batch size for training.',
                        default=4, type=int)
    parser.add_argument('--n_epochs',
                        help='Number of training epochs.',
                        default=2, type=int)
    parser.add_argument('--preprocess_size',
                        help='Preprocessing dimensions for training.',
                        default=[256, 256], nargs=2, type=int)
    parser.add_argument('--run_name',
                        help='Name of run. Used to create Tensoboard log.',
                        default=None)
    parser.add_argument('--tv_bounds',
                        help='Bounds for CV on TV regularization.',
                        nargs=2,
                        type=int)
    parser.add_argument('--loss_content_layers',
                        help='Names of layers to define content loss.',
                        nargs='*',
                        default=['conv2_2'])
    parser.add_argument('--loss_style_layers',
                        help='Names of layers to define style loss.',
                        nargs='*',
                        default=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3'])
    parser.add_argument('--content_weights',
                        help='Weights corresponding to content layers.',
                        nargs='*',
                        default=[1.0],
                        type=float)
    parser.add_argument('--style_weights',
                        help='Weights corresponding to style layers.',
                        nargs='*',
                        default=[1.0, 1.0, 1.0, 1.0],
                        type=float)
    parser.add_argument('--num_steps_ckpt',
                        help='Interval after which we save a checkpoint.',
                        default=1000,
                        type=int)
    parser.add_argument('--num_pipe_buffer',
                        help='Number of images loaded into RAM in pipeline.',
                        default=4000,
                        type=int)
    return parser


def create_content_loss(content_layers, target_content_layers,
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


def create_style_loss(grams, target_grams, style_weights):
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

    # Style loss (Note normalization is included in gram matrices already)
    style_losses = []
    for i in xrange(num_style_layers):
        gram, target_gram = grams[i], target_grams[i]
        style_weight = style_weights[i]
        loss = tf.reduce_sum(tf.square(gram - tf.constant(target_gram)))
        loss = style_weight * loss
        style_losses.append(loss)
    style_loss = tf.add_n(style_losses, name='style_loss')
    return style_loss


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
    ident = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    v_array = np.array([[ident], [-1*ident]])
    h_array = np.array([[ident, -1*ident]])
    v_filter = tf.constant(v_array, tf.float32)
    h_filter = tf.constant(h_array, tf.float32)

    vdiff = tf.nn.conv2d(X, v_filter, strides=[1, 1, 1, 1], padding='VALID')
    hdiff = tf.nn.conv2d(X, h_filter, strides=[1, 1, 1, 1], padding='VALID')

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
        shape = layer.get_shape().as_list()
        num_elements = shape[1] * shape[2] * shape[3]
        features_matrix = tf.reshape(layer, tf.pack([shape[0], -1, shape[3]]))
        gram_matrix = tf.matmul(features_matrix, features_matrix,
                                transpose_a=True)
        gram_matrix = gram_matrix / tf.cast(num_elements, tf.float32)
        grams.append(gram_matrix)
    return grams


def main(args):
    """main

    :param args:
        argparse.Namespace object from argparse.parse_args().
    """
    # Unpack command-line arguments.
    train_dir = args.train_dir
    style_img_path = args.style_img_path
    model_name = args.model_name
    preprocess_size = args.preprocess_size
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    run_name = args.run_name
    learn_rate = args.learn_rate
    tv_bounds = args.tv_bounds
    loss_content_layers = args.loss_content_layers
    loss_style_layers = args.loss_style_layers
    content_weights = args.content_weights
    style_weights = args.style_weights
    num_steps_ckpt = args.num_steps_ckpt
    num_pipe_buffer = args.num_pipe_buffer

    # Load in style image that will define the model.
    style_img = plt.imread(style_img_path)
    style_img = style_img[np.newaxis, :].astype(np.float32)

    # Get target Gram matrices from the style image.
    with tf.variable_scope('vgg'):
        X_vgg = tf.placeholder(tf.float32, shape=style_img.shape,
                               name='input')
        vggnet = vgg16.vgg16(X_vgg)
    with tf.Session() as sess:
        vggnet.load_weights('libs/vgg16_weights.npz', sess)
        print 'Precomputing target style layers.'
        target_grams = sess.run(get_style_layers(loss_style_layers),
                                feed_dict={'vgg/input:0': style_img})

    # Clean up so we can re-create vgg connected to our image network.
    print 'Resetting default graph.'
    tf.reset_default_graph()

    # Load in image transformation network into default graph.
    shape = [batch_size] + preprocess_size + [3]
    with tf.variable_scope('img_t_net'):
        img_t_out = create_net(shape)

    # Connect vgg directly to the image transformation network.
    with tf.variable_scope('vgg'):
        vggnet = vgg16.vgg16(img_t_out)

    # Get the input/output
    g = tf.get_default_graph()
    X = g.get_tensor_by_name('img_t_net/input:0')
    Y = g.get_tensor_by_name('img_t_net/output:0')

    # Get the gram matrices' tensors for the style loss features.
    input_img_grams = get_style_layers(loss_style_layers)

    # Get the tensors for content loss features.
    content_layers_names = ['vgg/' + i + ':0' for i in loss_content_layers]
    content_layers = [g.get_tensor_by_name(name) for name
                      in content_layers_names]

    # Create loss function
    content_targets = tuple(tf.placeholder(tf.float32,
                            shape=[None, None, None, None],
                            name='content_input_{}'.format(i))
                            for i, _ in enumerate(loss_content_layers))
    # perc_loss = create_perceptual_loss(input_img_grams, target_grams,
                                       # content_layers, content_targets,
                                       # style_weights, content_weights)
    cont_loss = create_content_loss(content_layers, content_targets,
                                    content_weights)
    style_loss = create_style_loss(input_img_grams, target_grams,
                                   style_weights)
    tv_loss = create_tv_loss(Y)
    beta = tf.placeholder(tf.float32, shape=[], name='tv_scale')
    # loss = tf.add(perc_loss, tf.mul(beta, tv_loss), name='loss')
    loss = cont_loss + style_loss + beta * tv_loss
    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('style_loss', loss)
        tf.summary.scalar('content_loss', cont_loss)
        tf.summary.scalar('tv_loss', tv_loss)

    # Setup input pipeline (delegate it to CPU to let GPU handle neural net)
    files = tf.train.match_filenames_once(train_dir + '/train-*')
    with tf.variable_scope('input_pipe'), tf.device('/cpu:0'):
        batch_op = datapipe.batcher(files, batch_size, preprocess_size,
                                    n_epochs, num_pipe_buffer)

    # We do not want to train VGG, so we must grab the subset.
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope='img_t_net')

    # Setup step + optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss, global_step,
                                                            train_vars)

    # Setup subdirectory for this run's Tensoboard logs.
    if run_name is None:
        current_dirs = [name for name in os.listdir('./summaries/train/')
                        if os.path.isdir('./summaries/train/' + name)]
        name = 'run0'
        count = 0
        while name in current_dirs:
            count += 1
            name = 'run{}'.format(count)
        run_name = name
        os.makedirs(run_name)

    # Savers and summary writers
    saver = tf.train.Saver()
    final_saver = tf.train.Saver(train_vars)
    merged = tf.summary.merge_all()
    full_log_path = './summaries/train/' + run_name
    train_writer = tf.summary.FileWriter(full_log_path)

    # We must include local variables because of batch pipeline.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Begin training.
    print 'Starting training...'
    with tf.Session() as sess:
        # Initialization
        sess.run(init_op)
        vggnet.load_weights('libs/vgg16_weights.npz', sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                current_step = sess.run(global_step)
                batch = sess.run(batch_op)

                # Collect content targets
                content_data = sess.run(content_layers,
                                        feed_dict={Y: batch})

                # TODO: swap out the hard-coded beta with cross-validation
                feed_dict = {X: batch,
                             content_targets: content_data,
                             beta: 0.}
                if (current_step % num_steps_ckpt == 0):
                    # Save a checkpoint
                    save_path = 'training/' + model_name + '.ckpt'
                    saver.save(sess, save_path, global_step=global_step)
                    summary, _, loss_out = sess.run([merged, optimizer, loss],
                                                    feed_dict=feed_dict)
                    train_writer.add_summary(summary, current_step)
                    print current_step, loss_out

                elif (current_step % 10 == 0):
                    # Collect some diagnostic data for Tensorboard.
                    summary, _, loss_out = sess.run([merged, optimizer, loss],
                                                    feed_dict=feed_dict)
                    train_writer.add_summary(summary, current_step)

                    # Do some standard output.
                    print current_step, loss_out
                else:
                    _, loss_out = sess.run([optimizer, loss],
                                           feed_dict=feed_dict)

        except tf.errors.OutOfRangeError:
            print('Done training.')
        finally:
            # Save the model (the image transformation network) for later usage
            # in predict.py
            final_saver.save(sess, 'models/' + model_name + '_final.ckpt')

            coord.request_stop()

        coord.join(threads)


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
