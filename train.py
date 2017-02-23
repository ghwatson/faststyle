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
import datapipe
import os
import argparse
import utils
import losses

# TODO: implement conditional default in argparse for beta. Depends on
# upsampling method.


def setup_parser():
    """Used to interface with the command-line."""
    parser = argparse.ArgumentParser(
                description='Train a style transfer net.')
    parser.add_argument('--train_dir',
                        help='Directory of TFRecords training data.')
    parser.add_argument('--model_name',
                        help='Name of model being trained.')
    parser.add_argument('--style_img_path',
                        default='./style_images/starry_night_crop.jpg',
                        help='Path to style target image.')
    parser.add_argument('--learn_rate',
                        help='Learning rate for Adam optimizer.',
                        default=1e-3, type=float)
    parser.add_argument('--batch_size',
                        help='Batch size for training.',
                        default=4, type=int)
    parser.add_argument('--n_epochs',
                        help='Number of training epochs.',
                        default=2, type=int)
    parser.add_argument('--preprocess_size',
                        help="""Dimensions to resize training images to before passing
                        them into the image transformation network.""",
                        default=[256, 256], nargs=2, type=int)
    parser.add_argument('--run_name',
                        help="""Name of log directory within the Tensoboard
                        directory (./summaries). If not set, will use
                        --model_name to create a unique directory.""",
                        default=None)
    parser.add_argument('--loss_content_layers',
                        help='Names of layers to define content loss.',
                        nargs='*',
                        default=['conv3_3'])
    parser.add_argument('--loss_style_layers',
                        help='Names of layers to define style loss.',
                        nargs='*',
                        default=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3'])
    parser.add_argument('--content_weights',
                        help="""Weights that multiply the content loss
                        terms.""",
                        nargs='*',
                        default=[1.0],
                        type=float)
    parser.add_argument('--style_weights',
                        help="""Weights that multiply the style loss terms.""",
                        nargs='*',
                        default=[5.0, 5.0, 5.0, 5.0],
                        type=float)
    parser.add_argument('--num_steps_ckpt',
                        help="""Save a checkpoint everytime this number of
                        steps passes in training.""",
                        default=1000,
                        type=int)
    parser.add_argument('--num_pipe_buffer',
                        help="""Number of images loaded into RAM in pipeline.
                        The larger, the better the shuffling, but the more RAM
                        filled, and a slower startup.""",
                        default=4000,
                        type=int)
    parser.add_argument('--num_steps_break',
                        help="""Max on number of steps. Training ends when
                        either num_epochs or this is reached (whichever comes
                        first).""",
                        default=-1,
                        type=int)
    parser.add_argument('--beta',
                        help="""TV regularization weight. If using deconv for
                        --upsample_method, try 1.e-4 for starters. Otherwise,
                        this is not needed.""",
                        default=0.0,
                        type=float)
    parser.add_argument('--style_target_resize',
                        help="""Scale factor to apply to the style target image.
                        Can change the dominant stylistic features.""",
                        default=1.0, type=float)
    parser.add_argument('--upsample_method',
                        help="""Either deconvolution as in the original paper,
                        or the resize convolution method. The latter seems
                        superior and does not require TV regularization through
                        beta.""",
                        choices=['deconv', 'resize'],
                        default='resize')
    return parser


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
    loss_content_layers = args.loss_content_layers
    loss_style_layers = args.loss_style_layers
    content_weights = args.content_weights
    style_weights = args.style_weights
    num_steps_ckpt = args.num_steps_ckpt
    num_pipe_buffer = args.num_pipe_buffer
    num_steps_break = args.num_steps_break
    beta_val = args.beta
    style_target_resize = args.style_target_resize
    upsample_method = args.upsample_method

    # Load in style image that will define the model.
    style_img = utils.imread(style_img_path)
    style_img = utils.imresize(style_img, style_target_resize)
    style_img = style_img[np.newaxis, :].astype(np.float32)

    # Alter the names to include a namescope that we'll use + output suffix.
    loss_style_layers = ['vgg/' + i + ':0' for i in loss_style_layers]
    loss_content_layers = ['vgg/' + i + ':0' for i in loss_content_layers]

    # Get target Gram matrices from the style image.
    with tf.variable_scope('vgg'):
        X_vgg = tf.placeholder(tf.float32, shape=style_img.shape, name='input')
        vggnet = vgg16.vgg16(X_vgg)
    with tf.Session() as sess:
        vggnet.load_weights('libs/vgg16_weights.npz', sess)
        print 'Precomputing target style layers.'
        target_grams = sess.run(utils.get_grams(loss_style_layers),
                                feed_dict={X_vgg: style_img})

    # Clean up so we can re-create vgg connected to our image network.
    print 'Resetting default graph.'
    tf.reset_default_graph()

    # Load in image transformation network into default graph.
    shape = [batch_size] + preprocess_size + [3]
    with tf.variable_scope('img_t_net'):
        X = tf.placeholder(tf.float32, shape=shape, name='input')
        Y = create_net(X, upsample_method)

    # Connect vgg directly to the image transformation network.
    with tf.variable_scope('vgg'):
        vggnet = vgg16.vgg16(Y)

    # Get the gram matrices' tensors for the style loss features.
    input_img_grams = utils.get_grams(loss_style_layers)

    # Get the tensors for content loss features.
    content_layers = utils.get_layers(loss_content_layers)

    # Create loss function
    content_targets = tuple(tf.placeholder(tf.float32,
                            shape=layer.get_shape(),
                            name='content_input_{}'.format(i))
                            for i, layer in enumerate(content_layers))
    cont_loss = losses.content_loss(content_layers, content_targets,
                                    content_weights)
    style_loss = losses.style_loss(input_img_grams, target_grams,
                                   style_weights)
    tv_loss = losses.tv_loss(Y)
    beta = tf.placeholder(tf.float32, shape=[], name='tv_scale')
    loss = cont_loss + style_loss + beta * tv_loss
    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('style_loss', style_loss)
        tf.summary.scalar('content_loss', cont_loss)
        tf.summary.scalar('tv_loss', beta*tv_loss)

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
    if not os.path.exists('./summaries/train/'):
        os.makedirs('./summaries/train/')
    if run_name is None:
        current_dirs = [name for name in os.listdir('./summaries/train/')
                        if os.path.isdir('./summaries/train/' + name)]
        name = model_name + '0'
        count = 0
        while name in current_dirs:
            count += 1
            name = model_name + '{}'.format(count)
        run_name = name

    # Savers and summary writers
    if not os.path.exists('./training'):  # Dir that we'll later save .ckpts to
        os.makedirs('./training')
    if not os.path.exists('./models'):  # Dir that save final models to
        os.makedirs('./models')
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

                feed_dict = {X: batch,
                             content_targets: content_data,
                             beta: beta_val}
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

                # Throw error if we reach number of steps to break after.
                if current_step == num_steps_break:
                    print('Done training.')
                    break
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
