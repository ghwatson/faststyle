"""
The Gatys et al original variant.  Note that normally VGG19 is used, which
produces better results. This instantiation uses VGG16.

Author: Grant Watson
Date: February 2017
"""

import tensorflow as tf
from libs import vgg16
import argparse
import losses
import numpy as np
import utils


def setup_parser():
    """Used to interface with the command-line."""
    parser = argparse.ArgumentParser(
                description='Train a style transfer net.')
    parser.add_argument('--style_img_path',
                        help='Path to style template image.')
    parser.add_argument('--cont_img_path',
                        help='Path to content template image.')
    parser.add_argument('--learn_rate',
                        help='Learning rate for optimizer.',
                        default=1e1, type=float)
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
    parser.add_argument('--num_steps_break',
                        help='Max number of steps to iterate optimizer.',
                        default=500,
                        type=int)
    parser.add_argument('--beta',
                        help="""TV regularization weight.""",
                        default=1.e-4,
                        type=float)
    parser.add_argument('--style_target_resize',
                        help="""Scale factor to apply to the style target image.
                        Can change the features that get pronounced.""",
                        default=1.0, type=float)
    parser.add_argument('--cont_target_resize',
                        help="""Resizes content input by this size. Output
                        image will have the same size.""",
                        default=1.0,
                        type=float)
    parser.add_argument('--output_img_path',
                        help='Desired output path. Defaults to out.jpg',
                        default='./out.jpg')
    return parser


def main(args):
    # Unpack command-line arguments.
    style_img_path = args.style_img_path
    cont_img_path = args.cont_img_path
    learn_rate = args.learn_rate
    loss_content_layers = args.loss_content_layers
    loss_style_layers = args.loss_style_layers
    content_weights = args.content_weights
    style_weights = args.style_weights
    num_steps_break = args.num_steps_break
    beta = args.beta
    style_target_resize = args.style_target_resize
    cont_target_resize = args.cont_target_resize
    output_img_path = args.output_img_path

    # Load in style image that will define the model.
    style_img = utils.imread(style_img_path)
    style_img = utils.imresize(style_img, style_target_resize)
    style_img = style_img[np.newaxis, :].astype(np.float32)

    # Alter the names to include a namescope that we'll use + output suffix.
    loss_style_layers = ['vgg/' + i + ':0' for i in loss_style_layers]
    loss_content_layers = ['vgg/' + i + ':0' for i in loss_content_layers]

    # Get target Gram matrices from the style image.
    with tf.variable_scope('vgg'):
        X_vgg = tf.placeholder(tf.float32, shape=style_img.shape,
                               name='input')
        vggnet = vgg16.vgg16(X_vgg)
    with tf.Session() as sess:
        vggnet.load_weights('libs/vgg16_weights.npz', sess)
        print 'Precomputing target style layers.'
        target_grams = sess.run(utils.get_grams(loss_style_layers),
                                feed_dict={'vgg/input:0': style_img})

    # Clean up so we can re-create vgg at size of input content image for
    # training.
    print 'Resetting default graph.'
    tf.reset_default_graph()

    # Read in + resize the content image.
    cont_img = utils.imread(cont_img_path)
    cont_img = utils.imresize(cont_img, cont_target_resize)
    cont_img = cont_img[np.newaxis, :].astype(np.float32)

    # Setup VGG and initialize it with white noise image that we'll optimize.
    shape = cont_img.shape
    with tf.variable_scope('to_train'):
        white_noise = np.random.rand(shape[0], shape[1],
                                     shape[2], shape[3])*255.0
        white_noise = tf.constant(white_noise.astype(np.float32))
        X = tf.get_variable('input', dtype=tf.float32,
                            initializer=white_noise)
    with tf.variable_scope('vgg'):
        vggnet = vgg16.vgg16(X)

    # Get the gram matrices' tensors for the style loss features.
    input_img_grams = utils.get_grams(loss_style_layers)

    # Get the tensors for content loss features.
    content_layers = utils.get_layers(loss_content_layers)

    # Get the target content features
    with tf.Session() as sess:
        vggnet.load_weights('libs/vgg16_weights.npz', sess)
        print 'Precomputing target content layers.'
        content_targets = sess.run(content_layers,
                                   feed_dict={'to_train/input:0': cont_img})

    # Create loss function
    cont_loss = losses.content_loss(content_layers, content_targets,
                                    content_weights)
    style_loss = losses.style_loss(input_img_grams, target_grams,
                                   style_weights)
    tv_loss = losses.tv_loss(X)
    loss = cont_loss + style_loss + beta * tv_loss

    # We do not want to train VGG, so we must grab the subset.
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope='to_train')

    # Setup step + optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learn_rate) \
                  .minimize(loss, global_step, train_vars)

    # Initializer
    init_op = tf.global_variables_initializer()

    # Begin training
    with tf.Session() as sess:
        sess.run(init_op)
        vggnet.load_weights('libs/vgg16_weights.npz', sess)

        current_step = 0
        while current_step < num_steps_break:
            current_step = sess.run(global_step)

            if (current_step % 10 == 0):
                # Collect some diagnostic data for Tensorboard.
                _, loss_out = sess.run([optimizer, loss])

                # Do some standard output.
                print current_step, loss_out
            else:
                # optimizer.minimize(sess)
                _, loss_out = sess.run([optimizer, loss])

        # Upon finishing, get the X tensor (our image).
        img_out = sess.run(X)

    # Save it.
    img_out = np.squeeze(img_out)
    utils.imwrite(output_img_path, img_out)


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
