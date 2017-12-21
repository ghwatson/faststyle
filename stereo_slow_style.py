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
import cv2
from datapipe_stereo import preprocessing, preprocessing_masks, \
    preprocessing_disparity


def setup_parser():
    """Used to interface with the command-line."""
    parser = argparse.ArgumentParser(
                description='Train a style transfer net.')
    parser.add_argument('--style_img_path',
                        help='Path to style template image.')
    parser.add_argument('--cont_img_path_left',
                        help='Path to content template image.')
    parser.add_argument('--cont_img_path_right',
                        help='Path to content template image.')
    parser.add_argument('--disparity_path',
                        help='Path to content template image.')
    parser.add_argument('--occlusion_mask_path',
                        help='Path to content template image.')
    parser.add_argument('--outofframe_mask_path',
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
    parser.add_argument('--disparity_weight',
                        help="""disparity loss weight.""",
                        default=1.0,
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
    cont_img_path_left = args.cont_img_path_left
    cont_img_path_right = args.cont_img_path_right
    disparity_path = args.disparity_path
    occ_mask_path = args.occlusion_mask_path
    oof_mask_path = args.outofframe_mask_path
    learn_rate = args.learn_rate
    loss_content_layers = args.loss_content_layers
    loss_style_layers = args.loss_style_layers
    content_weights = args.content_weights
    style_weights = args.style_weights
    disparity_weight = args.disparity_weight
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
    cont_img_l = utils.imread(cont_img_path_left)
    cont_img_l = utils.imresize(cont_img_l, cont_target_resize)
    cont_img_l = cont_img_l[np.newaxis, :].astype(np.float32)
    cont_img_r = utils.imread(cont_img_path_right)
    cont_img_r = utils.imresize(cont_img_r, cont_target_resize)
    cont_img_r = cont_img_r[np.newaxis, :].astype(np.float32)

    # Read in disparity mask
    disparity = utils.imread(disparity_path)
    rescale = np.array([4.0, 1./2**6, 1./2**14])
    disparity = np.sum(rescale*disparity, axis=2)
    disparity = utils.imresize(disparity, cont_target_resize)
    # rescale disparity.
    disparity = disparity * cont_target_resize
    disparity = tf.stack([disparity, tf.zeros(disparity.shape)],axis=2)
    h, w, _ = disparity.get_shape().as_list()
    X, Y = tf.meshgrid(range(w), range(h))
    warp_y = tf.to_float(Y) - disparity[:, :, 1]
    warp_x = tf.to_float(X) - disparity[:, :, 0]
    warp = tf.stack([warp_x, warp_y], axis=2)
    warp = warp[tf.newaxis, :, :, :]

    # TODO: read in occlusion mask
    occ_mask = utils.imread(occ_mask_path)
    occ_mask = utils.imresize(occ_mask, cont_target_resize,
                              method=cv2.INTER_NEAREST)
    occ_mask = occ_mask[np.newaxis, :].astype(np.float32)

    # TODO: read in outofframe mask
    oof_mask = utils.imread(oof_mask_path)
    oof_mask = utils.imresize(oof_mask, cont_target_resize,
                              method=cv2.INTER_NEAREST)
    oof_mask = oof_mask[np.newaxis, :].astype(np.float32)


    # Setup VGG and initialize it with white noise image that we'll optimize.
    shape = cont_img_l.shape
    with tf.variable_scope('to_train'):
        white_noise = np.random.rand(shape[0], shape[1],
                                     shape[2], shape[3])*255.0
        white_noise = tf.constant(white_noise.astype(np.float32))
        Xl = tf.get_variable('input_l', dtype=tf.float32,
                            initializer=white_noise)
        Xr = tf.get_variable('input_r', dtype=tf.float32,
                            initializer=white_noise)
        X = tf.concat([Xl, Xr], 0)
    with tf.variable_scope('vgg'):
        vggnet = vgg16.vgg16(X)

    # Get the gram matrices' tensors for the style loss features.
    input_img_grams = utils.get_grams(loss_style_layers)
    input_img_grams_l, input_img_grams_r = [], []
    # unwrap to left right eye. (2B -> 1B)
    for gram in input_img_grams:
        gram_l, gram_r = tf.split(gram, 2)
        input_img_grams_l.append(gram_l)
        input_img_grams_r.append(gram_r)

    # Get the tensors for content loss features.
    content_layers_b = utils.get_layers(loss_content_layers)
    content_layers_l, content_layers_r = [], []
    for layer in content_layers_b:
        layer_l, layer_r = tf.split(layer, 2)
        content_layers_l.append(layer_l)
        content_layers_r.append(layer_r)
    content_layers = [content_layers_l, content_layers_r]

    # Get the target content features
    with tf.Session() as sess:
        vggnet.load_weights('libs/vgg16_weights.npz', sess)
        print 'Precomputing target content layers.'
        content_targets_l, content_targets_r = sess.run(content_layers,
                                   feed_dict={'to_train/input_l:0': cont_img_l,
                                              'to_train/input_r:0': cont_img_r})

    # Create loss function
    cont_loss_l = losses.content_loss(content_layers_l, content_targets_l,
                                    content_weights)
    cont_loss_r = losses.content_loss(content_layers_r, content_targets_r,
                                    content_weights)
    style_loss_l = losses.style_loss(input_img_grams_l, target_grams,
                                     style_weights)
    style_loss_r = losses.style_loss(input_img_grams_r, target_grams,
                                     style_weights)

    target_mask = (1.-occ_mask/255.0)*(1.-oof_mask/255.0)
    disparity_loss = losses.disparity_loss(Xl, Xr, warp,
                                           disparity_weight,
                                           target_mask)


    tv_loss = losses.tv_loss(Xl) + losses.tv_loss(Xr)
    loss = cont_loss_l + cont_loss_r + style_loss_l + style_loss_r + \
            disparity_loss + beta * tv_loss

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
        print img_out.shape
        img_out = np.concatenate([img_out[0], img_out[1]], axis=1)
        print img_out.shape

    # Save it.
    img_out = np.squeeze(img_out)
    utils.imwrite(output_img_path, img_out)


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
