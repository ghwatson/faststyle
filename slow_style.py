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
    parser.add_argument('--style_img_paths',
                        default=['./style_images/starry_night_crop.jpg'],
                        nargs='+',
                        help="""Path to style template image. Can pass more
                        than one in the case of spatial control.""")
    parser.add_argument('--cont_img_path',
                        default='./results/chicago.jpg',
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
    parser.add_argument('--style_target_scales',
                        help="""Scale factor(s) to apply to the style target
                        image(s). Can change the dominant stylistic
                        features.""",
                        nargs='+',
                        default=[1.0], type=float)
    parser.add_argument('--cont_target_scale',
                        help="""Resizes content input by this size. Output
                        image will have the same size.""",
                        default=1.0,
                        type=float)
    parser.add_argument('--output_img_path',
                        help='Desired output path. Defaults to out.jpg',
                        default='./out.jpg')
    parser.add_argument('--region_weights',
                        help="""Region weights that go with each style image
                        provided. These are used in spatial control. The number
                        of weights provided must be the same as style images
                        provided. Setting a weight to zero will attempt to
                        keep the respective region entirely content.""",
                        nargs='+',
                        default=[1.0],
                        type=float)
    parser.add_argument('--style_mask_paths',
                        help="""These are paths to image masks with greyscale
                        values in the range [0,1]. Used to identify regions of
                        style images that can be used at test-time to filter
                        regions of a content image. if the string \"OPEN\" is
                        passed, then an open mask consisting of all 1's wil be
                        created.""",
                        nargs='*',
                        default=['OPEN'])
    parser.add_argument('--content_mask_paths',
                        help="""These are paths to image masks with greyscale
                        values in the range [0,1]. These specify which regions
                        of the content image we want to have affected by the
                        styles encoded within the trained model. Can pass OPEN
                        or CLOSED to specify an open or closed mask.""",
                        nargs='*',
                        default=['OPEN'])
    # To be added:
    # parser.add_argument('--spatial_strategy',
                        # help="""Specifies which propagation strategy to use for
                        # the supplied masks. Only relevant if utilizing spatial
                        # control. See utils.propagate_mask for details. The
                        # Ideal strategy is the Inside strategy combined with
                        # unguided stylization.""",
                        # nargs='*',
                        # choices=['All', 'Simple', 'Inside'],
                        # default='Simple')

    return parser


def main(args):
    # Unpack command-line arguments.
    style_img_paths = args.style_img_paths
    cont_img_path = args.cont_img_path
    learn_rate = args.learn_rate
    loss_content_layers = args.loss_content_layers
    loss_style_layers = args.loss_style_layers
    content_weights = args.content_weights
    style_weights = args.style_weights
    num_steps_break = args.num_steps_break
    beta = args.beta
    style_target_scales = args.style_target_scales
    cont_target_scale = args.cont_target_scale
    output_img_path = args.output_img_path
    style_mask_paths = args.style_mask_paths
    content_mask_paths = args.content_mask_paths
    region_weights = args.region_weights
    # spatial_strategy = args.spatial_strategy

    # Check options
    assert(len(style_mask_paths) == len(region_weights))
    assert(len(style_mask_paths) == len(style_img_paths))
    assert(len(content_mask_paths) == len(region_weights))

    # Load in style image that will define the model.
    style_imgs = [utils.imread(path) for path in style_img_paths]
    style_imgs = [utils.imresize(img, s) for img, s in
                  zip(style_imgs, style_target_scales)]
    style_imgs = [img[np.newaxis, :].astype(np.float32) for img in style_imgs]

    # Read in + resize the content image.
    cont_img = utils.imread(cont_img_path)
    cont_img = utils.imresize(cont_img, cont_target_scale)
    cont_img = cont_img[np.newaxis, :].astype(np.float32)

    # Pack data defining regions for spatial control into dict.
    regions = []
    num_regions = len(region_weights)
    for i in xrange(num_regions):
        # Get + preprocess style mask.
        if style_mask_paths[i] != 'OPEN':
            style_mask = utils.imread(style_mask_paths[i], 0)
            style_mask = utils.imresize(style_mask, style_target_scales[i])
            style_mask = style_mask/255.0
        else:
            # Generate open masks if no mask provided.
            style_mask = np.ones(style_imgs[i].shape[1:3])

        if content_mask_paths[i] == 'OPEN':
            # Generate open masks if no mask provided.
            content_mask = np.ones(cont_img.shape[1:3])
        elif content_mask_paths[i] == 'CLOSED':
            content_mask = np.zeros(cont_img.shape[1:3])
        else:
            content_mask = utils.imread(content_mask_paths[i], 0)
            content_mask = utils.imresize(content_mask, cont_target_scale)
            content_mask = content_mask/255.0

        # Normalize by number of regions.
        weight = region_weights[i]*1.0/num_regions

        # Pack data into region
        region = {'weight': weight, 'style_img': style_imgs[i],
                  'style_mask': style_mask, 'content_mask': content_mask}
        regions.append(region)

    # Alter the names to include a namescope that we'll use + output suffix.
    loss_style_layers = ['vgg/' + i + ':0' for i in loss_style_layers]
    loss_content_layers = ['vgg/' + i + ':0' for i in loss_content_layers]

    # Get target (guided) Gram matrices from the style image(s).
    for region in regions:
        with tf.variable_scope('vgg'):
            X_vgg = tf.placeholder(tf.float32, shape=region['style_img'].shape)
            vggnet = vgg16.vgg16(X_vgg)

        # Spatial control: given the current VGG architecture induced by the
        # style image, we propagate the mask.
        guide_channels = utils.propagate_mask(region['style_mask'],
                                              loss_style_layers)

        with tf.Session() as sess:
            vggnet.load_weights('libs/vgg16_weights.npz', sess)
            print 'Precomputing target style layers.'
            grams_tensors = utils.get_grams(loss_style_layers, guide_channels)
            grams_target = sess.run(grams_tensors,
                                    feed_dict={X_vgg: region['style_img']})

        # Pack this into the region data.
        region['grams_target'] = grams_target

        # Clean up so we can reinstantiate a new VGG (we're working with a
        # static graph for the time being).
        print 'Resetting default graph.'
        tf.reset_default_graph()

    # Setup VGG and initialize it with white noise image that we'll optimize.
    shape = cont_img.shape
    with tf.variable_scope('to_train'):
        noise = tf.random_uniform(shape)*255.0
        X = tf.get_variable('input', dtype=tf.float32, initializer=noise)
    with tf.variable_scope('vgg'):
        vggnet = vgg16.vgg16(X)

    # Get the tensors for content loss features.
    content_layers = utils.get_layers(loss_content_layers)

    # Get the gram matrices' tensors for the style loss features.
    for region in regions:
        guide_channels = utils.propagate_mask(region['content_mask'],
                                              loss_style_layers)
        input_img_grams = utils.get_grams(loss_style_layers, guide_channels)
        region['input_img_grams'] = input_img_grams

    # Get the target content features
    with tf.Session() as sess:
        vggnet.load_weights('libs/vgg16_weights.npz', sess)
        print 'Precomputing target content layers.'
        content_targets = sess.run(content_layers,
                                   feed_dict={'to_train/input:0': cont_img})

    # Create loss function
    cont_loss = losses.content_loss(content_layers, content_targets,
                                    content_weights)
    style_loss = 0
    for region in regions:
        region_style_loss = losses.style_loss(region['input_img_grams'],
                                              region['grams_target'],
                                              style_weights)
        style_loss += region['weight']*region_style_loss
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
