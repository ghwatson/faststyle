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
import datapipe_stereo
import os
import argparse
import utils
import losses

# TODO: implement conditional default in argparse for beta. Depends on
# upsampling method.

# TODO: debugging
from PIL import Image
import cv2
from tensorflow.contrib.resampler import resampler


def setup_parser():
    """Used to interface with the command-line."""
    parser = argparse.ArgumentParser(
                description='Train a style transfer net.')
    parser.add_argument('--train_dir',
                        help='Directory of TFRecords training data.')
    # TODO: remove default
    parser.add_argument('--train_stereo_dir',
                        help='Directory of TFRecords training data.',
                        default='/mnt/77D2D4A40B3D7BC7/tf_stereo')
    parser.add_argument('--model_name',
                        help='Name of model being trained.')
    parser.add_argument('--pretrain_path',
                        default=None,
                        help='Path to style target image.')
    parser.add_argument('--style_img_path',
                        default='./style_images/starry_night_crop.jpg',
                        help='Path to style target image.')
    parser.add_argument('--learn_rate',
                        help='Learning rate for Adam optimizer.',
                        default=1e-3, type=float)
    parser.add_argument('--batch_size',
                        help='Batch size for training.',
                        default=4, type=int)
    parser.add_argument('--batch_stereo_size',
                        help='Batch stereo size for training.',
                        default=2, type=int)
    # TODO: hack here from 2 to 4 since splitting batch across two eyes.
    parser.add_argument('--n_epochs',
                        help='Number of training epochs.',
                        default=4, type=int)
                        # default=2, type=int)
    parser.add_argument('--n_epochs_stereo',
                        help='Number of training epochs for stereo data.',
                        default=10, type=int)
                        # default=2, type=int)
    parser.add_argument('--preprocess_size',
                        help="""Dimensions to resize training images to before passing
                        them into the image transformation network.""",
                        default=[256, 256], nargs=2, type=int)
    parser.add_argument('--preprocess_stereo_size',
                        help="""Same as preprocess_size but for stereo training
                        data.""",
                        default=[436, 1024], nargs=2, type=int)
    #TODO: hack above to get power of 2...fix this?
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
    # TODO: search over this hyperparameter. Johnson with temporal loss?
    parser.add_argument('--disparity_weight',
                        help="""disparity loss weight.""",
                        default=1.0,
                        type=float)
    parser.add_argument('--num_steps_ckpt',
                        help="""Save a checkpoint everytime this number of
                        steps passes in training.""",
                        default=1000,
                        type=int)
    # TODO: separate buffer for stereo training?
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
    parser.add_argument('--num_steps_break_stereo',
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
    train_stereo_dir = args.train_stereo_dir
    style_img_path = args.style_img_path
    pretrain_path = args.pretrain_path
    model_name = args.model_name
    preprocess_size = args.preprocess_size
    preprocess_stereo_size = args.preprocess_stereo_size
    batch_size = args.batch_size
    batch_stereo_size = args.batch_stereo_size
    n_epochs = args.n_epochs
    n_epochs_stereo = args.n_epochs_stereo
    run_name = args.run_name
    learn_rate = args.learn_rate
    loss_content_layers = args.loss_content_layers
    loss_style_layers = args.loss_style_layers
    content_weights = args.content_weights
    style_weights = args.style_weights
    disparity_weight = args.disparity_weight
    num_steps_ckpt = args.num_steps_ckpt
    num_pipe_buffer = args.num_pipe_buffer
    num_steps_break = args.num_steps_break
    num_steps_break_stereo = args.num_steps_break_stereo
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
    # TODO: fix this.
    # shape = [batch_size] + preprocess_size + [3]
    shape = [batch_stereo_size] + preprocess_stereo_size + [3]
    with tf.variable_scope('img_t_net'):
        Xl = tf.placeholder(tf.float32, shape=shape, name='input_left')
        Xr = tf.placeholder(tf.float32, shape=shape, name='input_right')
        X = tf.concat([Xl, Xr], 3)
        Y = create_net(X, upsample_method, stereo=True)
        # we take batches over left and right, then concatenate along the batch
        # dimension so that we can pass it all into vgg.
        Yl = Y[:, :, :, 0:3]
        Yr = Y[:, :, :, 3:6]
        Y = tf.concat([Yl, Yr], 0)

    # Connect vgg directly to the image transformation network.
    with tf.variable_scope('vgg'):
        vggnet = vgg16.vgg16(Y)

    # Get the gram matrices' tensors for the style loss features.
    input_img_grams = utils.get_grams(loss_style_layers)
    input_img_grams_l, input_img_grams_r = [], []
    # unwrap to left right eye. (2B -> 1B)
    for gram in input_img_grams:
        gram_l, gram_r = tf.split(gram, 2)
        input_img_grams_l.append(gram_l)
        input_img_grams_r.append(gram_r)

    # Get the tensors for content loss features.
    content_layers = utils.get_layers(loss_content_layers)
    content_layers_l, content_layers_r = [], []
    for layer in content_layers:
        layer_l, layer_r = tf.split(layer, 2)
        content_layers_l.append(layer_l)
        content_layers_r.append(layer_r)

    # Create content loss functions
    content_targets_l = tuple(tf.placeholder(tf.float32,
                              shape=layer.get_shape(),
                              name='content_input_l_{}'.format(i))
                              for i, layer in enumerate(content_layers_l))
    cont_loss_l = losses.content_loss(content_layers_l, content_targets_l,
                                      content_weights)
    content_targets_r = tuple(tf.placeholder(tf.float32,
                              shape=layer.get_shape(),
                              name='content_input_r_{}'.format(i))
                              for i, layer in enumerate(content_layers_r))
    cont_loss_r = losses.content_loss(content_layers_r, content_targets_r,
                                      content_weights)

    # Create style loss functions
    # TODO: remove this maybe. debugging.
    # blind_l = tf.placeholder(tf.float32, name='blind_l')
    # blind_r = tf.placeholder(tf.float32, name='blind_r')
    style_loss_l = losses.style_loss(input_img_grams_l, target_grams,
                                     style_weights)
    style_loss_r = losses.style_loss(input_img_grams_r, target_grams,
                                     style_weights)

    # Create disparity loss function
    st_shape = [batch_stereo_size] + preprocess_stereo_size
    target_warps = tf.placeholder(tf.float32,
                                  shape=st_shape + [2],
                                  name='disparity_input')
    target_occlusions = tf.placeholder(tf.float32,
                                       shape=st_shape + [1],
                                       name='occlusion_input')
    target_outofframes = tf.placeholder(tf.float32,
                                        shape=st_shape + [1],
                                        name='outofframes_input')
    target_mask = (1.-target_occlusions/255.0)*(1.-target_outofframes/255.0)
    disparity_loss = losses.disparity_loss(Yl, Yr, target_warps,
                                           disparity_weight,
                                           target_mask)
    # TODO: debugging
    prewarp = target_mask*resampler(Xr, target_warps, name='warped')


    # pixel_loss_l = losses.pixel_loss(Yl, np.zeros(Yl.shape))
    # pixel_loss_r = losses.pixel_loss(Yr, np.zeros(Yr.shape))
    # tv_loss = losses.tv_loss(Y)
    # beta = tf.placeholder(tf.float32, shape=[], name='tv_scale')
    # loss = cont_loss + style_loss_l + blind_weight*style_loss_r + beta*tv_loss
    # loss = blind_l*(cont_loss_l + style_loss_l + pixel_loss_r) + \
            # blind_r*(cont_loss_r + style_loss_r + pixel_loss_l)

    # Construct total loss
    loss = cont_loss_l + style_loss_l + cont_loss_r + style_loss_r
    loss_stereo = loss + disparity_loss
    summs = []
    with tf.name_scope('summaries'):
        # TODO: neaten this up.
        summ = tf.summary.scalar('loss', loss)
        summs.append(summ)
        summ = tf.summary.scalar('style_loss', style_loss_l+style_loss_r)
        summs.append(summ)
        summ = tf.summary.scalar('content_loss', cont_loss_l+cont_loss_r)
        summs.append(summ)
        summ_disp = tf.summary.scalar('disparity_loss', disparity_loss)
        # tf.summary.scalar('tv_loss', beta*tv_loss)

    # Setup input pipeline (delegate it to CPU to let GPU handle neural net)
    files = tf.train.match_filenames_once(train_dir + '/train-*')
    files_stereo = tf.train.match_filenames_once(train_stereo_dir + '/train-*')
    with tf.variable_scope('input_pipe'), tf.device('/cpu:0'):
        batch_op = datapipe.batcher(files, batch_size, preprocess_size,
                                    n_epochs, num_pipe_buffer)
        batch_stereo_op = datapipe_stereo.batcher(files_stereo,
                                                  batch_stereo_size,
                                                  preprocess_stereo_size,
                                                  n_epochs_stereo,
                                                  num_pipe_buffer)

    # We do not want to train VGG, so we must grab the subset.
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope='img_t_net')

    # Setup step + optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss, global_step,
                                                            train_vars)
    optimizer_stereo = tf.train.AdamOptimizer(learn_rate).minimize(loss_stereo, global_step,
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
    merged = tf.summary.merge(summs)
    merged_stereo = tf.summary.merge(summs + [summ_disp])
    full_log_path = './summaries/train/' + run_name
    train_writer = tf.summary.FileWriter(full_log_path)

    # We must include local variables because of batch pipeline.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Begin training.
    fetch = [merged, optimizer, loss]
    print 'Starting training...'
    # zero_mask = np.zeros(batch_op.shape)
    with tf.Session() as sess:
        # Initialization
        sess.run(init_op)
        vggnet.load_weights('libs/vgg16_weights.npz', sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        if pretrain_path is None:
            # We first do pretraining.
            print 'Starting pretraining...'

            try:
                print coord.should_stop()
                while not coord.should_stop():
                    current_step = sess.run(global_step)
                    # TODO: in future grab single stereo batch?
                    batch_l = sess.run(batch_op)
                    batch_r = sess.run(batch_op)
                    # print batch_l.shape

                    # TODO: debugging stuff below
                    # if (current_step % 2) == 0:
                        # # mask left eye to zero.
                        # batch_l = batch_l*zero_mask
                        # blind_l_val = 0
                        # blind_r_val = 1
                    # else:
                        # # mask right eye to zero.
                        # batch_r = batch_r*zero_mask
                        # blind_l_val = 1
                        # blind_r_val = 0

                    # Collect content targets
                    content_data = sess.run(content_layers,
                                            feed_dict={Yl: batch_l,
                                                       Yr: batch_r})
                    content_data_l, content_data_r = [], []
                    for layer in content_data:
                        layer_l, layer_r = np.split(layer, 2)
                        content_data_l.append(layer_l)
                        content_data_r.append(layer_r)

                    feed_dict = {Xl: batch_l, Xr: batch_r,
                                 content_targets_l: content_data_l,
                                 content_targets_r: content_data_r}
                                 # disparity_loss: 0.0}
                                 # target_warps: np.zeros((4,256,256,3)),
                                 # target_occlusions: 0.0,
                                 # target_outofframes: 0.0,
                                 # blind_l: blind_l_val,
                                 # blind_r: blind_r_val}
                                 # beta: beta_val,
                    if (current_step % num_steps_ckpt == 0):
                        # Save a checkpoint
                        save_path = 'training/' + model_name + '.ckpt'
                        saver.save(sess, save_path, global_step=global_step)
                        summary, _, loss_out = sess.run(fetch,
                                                        feed_dict=feed_dict)
                        train_writer.add_summary(summary, current_step)
                        print current_step, loss_out

                    elif (current_step % 10 == 0):
                        # Collect some diagnostic data for Tensorboard.
                        summary, _, loss_out = sess.run(fetch,
                                                        feed_dict=feed_dict)
                        train_writer.add_summary(summary, current_step)

                        # Do some standard output.
                        print current_step, loss_out
                    else:
                        _, loss_out = sess.run(fetch[1:],
                                               feed_dict=feed_dict)

                    # Throw error if we reach number of steps to break after.
                    if current_step == num_steps_break:
                        print('Done pretraining.')
                        break
            except tf.errors.OutOfRangeError:
                print('Done pretraining.')
            finally:
                # Save the model (the image transformation network) for later
                # usage in predict.py
                save_str = 'models/' + model_name + '_prestereo.ckpt'
                final_saver.save(sess, save_str)

        else:
            # Load in the pretrained model.
            print('Loading in pretrained model')
            final_saver.restore(sess, pretrain_path)

        # ---------------------------------------------------------
        # Perform stereo training! --------------------------------
        # ---------------------------------------------------------

        # TODO: debugging
        warp_t = tf.get_default_graph().get_tensor_by_name('warped/Resampler:0')

        print 'starting stereo training'
        fetch = [merged_stereo, optimizer_stereo, loss_stereo]
        fetch_debug = [merged_stereo, optimizer_stereo, loss_stereo,
                       warp_t, Yl, Yr, prewarp, target_mask]

        try:
            while not coord.should_stop():
                current_step = sess.run(global_step)

                # TODO: move squeeze to tensors?
                batch = sess.run(batch_stereo_op)
                #batch_l = batch[:, 0, :, :, :]
                #batch_r = batch[:, 1, :, :, :]
                #batch_d = batch[:, 2, :, :, :]
                #batch_occ = batch[:, 3, :, :, :]
                #batch_oof = batch[:, 4, :, :, :]
                batch_l, batch_r, batch_d, batch_occ, batch_oof = batch

                # Collect content targets
                content_data = sess.run(content_layers,
                                        feed_dict={Yl: batch_l, Yr: batch_r})
                content_data_l, content_data_r = [], []
                for layer in content_data:
                    layer_l, layer_r = np.split(layer, 2)
                    content_data_l.append(layer_l)
                    content_data_r.append(layer_r)

                feed_dict = {Xl: batch_l, Xr: batch_r,
                             content_targets_l: content_data_l,
                             content_targets_r: content_data_r,
                             target_warps: batch_d,
                             target_occlusions: batch_occ,
                             target_outofframes: batch_oof}

                # TODO: debugging
                loss_out, warp_v, l_v, r_v, prewarp_v, m_v = sess.run(fetch_debug[2:],
                                       feed_dict=feed_dict)
                print warp_v.shape
                viz = np.concatenate([warp_v, l_v, r_v, batch_l, batch_r,
                                      prewarp_v],
                                axis=1)
                utils.imwrite('./viz.png', viz[0])
                print m_v.shape
                print batch_occ.shape
                print np.unique(batch_occ[0])
                utils.imwrite_greyscale('./viz_mask.png',batch_occ[0])
                # img = Image.fromarray(viz[0].astype('uint8'))
                # img.show()
                raise tf.errors.OutOfRangeError

                if (current_step % num_steps_ckpt == 0):
                    # Save a checkpoint
                    save_path = 'training/' + model_name + '_poststereo.ckpt'
                    saver.save(sess, save_path, global_step=global_step)
                    summary, _, loss_out = sess.run(fetch,
                                                    feed_dict=feed_dict)
                    train_writer.add_summary(summary, current_step)
                    print current_step, loss_out

                elif (current_step % 10 == 0):
                    # Collect some diagnostic data for Tensorboard.
                    summary, _, loss_out = sess.run(fetch,
                                                    feed_dict=feed_dict)
                    train_writer.add_summary(summary, current_step)

                    # Do some standard output.
                    print current_step, loss_out
                else:
                    # _, loss_out = sess.run(fetch[1:],
                                           # feed_dict=feed_dict)
                    # TODO: debugging
                    _, loss_out, warp_v, l_v, r_v, prewarp_v, m_v = sess.run(fetch_debug[1:],
                                           feed_dict=feed_dict)
                    print warp_v.shape
                    viz = np.concatenate([warp_v, l_v, r_v, batch_l, batch_r,
                                          prewarp_v],
                                    axis=1)
                    utils.imwrite('./viz.png', viz[0])
                    print m_v.shape
                    print np.unique(m_v)
                    utils.imwrite('./viz_mask.png', m_v[0])
                    # img = Image.fromarray(viz[0].astype('uint8'))
                    # img.show()
                    raise tf.errors.OutOfRangeError

                # Throw error if we reach number of steps to break after.
                if current_step == num_steps_break_stereo:
                    print('Done stereo training.')
                    break
        except tf.errors.OutOfRangeError:
            print('Done stereo training.')
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
