"""
Used to load and apply a trained faststyle model to an image in order to
stylize it.

File author: Grant Watson
Date: Jan 2017
"""

import tensorflow as tf
import numpy as np
from im_transf_net import create_net
import argparse
import utils

# TODO: handle the upsampling thing better. Really, shouldn't need to
# explicitly have to give it.


def setup_parser():
    """Options for command-line input."""
    parser = argparse.ArgumentParser(description="""Use a trained fast style
                                     transfer model to filter an input
                                     image, and save to an output image.""")
    parser.add_argument('--input_img_path_left',
                        help='Input content image that will be stylized.')
    parser.add_argument('--input_img_path_right',
                        help='Input content image that will be stylized.')
    parser.add_argument('--output_img_path',
                        help='Desired output image path.',
                        default='./results/styled.jpg')
    parser.add_argument('--model_path',
                        default='./models/starry_final.ckpt',
                        help='Path to .ckpt for the trained model.')
    parser.add_argument('--content_target_resize',
                        help="""Resize input content image. Useful if having
                        OOM issues.""",
                        default=1.0,
                        type=float)
    parser.add_argument('--upsample_method',
                        help="""The upsample method that was used to construct
                        the model being loaded. Note that if the wrong one is
                        chosen an error will occur.""",
                        choices=['resize', 'deconv'],
                        default='resize')
    return parser


if __name__ == '__main__':

    # Command-line argument parsing.
    parser = setup_parser()
    args = parser.parse_args()
    input_img_path_left = args.input_img_path_left
    input_img_path_right = args.input_img_path_right
    output_img_path = args.output_img_path
    model_path = args.model_path
    upsample_method = args.upsample_method
    content_target_resize = args.content_target_resize

    # Read + preprocess input image.
    img_l = utils.imread(input_img_path_left)
    img_l = utils.imresize(img_l, content_target_resize)
    img_4d_l = img_l[np.newaxis, :]
    img_r = utils.imread(input_img_path_right)
    img_r = utils.imresize(img_r, content_target_resize)
    img_4d_r = img_r[np.newaxis, :]
    print img_4d_l.shape
    print img_4d_r.shape
    # TODO: in future we input stereo images into this code.

    # Create the graph.
    with tf.variable_scope('img_t_net'):
        Xl = tf.placeholder(tf.float32, shape=img_4d_l.shape, name='input')
        Xr = tf.placeholder(tf.float32, shape=img_4d_r.shape, name='input')
        X = tf.concat([Xl, Xr], 3)
        Y = create_net(X, upsample_method, stereo=True)

    # Saver used to restore the model to the session.
    saver = tf.train.Saver()

    # Filter the input image.
    with tf.Session() as sess:
        print 'Loading up model...'
        saver.restore(sess, model_path)
        print 'Evaluating...'
        stereo_out = sess.run(Y, feed_dict={Xl: img_4d_l, Xr: img_4d_r})

    # Postprocess + save the output image.
    print 'Saving stereo image.'
    stereo_out = np.squeeze(stereo_out)
    l_out = stereo_out[:, :, 0:3]
    r_out = stereo_out[:, :, 3:6]
    img_out = np.hstack([l_out, r_out])
    utils.imwrite(output_img_path, img_out)

    print 'Done.'
