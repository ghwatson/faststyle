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
    parser.add_argument('--input_img_path',
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
    input_img_path = args.input_img_path
    output_img_path = args.output_img_path
    model_path = args.model_path
    upsample_method = args.upsample_method
    content_target_resize = args.content_target_resize

    # Read + preprocess input image.
    img = utils.imread(input_img_path)
    img = utils.imresize(img, content_target_resize)
    img_4d = img[np.newaxis, :]

    # Create the graph.
    with tf.variable_scope('img_t_net'):
        X = tf.placeholder(tf.float32, shape=img_4d.shape, name='input')
        Y = create_net(X, upsample_method)

    # Saver used to restore the model to the session.
    saver = tf.train.Saver()

    # Filter the input image.
    with tf.Session() as sess:
        print 'Loading up model...'
        saver.restore(sess, model_path)
        print 'Evaluating...'
        img_out = sess.run(Y, feed_dict={X: img_4d})

    # Postprocess + save the output image.
    print 'Saving image.'
    img_out = np.squeeze(img_out)
    utils.imwrite(output_img_path, img_out)

    print 'Done.'
