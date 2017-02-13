""" 
Used to load and apply a trained faststyle model to an image in order to
stylize it.

Note: The name predict.py is a bit of a misnomer since were not really
predicting anything. More inferring.

File author: Grant Watson
Date: Jan 2017
"""

# TODO: reduce the package dependency by just relying on one library.

import tensorflow as tf
import numpy as np
from im_transf_net import create_net
import cv2
from matplotlib import pyplot as plt
import argparse


def setup_parser():
    """Options for command-line input."""
    parser = argparse.ArgumentParser(description="""Use a trained fast style
                                     transfer model to filter an input
                                     image, and save to an output image.""")
    parser.add_argument('--input_img_path',
                        help='Input content image that will be stylized.',
                        required=True)
    parser.add_argument('--output_img_path',
                        help='Desired output image path.',
                        default='./out.png')
    parser.add_argument('--model_path',
                        help='Path to .ckpt for the trained model.')
    return parser


if __name__ == '__main__':

    # Command-line argument parsing.
    parser = setup_parser()
    args = parser.parse_args()
    input_img_path = args.input_img_path
    output_img_path = args.output_img_path
    model_path = args.model_path

    # Read + format input image.
    img = plt.imread(input_img_path)
    img_4d = img[np.newaxis, :]

    # Create the graph.
    shape = img_4d.shape
    with tf.variable_scope('img_t_net'):
        out = create_net(shape)

    saver = tf.train.Saver()

    # Filter the input image.
    g = tf.get_default_graph()
    X = g.get_tensor_by_name('img_t_net/input:0')
    Y = g.get_tensor_by_name('img_t_net/output:0')
    with tf.Session() as sess:
        print 'Loading up model...'
        saver.restore(sess, model_path)
        print 'Evaluating...'
        img_out = sess.run(Y, feed_dict={X: img_4d})

    # Save the output image.
    print 'Saving image.'
    img_out = np.squeeze(img_out)
    img_out = np.stack([img_out[:, :, 2],
                        img_out[:, :, 1],
                        img_out[:, :, 0]], 2)
    cv2.imwrite(output_img_path, img_out)

    print 'Done.'
