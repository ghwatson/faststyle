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
    parser.add_argument('--content_mask_paths',
                        help="""These are paths to image masks with greyscale
                        values in the range [0,1]. These specify which regions
                        of the content image we want to have affected by the
                        styles encoded within the trained model.""",
                        nargs='*',
                        default=['OPEN'])
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
    content_mask_paths = args.content_mask_paths
    with_spatial_control = content_mask_paths != ['OPEN']

    # Read + preprocess input image.
    img = utils.imread(input_img_path)
    img = utils.imresize(img, content_target_resize)
    img_4d = img[np.newaxis, :].astype(np.float32)

    # Read + preprocess + concat (to img_4D) content masks.
    print content_mask_paths
    if with_spatial_control:
        for path in content_mask_paths:
            if path == 'OPEN':
                mask = np.ones(img.shape[0:2]).astype(np.float32)
            elif path == 'CLOSED':
                mask = np.zeros(img.shape[0:2]).astype(np.float32)
            else:
                mask = utils.imread(path, 0)

            # Add the mask channel to image.
            mask = mask[np.newaxis, :, :, np.newaxis]
            img_4d = np.concatenate([img_4d, mask], 3)

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
