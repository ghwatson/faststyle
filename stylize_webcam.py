"""Use a model to stylize an OpenCV webcam feed. Not for use with spatially
controlled models.

File author: Grant Watson
Date: Feb 2017
"""

import cv2
import tensorflow as tf
from im_transf_net import create_net
import numpy as np
import argparse

# TODO: feed appropriate fps to writer.
# TODO: mask detector such as fcn?


def setup_parser():
    """Options for command-line input."""
    parser = argparse.ArgumentParser(description="""Use a trained fast style
                                     transfer model to filter webcam feed.
                                     Saves to video if --out_path is
                                     provided.""")
    parser.add_argument('--model_path',
                        default='./models/starry_final.ckpt',
                        help='Path to .ckpt for the trained model.')
    parser.add_argument('--out_path',
                        default=None,
                        help="""Path to save webcam feed to. For example:
                        ./output.avi""")
    parser.add_argument('--upsample_method',
                        help="""The upsample method that was used to construct
                        the model being loaded. Note that if the wrong one is
                        chosen an error will be thrown.""",
                        choices=['resize', 'deconv'],
                        default='resize')
    parser.add_argument('--resolution',
                        help="""Dimensions for webcam. Note that, depending on
                        the webcam, only certain resolutions will be possible.
                        Leave this argument blank if want to use default
                        resolution.""",
                        nargs=2,
                        type=int,
                        default=None)
    return parser


if __name__ == '__main__':

    # Command-line argument parsing.
    parser = setup_parser()
    args = parser.parse_args()
    model_path = args.model_path
    out_path = args.out_path
    upsample_method = args.upsample_method
    resolution = args.resolution

    # Instantiate video capture object.
    cap = cv2.VideoCapture(0)

    # Set resolution
    if resolution is not None:
        x_length, y_length = resolution
        cap.set(3, x_length)  # 3 and 4 are OpenCV property IDs.
        cap.set(4, y_length)
    x_new = int(cap.get(3))
    y_new = int(cap.get(4))
    print 'Resolution is: {0} by {1}'.format(x_new, y_new)

    # Create the graph.
    shape = [1, y_new, x_new, 3]
    with tf.variable_scope('img_t_net'):
        X = tf.placeholder(tf.float32, shape=shape, name='input')
        Y = create_net(X, upsample_method)

    # Saver used to restore the model to the session.
    saver = tf.train.Saver()

    # Instantiate a Writer to save the video.
    if out_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_path, fourcc, 15.0, (x_new, y_new))

    # Begin filtering.
    with tf.Session() as sess:
        print 'Loading up model...'
        saver.restore(sess, model_path)
        print 'Begin filtering...'
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Make frame 4-D
            img_4d = frame[np.newaxis, :]

            # Our operations on the frame come here
            img_out = sess.run(Y, feed_dict={X: img_4d})
            img_out = np.squeeze(img_out).astype(np.uint8)
            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

            # Write frame to file.
            if out_path is not None:
                out.write(img_out)

            # Put the FPS on it.
            # img_out = cv2.putText(img_out, 'fps: {}'.format(fps), (50, 50),
                                  # cv2.FONT_HERSHEY_SIMPLEX,
                                  # 1.0, (255, 0, 0), 3)

            # Display the resulting frame
            cv2.imshow('frame', img_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    if out_path is not None:
        out.release()
    cv2.destroyAllWindows()
