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

# TODO: Move style image to file argument
# TODO: Ensure that vgg16 has trainable=False
# TODO: Refactor into functions for readability
# TODO: Add in GPU (g.device) when we push thru ssh to home gpu computer

if __name__ == "__main__":

    # Training hyperparameters
    mscoco_shape = [256, 256, 3]  # We'll compress to this
    batch_size = 2  # TODO: change back to 4
    n_iter = 40000
    learn_rate = 1e-3
    tv_reg_bounds = [1e-6, 1e-4]
    layers_feat_loss = ['relu2_2']
    layers_style_loss = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']

    # Load in image transformation network into default graph.
    shape = [batch_size] + mscoco_shape
    with tf.variable_scope('img_t_net'):
        img_t_out = create_net(shape)

    # Get vgg to use for perceptual loss network.
    vggnet = vgg16.vgg16(img_t_out)

    # Get the input
    g = tf.get_default_graph()
    X = g.get_tensor_by_name('img_t_net/input:0')

    # TODO: Remove this.
    print img_t_out.get_shape().as_list()
    img = np.zeros((256, 256, 3))
    img *= 0.0
    img[0:128, :, 0] = 1.0
    img = img[np.newaxis, :]
    img2 = np.zeros((256, 256, 3))
    img2[0:128, :, 0] = 0.5
    img2 = img2[np.newaxis, :]
    img = np.append(img, img2, axis=0)
    print 'Running session'
    with tf.Session() as sess:
            vggnet.load_weights('libs/vgg16_weights.npz', sess)
            sess.run(tf.initialize_all_variables())
            #out = sess.run(vggnet.probs, feed_dict={X:img})

    # Prime MS-Coco dataqueuer.

    # Load in style image.

    # Train.

    # Save the model (the image transformation network) for later usage in
    # predict.py.
