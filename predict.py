""" 
Used to load and apply a trained faststyle model to an 
image in order to stylize it.

File author: Grant Watson
Date: Jan 2017
"""

# TODO: move inputs into arguments to be passed to file.

import tensorflow as tf
import numpy as np
from im_transf_net import create_net
import cv2


img_path = '/media/ghwatson/STORAGE/data/mocktrain2014/COCO_train2014_000000581198.jpg'
model_path = 'models/starry_night_final.ckpt'
img = cv2.imread(img_path)
img_4d = img[np.newaxis, :]
shape = img_4d.shape

# Create the graph.
with tf.variable_scope('img_t_net'):
    out = create_net(shape)

saver = tf.train.Saver()

# Test it on an input image.
g = tf.get_default_graph()
X = g.get_tensor_by_name('img_t_net/input:0')
Y = g.get_tensor_by_name('img_t_net/output:0')
with tf.Session() as sess:
    print 'Loading up model...'
    saver.restore(sess, model_path)
    print 'Evaluating...'
    img_out = sess.run(Y, feed_dict={X: img_4d})

# Save the output image.
img_out = np.squeeze(img_out)
cv2.imwrite('eval.jpg', img_out)


print 'Done.'
