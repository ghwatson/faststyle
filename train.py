"""
Train image transformation network in conjunction with perceptual loss. Save
the image transformation network for later application.

File author: Grant Watson
Date: Jan 2017
"""

import tensorflow as tf
from libs import vgg16
from im_transf_net import create_net

# TODO: Move style image to file argument
# TODO: Ensure that vgg16 has trainable=False
# TODO: Refactor into functions for readability

if __name__ == "__main__":

    # Training hyperparameters
    mscoco_shape = [256, 256, 3]  # We'll compress to this
    batch_size = 4
    n_iter = 40000
    learn_rate = 1e-3
    tv_reg_bounds = [1e-6, 1e-4]
    layers_feat_loss = ['relu2_2']
    layers_style_loss = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']

    # Get vgg to use for perceptual loss network.
    vggnet = vgg16.get_vgg_model()
    gdef_vgg = vggnet['graph_def']

    # Load in image transformation network into default graph.
    shape = [batch_size] + mscoco_shape
    gdef_img_t = create_net(shape).as_graph_def()

    # Conjoin graphs into 1 graph.
    with tf.Graph().as_default() as g:
        X = tf.placeholder(tf.float32, shape=shape, name="input")
        img_t_out = tf.import_graph_def(gdef_img_t,
                                        input_map={'input:0': X},
                                        return_elements=['output:0'],
                                        name='img_t_net')
        vgg_out = tf.import_graph_def(gdef_vgg,
                                      input_map={'images:0': img_t_out},
                                      name='vgg_net')

    print gdef_vgg

    # Define loss for optimization. We want to just train the image
    # transformation network, and not VGG16.
    varss = [op.outputs[0] for op in tf.get_default_graph().get_operations() if
    op.type == "Variable"]
    print varss
    #with tf.Session(graph=g) as sess:
        #train_vars = tf.trainable_variables()
        #print train_vars

    # Prime MS-Coco dataqueuer.

    # Load in style image.

    # Train.

    # Save the model (the image transformation network) for later usage in
    # predict.py.
