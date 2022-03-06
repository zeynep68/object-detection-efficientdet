import math
import numpy as np
import tensorflow as tf

from numpy.random import seed
from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Conv2D, Activation, Concatenate,
                                     Reshape, ReLU)

from efficient_det.models.feature_pyramid_network import create_feature_network
from efficient_det.models.efficient_net import create_efficientnet

tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

seed(1)
tf.random.set_seed(2)


def my_init(shape, dtype=None):
    """ Create bias initializer.
    https://github.com/xuannianz/EfficientDet/blob/master/initializers.py
    """
    probability = 0.01

    return np.ones(shape) * -math.log((1 - probability) / probability)


def box_net(feature, width=64, depth=3, num_boxes=15):
    """ Creates head for box prediction.
    
    Args:
        feature: Input to feed into the regression network.
        width: Width of regression network.
        depth: Depth of regression network.
        num_boxes: Number of boxes for a feature map.

    Returns: A Model that predicts regression values for each default box.

    """
    for i in range(depth):
        feature = Conv2D(filters=width, kernel_size=3, strides=1,
                         padding="same", bias_initializer="zeros",
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.01,
                                                         seed=None))(feature)
        feature = ReLU()(feature)

    feature = Conv2D(filters=4 * num_boxes, kernel_size=3, strides=1,
                     padding="same", bias_initializer="zeros",
                     kernel_initializer=RandomNormal(mean=0.0, stddev=0.01,
                                                     seed=None))(feature)
    feature = Reshape((-1, 4))(feature)

    return feature


def class_net(feature, width=64, depth=3, num_boxes=15, num_classes=14):
    """ Create head for class prediction.

    Args:
        feature: Input to feed into the classification network.
        width: Width of classification network.
        depth: Depth of classification network.
        num_boxes: Number of boxes for a feature map.
        num_classes: Number of classes.

    Returns: A Model that predicts classes for each default box.

    """

    for i in range(depth):
        feature = Conv2D(filters=width, kernel_size=3, strides=1,
                         padding="same", bias_initializer="zeros",
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.01,
                                                         seed=None))(feature)
        feature = ReLU()(feature)

    feature = Conv2D(filters=num_boxes * num_classes, kernel_size=3, strides=1,
                     padding="same", bias_initializer=my_init,
                     kernel_initializer=RandomNormal(mean=0.0, stddev=0.01,
                                                     seed=None))(feature)
    feature = Reshape((-1, num_classes))(feature)
    feature = Activation("sigmoid")(feature)

    return feature


def efficientdet(input_shape=(512, 512, 3), enet_params=(1.0, 1.0, 0.2),
                 fpn_params=(3, 64), pred_params=(64, 3), activation_func=ReLU):
    """ Create EfficientDet model.

    Args:
        input_shape: Input image shape.
        enet_params: Parameters to use in EfficientNet.
        fpn_params: Parameters to use in the Bidirectional Feature Pyramid
        Network.
        pred_params: Parameters to use in the prediction heads.
        activation_func: Activation function.

    Returns: A Model for training and evaluation.

    """
    input_img = tf.keras.layers.Input(input_shape)

    backbone_network = create_efficientnet(input_img, *enet_params,
                                           activation_func=activation_func)

    features = create_feature_network(backbone_network, *fpn_params)

    classification = [class_net(feature, *pred_params) for feature in features]
    classification = Concatenate(axis=1, name="classification")(classification)

    regression = [box_net(feature, *pred_params) for feature in features]
    regression = Concatenate(axis=1, name="regression")(regression)

    return Model(inputs=[input_img], outputs=[regression, classification])
