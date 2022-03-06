import tensorflow as tf

from numpy.random import seed
from tensorflow.keras.layers import (Conv2D, UpSampling2D, MaxPooling2D,
                                     DepthwiseConv2D, ReLU, BatchNormalization)

tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

seed(1)
tf.random.set_seed(2)


def conv_layer(x, num_filters, kernel_size=1, strides=1):
    """ Use convolution layer.
    """
    return Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides,
                  padding="same", use_bias=True)(x)


def top_down_fusion(p_in, p_top_down, kernel_size=3, strides=1,
                    fuse_nodes=True):
    """ Fuse two incoming nodes on the top down path of the feature network.
    """
    if fuse_nodes:
        p_in = depth_block(p_in, kernel_size, strides)

    return up_sampling(p_in, size=2) + p_top_down


def up_sampling(x, size=2):
    """ Up-sample.
    """
    return UpSampling2D(size=size, interpolation="nearest")(x)


def down_sampling(x, size=2):
    """ Down-sample with max pooling.
    """
    return MaxPooling2D(pool_size=2, strides=size, padding="same")(x)


def bottom_up_fusion(p_bottom_up, p_in, p_in_prime=0.0, kernel_size=3,
                     strides=1):
    """ Fuse three incoming nodes on the bottom up path of the feature network.
    """
    p_bottom_up = depth_block(p_bottom_up, kernel_size, strides)
    p_bottom_up = down_sampling(p_bottom_up, size=2)

    return p_bottom_up + p_in_prime + p_in


def depth_wise_conv(x, kernel_size=3, strides=1):
    """ Use depth-wise convolution.
    """
    return DepthwiseConv2D(kernel_size=kernel_size, strides=strides,
                           padding="same", use_bias=False)(x)


def depth_block(x, kernel_size=3, strides=1):
    """ Combine depth-wise convolution, batch normalization and relu.
    """
    x = depth_wise_conv(x, kernel_size, strides)
    x = BatchNormalization(axis=3, trainable=False)(x)

    return ReLU()(x)


def create_feature_network(backbone_layers, depth=3, num_features=64):
    """ Create feature pyramid network.

    Args:
        backbone_layers: Use EfficientNetB0 layers.
        num_features: Num channels to project.
        depth: Number of repeats.

    Returns: List of feature maps with different feature map size.

    """
    for i in range(depth):
        if i == 0:
            _, _, p3_in, p4_in, p5_in = backbone_layers
            p3_in = conv_layer(p3_in, num_features)
            p4_in = conv_layer(p4_in, num_features)
            p5_in = conv_layer(p5_in, num_features)
            p6_in = conv_layer(p5_in, num_features, kernel_size=3, strides=2)
            p7_in = conv_layer(p6_in, num_features, kernel_size=3, strides=2)
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = backbone_layers
            p3_in = conv_layer(p3_in, num_features)
            p4_in = conv_layer(p4_in, num_features)
            p5_in = conv_layer(p5_in, num_features)
            p6_in = conv_layer(p6_in, num_features)
            p7_in = conv_layer(p7_in, num_features)

        p6_prime = top_down_fusion(p7_in, p6_in)
        p5_prime = top_down_fusion(p6_prime, p5_in)
        p4_prime = top_down_fusion(p5_prime, p4_in)
        p3_out = top_down_fusion(p4_prime, p3_in)

        p4_out = bottom_up_fusion(p3_out, p4_in, p4_prime)
        p5_out = bottom_up_fusion(p4_out, p5_in, p5_prime)
        p6_out = bottom_up_fusion(p5_out, p6_in, p6_prime)
        p7_out = bottom_up_fusion(p6_out, p7_in)
        p7_out = depth_block(p7_out)

        backbone_layers = p3_out, p4_out, p5_out, p6_out, p7_out

    return backbone_layers
