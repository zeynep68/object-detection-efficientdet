from tensorflow.keras import Model, Input
from tensorflow.keras.initializers import RandomNormal, VarianceScaling
from tensorflow.keras.layers import (Conv2D, Dense, BatchNormalization,
                                     Activation, DepthwiseConv2D,
                                     GlobalAveragePooling2D, Dropout,
                                     Multiply, Add, ReLU, Reshape)

from efficient_det.utils.scale_model import round_filters, round_repeats


def mb_conv_block(x_in, in_channel, squeeze, kernel_size, strides,
                  activation_func=ReLU, se_ratio=0.25, skip_conv=False,
                  use_se=True):
    """ Create inverted residual block aka MBConv Block.

    Args:
        x_in: Input to feed into mb_conv_block.
        in_channel: Number of input filters.
        squeeze: Number of output filters.
        kernel_size: Kernel size.
        strides: Stride value.
        se_ratio: Squeeze and excitation ratio.
        activation_func: Activation function.
        skip_conv: To skip first conv layer or not.
        use_se: To use squeeze and excitation or not.

    Returns: Network output of this block.

    """

    initializer = VarianceScaling(scale=2.0, mode="fan_out",
                                  distribution="normal")
    if not skip_conv:
        x = Conv2D(filters=in_channel * 6, kernel_size=1, strides=1,
                   padding="same", use_bias=False,
                   kernel_initializer=initializer)(x_in)
        x = BatchNormalization(axis=3, trainable=False)(x)
        x = activation_func()(x)

    else:
        x = x_in

    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides,
                        padding="same", use_bias=False,
                        kernel_initializer=initializer)(x)
    x = BatchNormalization(axis=3, trainable=False)(x)
    x = activation_func()(x)

    if use_se:
        num_channel = in_channel * 6 if not skip_conv else in_channel

        x_se = GlobalAveragePooling2D()(x)
        x_se = Reshape((1, 1, num_channel))(x_se)
        x_se = Conv2D(max(int(in_channel * se_ratio), 1), padding="same",
                      kernel_size=1, kernel_initializer=initializer)(x_se)
        x_se = activation_func()(x_se)
        x_se = Conv2D(num_channel, kernel_size=1, padding="same",
                      kernel_initializer=initializer)(x_se)
        x_se = Activation("sigmoid")(x_se)
        x = Multiply()([x, x_se])

    x = Conv2D(filters=squeeze, kernel_size=1, strides=1, padding="same",
               use_bias=False, kernel_initializer=initializer)(x)
    x = BatchNormalization(axis=3, trainable=False)(x)

    if (in_channel == squeeze) and (strides == 1):
        return Add()([x, x_in])

    return x


def mb_conv_sequence(x, n, in_channel, squeeze, kernel_size, strides,
                     activation_func=ReLU, se_ratio=0.25, skip_conv=False,
                     use_se=True):
    """ Repeat mb_conv_block n times.

    Args:
        x: Input to feed into the mb_conv_block.
        n: Number to repeat mb_conv_block within a stage.
        in_channel: Number of input filters.
        squeeze: Number of output filters.
        kernel_size: Kernel size.
        strides: Stride value.
        activation_func: Activation function.
        se_ratio: Squeeze and excitation ratio.
        skip_conv: To skip first conv layer or not.
        use_se: To use squeeze and excitation or not.

    Returns: Network output of this block.

    """

    for i in range(n):
        x = mb_conv_block(x_in=x, in_channel=in_channel, squeeze=squeeze,
                          kernel_size=kernel_size, strides=strides,
                          skip_conv=skip_conv, se_ratio=se_ratio,
                          activation_func=activation_func, use_se=use_se)
        strides = 1
        in_channel = squeeze

    return x


def create_efficientnet(input_img=Input(shape=(512, 512, 3)), width=1.0,
                        depth=1.0, dropout=0.2, activation_func=ReLU,
                        include_top=False, num_classes=14):
    """ Create EfficientNet model.

    Args:
        input_img: To build model with known shape.
        width: Coefficient to scale number of filters for each stage.
        depth: Coefficient to scale number of repeats within a stage.
        dropout: Dropout rate.
        activation_func: Activation function.
        include_top: Whether to use stage9 or not. Stage9 can be used for
        classification.
        num_classes: Number of labels.

    """
    stage1 = Conv2D(filters=round_filters(32, width), kernel_size=3, strides=2,
                    padding='same', use_bias=False,
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01,
                                                    seed=None))(input_img)
    stage1 = BatchNormalization(axis=3, trainable=False)(stage1)
    stage1 = activation_func()(stage1)

    stage2 = mb_conv_sequence(x=stage1, n=1,
                              in_channel=round_filters(width, 32),
                              squeeze=round_filters(width, 16), kernel_size=3,
                              strides=1, activation_func=activation_func,
                              skip_conv=True)

    stage3 = mb_conv_sequence(x=stage2, n=round_repeats(depth, 2),
                              in_channel=round_filters(width, 16),
                              squeeze=round_filters(width, 24), kernel_size=3,
                              strides=2, activation_func=activation_func)

    stage4 = mb_conv_sequence(x=stage3, n=round_repeats(depth, 2),
                              in_channel=round_filters(width, 24),
                              squeeze=round_filters(width, 40), kernel_size=5,
                              strides=2, activation_func=activation_func)

    stage5 = mb_conv_sequence(x=stage4, n=round_repeats(depth, 3),
                              in_channel=round_filters(width, 40),
                              squeeze=round_filters(width, 80), kernel_size=3,
                              strides=2, activation_func=activation_func)

    stage6 = mb_conv_sequence(x=stage5, n=round_repeats(depth, 3),
                              in_channel=round_filters(width, 80),
                              squeeze=round_filters(width, 112), kernel_size=5,
                              strides=1, activation_func=activation_func)

    stage7 = mb_conv_sequence(x=stage6, n=round_repeats(depth, 4),
                              in_channel=round_filters(width, 112),
                              squeeze=round_filters(width, 192), kernel_size=5,
                              strides=2, activation_func=activation_func)

    stage8 = mb_conv_sequence(x=stage7, n=round_repeats(depth, 1),
                              in_channel=round_filters(width, 192),
                              squeeze=round_filters(width, 320), kernel_size=3,
                              strides=1, activation_func=activation_func)
    if include_top:
        stage9 = Conv2D(filters=round_filters(width, 1280), kernel_size=1,
                        strides=1, padding="same", use_bias=False,
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01,
                                                        seed=None))(stage8)
        stage9 = BatchNormalization(axis=3, trainable=False)(stage9)
        stage9 = activation_func()(stage9)
        stage9 = GlobalAveragePooling2D()(stage9)
        stage9 = Dropout(dropout)(stage9)
        stage9 = Dense(num_classes, activation="softmax")(stage9)

        return Model(inputs=[input_img], outputs=[stage9])

    return [stage1, stage3, stage4, stage6, stage8]
