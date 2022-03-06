import math


def round_filters(width_coefficient, filters, depth_divisor=8.0):
    """ Round number of filters based on width coefficient.

    Args:
        width_coefficient: Coefficient to scale network width.
        filters: Number of filters for a given layer.
        depth_divisor: Constant.

    From tensorflow implementation:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet
    /efficientnet_model.py
    """
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2)
    new_filters = new_filters // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)

    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor

    return int(new_filters)


def round_repeats(depth_coefficient, repeats):
    """ Round number of filters based on depth coefficient.

    Args:
        depth_coefficient: Coefficient to scale number of repeats.
        repeats: Number to repeat mb_conv_block.

    From tensorflow implementation:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet
    /efficientnet_model.py

    """
    return int(math.ceil(depth_coefficient * repeats))
