import numpy as np


def scale_formula(k, m=5, s_min=0.2, s_max=0.9):
    """ Scale formula.

    Args:
        k: K-th feature map level.
        m: Number of feature map levels.
        s_min: Scale factor of the lowest layer.
        s_max: Scale factor of the highest layer.

    Returns: Scale value for the k-th feature map.

    """
    return s_min + ((s_max - s_min) / (m - 1)) * (k - 1)


def generate_anchors_for_batch(batch_size, anchors):
    """ Generate default boxes for a batch.

    Args:
        batch_size: Number of items in a batch.
        anchors: All default boxes generated from all feature maps.

    Returns: With shape (batch_size, anchors.shape).

    """
    boxes = [np.expand_dims(anchors, axis=0) for _ in range(batch_size)]

    return np.concatenate(boxes, axis=0)


def generate_anchors(bases=None, aspect_ratios=None, feature_maps=None,
                     strides=None, img_shape=512):
    """ Generate default boxes.

    Args:
        bases: Factor to scale default boxes.
        aspect_ratios: Contains aspect ratios to use for each location
        in a feature map.
        feature_maps: List of feature map size.
        strides: Stride values.
        img_shape: Image resolution.

    Returns:

    """
    if aspect_ratios is None:
        aspect_ratios = [.5, 1., 2.]
    if bases is None:
        bases = [32., 64., 128., 256., 512.]

    if feature_maps is None:
        feature_maps = compute_feature_map_shapes(img_shape)
    if strides is None:
        strides = compute_feature_map_strides(feature_maps, img_shape)

    n = len(feature_maps)
    scales = scale_formula(np.arange(n) + 1)

    boxes = np.zeros((0, 4), dtype=np.float32)

    for i in range(n):
        width, height = compute_width_height_of_boxes(bases[i], aspect_ratios,
                                                      scales)
        shifted_boxes = place_anchors(feature_maps[i], strides[i], width,
                                      height)
        boxes = np.append(boxes, shifted_boxes, axis=0)

    return boxes


def place_anchors(size, stride, width, height):
    """ Generate default boxes for a feature map.

    Args:
        size: Feature map resolution.
        stride: Value to stride across the grid.
        width: Width of default boxes for this feature map.
        height: Height of default boxes for this feature map.

    Returns: With shape (size * size * num_boxes, 4).

    """
    num_boxes = width.shape[0]

    width = np.tile(width, size * size)
    height = np.tile(height, size * size)

    center_x = np.arange(0 + 0.5, size) * stride
    center_x = np.tile(center_x, size)
    center_x = np.repeat(center_x, num_boxes)

    center_y = np.arange(0 + 0.5, size) * stride
    center_y = np.repeat(center_y, size)
    center_y = np.repeat(center_y, num_boxes)

    boxes = np.stack((center_x, center_y, width, height), axis=1)

    return transform_box_format(boxes)


def transform_box_format(boxes):
    """ Change saving format of boxes.

    Args:
        boxes: Boxes with center, width, height coordinates.

    Returns: Transformed boxes with format (x_min, y_min, x_max, y_max).

    """
    x_min = boxes[:, 0] - (boxes[:, 2] / 2.0)
    y_min = boxes[:, 1] - (boxes[:, 3] / 2.0)
    x_max = boxes[:, 0] + (boxes[:, 2] / 2.0)
    y_max = boxes[:, 1] + (boxes[:, 3] / 2.0)

    return np.stack((x_min, y_min, x_max, y_max), axis=1)


def compute_width_height_of_boxes(base, aspect_ratios, scales):
    """ Create default boxes for a feature map. Number of boxes corresponds
    to len(aspect_ratios).

    Args:
        base: Scale boxes with base.
        aspect_ratios: List of aspect ratios. Will be used for each location
        in a feature map.
        scales: Scale value for a feature map.

    Returns: Width and height of boxes.

    """
    multiplier = base * np.tile(scales, len(aspect_ratios))

    width = multiplier / np.repeat(np.sqrt(aspect_ratios), len(scales))
    height = multiplier * np.repeat(np.sqrt(aspect_ratios), len(scales))

    return width, height


def compute_feature_map_shapes(image_shape=512,
                               divisor=np.array([3, 4, 5, 6, 7])):
    """ Compute feature map sizes.

    Args:
        image_shape: Image resolution.
        divisor: Divide to get feature map shapes. Setting this list depends
        on feature map architecture.

    Returns: Feature map shapes.

    """
    return image_shape // (2 ** divisor)


def compute_feature_map_strides(feature_maps, image_shape=512):
    """ Compute strides.

    Args:
        feature_maps: List of feature map shapes.
        image_shape: Image resolution.

    Returns: Stride values corresponding to feature maps.

    """
    image_shape = np.ones_like(feature_maps) * image_shape
    return image_shape // feature_maps
