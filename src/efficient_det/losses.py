import tensorflow as tf


def filter_ignored_images(y_true, y_pred, classification=False):
    """ Filter those images which are not meaningful.

    Args:
        y_true: Target tensor from the dataset generator.
        y_pred: Predicted tensor from the network.
        classification: To filter for classification or
        regression.

    Returns: Filtered tensors.

    """
    states = y_true[:, :, -1]

    if classification:
        indexes = tf.where(tf.math.not_equal(states, -1))
    else:
        indexes = tf.where(tf.math.equal(states, 1))

    pred = y_pred
    true = y_true[:, :, :-1]

    true_filtered = tf.gather_nd(true, indexes)
    pred_filtered = tf.gather_nd(pred, indexes)

    return true_filtered, pred_filtered, indexes, states


def focal_loss(y_true, y_pred):
    """
    Calculate the binary classification loss as a weighted cross entropy loss
    term. See Focal loss for dense object detection
    https://arxiv.org/abs/1708.02002

    Args:
        y_true: Target tensor from the dataset generator with shape
        (batch_size, num_boxes, num_classes +1 ). Last value for each box is
        the state of the default box.
        y_pred: Predicted tensor from the network with shape
        (batch_size, num_boxes, num_classes+1).

    Returns: Focal loss of y_pred w.r.t. y_true.

    """
    gamma = 2.0

    true, pred, _, states = filter_ignored_images(y_true, y_pred,
                                                  classification=True)
    alpha = tf.keras.backend.ones_like(true) * 0.25
    alpha = tf.where(tf.math.equal(true, 1), alpha, 1 - alpha)

    cross_entropy_loss = tf.keras.backend.binary_crossentropy(true, pred)

    multiplier = tf.where(tf.math.equal(true, 1), 1 - pred, pred)

    loss = alpha * (multiplier ** gamma) * cross_entropy_loss

    normalizer = tf.where(tf.math.equal(states, 1))
    normalizer = tf.cast(tf.shape(normalizer)[0], dtype=tf.float32)
    normalizer = tf.math.maximum(1.0, normalizer)

    return tf.math.reduce_sum(loss) / normalizer


def smooth_l1(y_true, y_pred):
    """ Calculate the localization loss.

    Args:
        y_true: Target tensor from the dataset generator with shape
        (batch_size, num_boxes, 4 + 1). The last value for each box is the state
        of the default_box.
        y_pred: Predicted tensor from the network with shape
        (batch_size, num_boxes, num_classes).

    Returns: Smooth l1 loss of y_pred w.r.t. y_true.

    """
    beta = 9.0

    true, pred, indexes, _ = filter_ignored_images(y_true, y_pred)

    absolute_loss = tf.math.abs(pred - true) - (0.5 / beta)
    squared_loss = 0.5 * ((pred - true) ** 2) * beta

    loss = tf.where(tf.math.abs(pred - true) < (1.0 / beta), squared_loss,
                    absolute_loss)

    normalizer = tf.math.maximum(1, tf.shape(indexes)[0])
    normalizer = tf.cast(normalizer, dtype=tf.float32)

    return tf.math.reduce_sum(loss) / normalizer
