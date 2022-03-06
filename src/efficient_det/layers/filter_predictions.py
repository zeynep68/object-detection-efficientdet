import tensorflow as tf


def regress_boxes(regress, boxes):
    """ Regress default boxes to objects in the image.

    Args:
        regress: Values to adjust default boxes.
        boxes: Boxes to be adjusted.

    Returns: Shifted Boxes.

    """
    std = [0.2, 0.2, 0.2, 0.2]

    width = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x_min = boxes[:, :, 0] + (regress[:, :, 0] * std[0]) * width
    y_min = boxes[:, :, 1] + (regress[:, :, 1] * std[1]) * height
    x_max = boxes[:, :, 2] + (regress[:, :, 2] * std[2]) * width
    y_max = boxes[:, :, 3] + (regress[:, :, 3] * std[3]) * height

    return tf.stack((x_min, y_min, x_max, y_max), axis=2)


def clip_boxes(boxes, img_shape=(512, 512, 3)):
    """ Boxes to check if they lie inside the image shape.

    Args:
        boxes: Boxes to clip values off.
        img_shape: Image resolution.

    Returns: Boxes inside the image.

    """
    width, height = img_shape[0], img_shape[1]

    x_min, y_min, x_max, y_max = tf.unstack(boxes, axis=2)

    return tf.stack([tf.clip_by_value(x_min, 0, width - 1),
                     tf.clip_by_value(y_min, 0, height - 1),
                     tf.clip_by_value(x_max, 0, width - 1),
                     tf.clip_by_value(y_max, 0, height - 1)], axis=2)


def decode_outputs(regression, classification, confidence_threshold=0.01,
                   nms_threshold=0.5, max_detections=300):
    """ Filter the relevant boxes with their corresponding scores and labels.

    Args:
        regression: Regression head output.
        classification: Classification head output.
        confidence_threshold: Score threshold.
        nms_threshold: NMS threshold.
        max_detections: Number of maximum detections to keep.

    Returns: Filtered detections.

    """

    boxes_all, scores_all, labels_all = [], [], []

    for i in range(regression.shape[0]):  # iterate over batch

        scores, boxes = filter(regression[i], classification[i],
                               confidence_threshold, nms_threshold,
                               max_detections)

        labels = get_label(scores)
        scores = get_highest_scores(scores)

        scores, boxes, labels = get_top_k(scores, boxes, labels, max_detections)

        fill = get_fill_number(scores, max_detections)

        boxes, scores, labels = fill_array(boxes, scores, labels, fill)

        boxes_all.append(boxes)
        scores_all.append(scores)
        labels_all.append(labels)

    return concatenate_predictions(boxes_all, scores_all, labels_all)


def filter(regression_, classification_, confidence_threshold, nms_threshold,
           max_detections):
    """ Filter network outputs by their score and nms_threshold.

    Args:
        regression_: Regression network outputs.
        classification_: Classification network outputs.
        confidence_threshold: Score threshold.
        nms_threshold: NMS threshold.
        max_detections: Number of max detections to keep.

    Returns: Filtered scores and boxes.

    """
    boxes, scores, confidence_indexes = filter_by_scores(regression_,
                                                         classification_,
                                                         confidence_threshold)

    nms_indexes = filter_by_nms_threshold(boxes, scores, max_detections,
                                          nms_threshold)

    indexes = tf.gather_nd(confidence_indexes, nms_indexes)

    scores = tf.gather_nd(classification_, indexes)
    boxes = tf.gather_nd(regression_, indexes)

    return scores, boxes


def filter_by_nms_threshold(boxes, scores, max_detections, nms):
    """ Filter by nms threshold.
    """
    nms = tf.image.non_max_suppression(boxes, scores, max_detections, nms)

    return tf.expand_dims(nms, axis=1)


def filter_by_scores(regression, classification, confidence_threshold):
    """ Filter network outputs by their scores.

    Args:
        regression: Network offset values.
        classification: Network scores.
        confidence_threshold: Score threshold.

    Returns: Filtered boxes, scores, filter_indexes.

    """
    scores = tf.math.reduce_max(classification, axis=1)

    confidence_indexes = tf.where(tf.math.greater(scores, confidence_threshold))

    boxes = tf.gather_nd(regression, confidence_indexes)
    scores = tf.gather_nd(scores, confidence_indexes)

    return boxes, scores, confidence_indexes


def get_label(scores):
    """ Compute labels using score values.

    Args:
        scores: Score values for each detection.

    Returns: Label for each detection.

    """
    return tf.math.argmax(scores, axis=1)


def get_highest_scores(scores):
    """ Compute highest score value.

    Args:
        scores: Score values for each detection.

    Returns: Highest score value for each detection.

    """
    return tf.math.reduce_max(scores, axis=1)


def concatenate_predictions(boxes, scores, labels):
    """ Change way of storing predictions from list elements into tensor.

    Args:
        boxes: Bounding box detections.
        scores: Highest score for each detection.
        labels: Label for each detection.

    Returns: Tensor objects.

    """
    boxes = tf.stack(boxes, axis=0)
    scores = tf.stack(scores, axis=0)
    labels = tf.stack(labels, axis=0)

    return boxes, scores, labels


def fill_array(boxes, scores, labels, fill):
    """ Make tensors to same size to avoid ragged tensors..
    """
    fill_boxes = tf.cast(tf.fill((fill, 4), 0), dtype=tf.float32)
    fill_scores = tf.cast(tf.fill((fill, 1), 0), dtype=tf.float32)
    fill_labels = tf.cast(tf.fill((fill, 1), 0), dtype=tf.int64)

    boxes = tf.concat([boxes, fill_boxes], axis=0)
    scores = tf.concat([scores, fill_scores[:, 0]], axis=0)
    labels = tf.concat([labels, fill_labels[:, 0]], axis=0)
    return boxes, scores, labels


def get_top_k(scores, boxes, labels, max_detections):
    """ Get top-k predictions w.r.t. number max_detections.
    """
    k = tf.math.minimum(max_detections, tf.keras.backend.shape(scores)[0])
    scores, top_indexes = tf.nn.top_k(scores, k=k)

    boxes = tf.keras.backend.gather(boxes, top_indexes)
    labels = tf.keras.backend.gather(labels, top_indexes)

    return scores, boxes, labels


def get_fill_number(scores, max_detections):
    """ Compute difference value between the maximum number of detections
    you'd like to keep and the actual number of detections.

    Args:
        scores: Score values for each detection.
        max_detections: Number of max detections to keep.

    """
    number_of_detections = tf.keras.backend.shape(scores)[0]
    diff = max_detections - number_of_detections
    return tf.math.maximum(diff, 0)


def get_detections(regression, classification, anchors, img_shape):
    """ Append layers to interpret the network outputs.

    Args:
        regression: Regression head output.
        classification: Classification head output.
        anchors: Default boxes.
        img_shape: Image resolution.

    Returns: Detected boxes with corresponding scores and labels.

    """
    boxes = regress_boxes(regression, anchors)
    boxes = clip_boxes(boxes, img_shape)
    detections = decode_outputs(boxes, classification)

    return detections
