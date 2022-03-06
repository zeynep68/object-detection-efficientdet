"""
Copyright 2017-2018 Fizyr (https://github.com/fizyr/keras-retinanet)
Copyright 2021 Zeynep Boztoprak

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from efficient_det.layers.filter_predictions import get_detections
from efficient_det.utils.overlap import compute_intersection_over_union
from efficient_det.utils.anchors import generate_anchors_for_batch

from tensorflow import keras

import time
import numpy as np
import progressbar

assert (callable(
    progressbar.progressbar)), "Using wrong progressbar module, install " \
                               "'progressbar2' instead."


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100):
    """ Get the detections from the model using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections,
        4 + num_classes]
    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
    # Returns
        A list of lists containing the detections for each image in the
        generator.
    """
    all_detections = [[None for i in range(generator.num_classes)] for j in
                      range(generator.size())]
    all_inferences = [None for i in range(generator.size())]

    for i in range(generator.size()):
        image = generator.load_image(i)
        h, w = image.shape[:2]
        image, offset_h, offset_w = generator.preprocess_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        start = time.time()
        anchors_batch = generate_anchors_for_batch(1, generator.anchors)
        regression, classification = model([np.expand_dims(image, axis=0)])
        boxes, scores, labels = get_detections(regression, classification,
                                               anchors_batch,
                                               generator.image_shape)
        boxes = boxes.numpy()
        scores = scores.numpy()
        labels = labels.numpy()

        inference_time = time.time() - start

        # correct boxes for image scale
        boxes[:, :, 0] = boxes[:, :, 0] / offset_w
        boxes[:, :, 2] = boxes[:, :, 2] / offset_w
        boxes[:, :, 1] = boxes[:, :, 1] / offset_h
        boxes[:, :, 3] = boxes[:, :, 3] / offset_h

        #boxes /= scale
        boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w - 1)
        boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h - 1)
        boxes[:, :, 2] = np.clip(boxes[:, :, 2], 0, w - 1)
        boxes[:, :, 3] = np.clip(boxes[:, :, 3], 0, h - 1)

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes = boxes[0, indices[scores_sort], :]
        image_scores = scores[scores_sort]
        image_labels = labels[0, indices[scores_sort]]
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1),
             np.expand_dims(image_labels, axis=1)], axis=1)

        # copy detections to all_detections
        for label in range(generator.num_classes):
            all_detections[i][label] = image_detections[
                                       image_detections[:, -1] == label, :-1]

        all_inferences[i] = inference_time

    return all_detections, all_inferences


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the
        generator.
    """
    all_annotations = [[None for i in range(generator.num_classes)] for j in
                       range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()),
                                     prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes):
            all_annotations[i][label] = annotations['boxes'][
                                        annotations['labels'] == label,
                                        :].copy()

    return all_annotations


def evaluate(generator, model, iou_threshold=0.5, score_threshold=0.05,
             max_detections=100):
    """ Evaluate a given dataset using a given model.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is
        positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections, all_inferences = _get_detections(generator, model,
                                                     score_threshold,
                                                     max_detections)
    all_annotations = _get_annotations(generator)
    average_precisions = {}
    num_fp = 0
    num_tp = 0

    # process detections and annotations
    for label in range(generator.num_classes):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])
                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_intersection_over_union(
                    np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not \
                        in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(
            true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

        if not false_positives.shape[0] == 0:
            num_fp += false_positives[-1]
        if not true_positives.shape[0] == 0:
            num_tp += true_positives[-1]

    # inference time
    inference_time = np.sum(all_inferences) / generator.size()
    print(f'Number of false positives: {num_fp}.')
    print(f'Number of true positives: {num_tp}.')

    return average_precisions, inference_time, num_tp, num_fp
