import cv2

from efficient_det.configuration.constants import NAMES_TO_LABELS


def draw_boxes(draw_image, boxes, scores, labels, threshold=0.6):
    """ Draw bounding boxes to given image.

    Args:
        draw_image: Image to draw boxes.
        boxes: Predicted bounding boxes.
        scores: Predicted scores.
        labels: Predicted labels.
        threshold: Boxes whose score is above this threshold will be drawn.

    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(boxes.shape[1]):

        x_min = boxes[0][i][0]
        y_min = boxes[0][i][1]
        x_max = boxes[0][i][2]
        y_max = boxes[0][i][3]

        score = scores[0][i]

        label = labels[0][i]
        label_name = list(NAMES_TO_LABELS.keys())[label]

        start_point = (x_min, y_min)
        end_point = (x_max, y_max)

        if score > threshold:
            text = " ".join([label_name, str(round(score, 4))])

            draw_image = cv2.rectangle(draw_image, start_point, end_point,
                                       color=(0, 255, 255), thickness=3)
            draw_image = cv2.putText(draw_image, text, (x_min, y_max), font,
                                     fontScale=0.5, color=(0, 0, 0),
                                     thickness=1)

    return draw_image
