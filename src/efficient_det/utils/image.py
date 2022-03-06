import cv2
import numpy as np


def normalize(image):
    """ Normalize image along channel axis.

    Args:
        image: Image array.

    Returns: Normalized image.

    """
    # std and mean are dataset specific
    std = [0.25472827, 0.25604966, 0.26684684]
    mean = [0.48652189, 0.50312634, 0.44743868]

    new_image = image / 255.
    new_image = (new_image - mean) / std

    return new_image


