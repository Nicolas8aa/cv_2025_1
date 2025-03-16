import cv2
import numpy as np

import cv2
import numpy as np


import cv2
import numpy as np

def remove_background(image):
    """
    Removes background using simple thresholding.
    :param image: Input RGB image.
    :return: Image with background removed.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply thresholding
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Invert the mask
    mask = cv2.bitwise_not(mask)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)

    return result


def bilateral_segmentation(im_rgb, lower_bound, upper_bound):
    """
    Applies color-based segmentation using the LAB color space.
    :param im_rgb: Input RGB image.
    :param lower_bound: Lower bound of LAB values for thresholding.
    :param upper_bound: Upper bound of LAB values for thresholding.
    :return: Binary mask with segmented objects.
    """
    # Convert to LAB color space
    im_lab = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2LAB)
    
    # Apply thresholding on the A or B channel (based on color contrast of eggs)
    mask = cv2.inRange(im_lab, np.array(lower_bound, dtype=np.uint8), np.array(upper_bound, dtype=np.uint8))

    return mask
