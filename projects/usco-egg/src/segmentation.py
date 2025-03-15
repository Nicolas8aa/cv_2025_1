import cv2
import numpy as np

import cv2
import numpy as np

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
    
    # Split LAB channels
    L, A, B = cv2.split(im_lab)
    
    # Apply thresholding on the A or B channel (based on color contrast of eggs)
    mask = cv2.inRange(im_lab, np.array(lower_bound, dtype=np.uint8), np.array(upper_bound, dtype=np.uint8))

    # Create a binary mask (white for detected regions, black elsewhere)
    binary_mask = np.zeros_like(mask)
    binary_mask[mask > 0] = 255

    return binary_mask
