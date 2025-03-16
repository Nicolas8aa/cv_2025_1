import cv2
import numpy as np


def hsv_segmentation(im_rgb, bounds_list):
    """
    Applies color-based segmentation using the HSV color space.
    
    - im_rgb: Input RGB image.
    - bounds_list: List of (lower_bound, upper_bound) HSV ranges for different egg colors.
    
    Returns:
    - Binary mask with segmented eggs.
    """
    # Convert to HSV color space
    im_hsv = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2HSV)
    
    # Initialize an empty mask
    final_mask = np.zeros(im_hsv.shape[:2], dtype=np.uint8)

    # Apply thresholding for each color range
    for lower_bound, upper_bound in bounds_list:
        mask = cv2.inRange(im_hsv, np.array(lower_bound, dtype=np.uint8), np.array(upper_bound, dtype=np.uint8))
        
        # Combine masks using bitwise OR
        final_mask = cv2.bitwise_or(final_mask, mask)

    return final_mask


def lab_segmentation(im_rgb, bounds_list):
    """
    Applies color-based segmentation using the LAB color space.
    
    - im_rgb: Input RGB image.
    - bounds_list: List of (lower_bound, upper_bound) LAB ranges for different egg colors.
    
    Returns:
    - Binary mask with segmented eggs.
    """
    # Convert to LAB color space
    im_lab = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2LAB)
    
    # Initialize an empty mask
    final_mask = np.zeros(im_lab.shape[:2], dtype=np.uint8)

    # Apply thresholding for each color range
    for lower_bound, upper_bound in bounds_list:
        mask = cv2.inRange(im_lab, np.array(lower_bound, dtype=np.uint8), np.array(upper_bound, dtype=np.uint8))
        
        # Combine masks using bitwise OR
        final_mask = cv2.bitwise_or(final_mask, mask)

    return final_mask