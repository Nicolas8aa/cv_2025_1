import cv2
import numpy as np

def adjust_lighting(image):
    """Applies a normalization to correct lighting variations in the image."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    # Normalize L-channel
    L = cv2.normalize(L, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Merge and convert back
    lab_corrected = cv2.merge((L, A, B))
    img_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)

    return img_corrected
