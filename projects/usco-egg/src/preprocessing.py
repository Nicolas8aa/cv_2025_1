# I need to import the necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


dirname =  os.path.dirname(__file__)
data_dir = os.path.join(dirname, '../data')



def apply_clahe_lab(image):
    """Applies CLAHE on the L-channel of the LAB color space to normalize brightness."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)

    # Merge back and convert to RGB
    lab_clahe = cv2.merge((L_clahe, A, B))
    img_corrected = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return img_corrected



def normalize_brightness(im_rgb):
    im_hsv = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2HSV)
    V = im_hsv[:, :, 2]  # Extract V channel

    # Apply CLAHE to V channel to normalize brightness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    V_normalized = clahe.apply(V)

    im_hsv[:, :, 2] = V_normalized  # Replace original V channel
    return cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)  # Convert back to RGB


def normalize_egg_color_hsv(image):
    """
    Normalize egg colors in HSV space by adjusting the V channel.
      :param image: Input RGB image.
      :return: Color-normalized image in RGB.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    # Apply CLAHE to correct brightness inconsistencies
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    V_clahe = clahe.apply(V)

    # Merge the adjusted channels
    hsv_corrected = cv2.merge((H, S, V_clahe))


    return image

    return cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2RGB)



def normalize_egg_color_lab(image):
    """
    Applies a normalization to correct lighting variations in the image.
      :param image: Input RGB image.
      :return: Color-normalized image in RGB.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    # Normalize L-channel
    L = cv2.normalize(L, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # A = cv2.normalize(A, None, 128, 128, cv2.NORM_MINMAX)
    # B = cv2.normalize(B, None, 128, 128, cv2.NORM_MINMAX)

    # Merge and convert back
    lab_corrected = cv2.merge((L, A, B))
    img_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)

    return img_corrected




def analyze_image(image):
    # Get properties
    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]  # Check grayscale or RGB
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])  # Histogram of first channel

    return {
        "shape": (height, width, channels),
        "histogram": hist
    }

def show_histograms(image_data):
    n_cols = 5
    n_rows = len(image_data.items()) // n_cols + 1

    # Plot histograms to analyze brightness distribution
    plt.figure(figsize=(12, 4))
    for i, (path, data) in enumerate(image_data.items()):
        plt.subplot(n_rows,n_cols,i + 1)
        plt.plot(data["histogram"], color="black")
        plt.title(f"Histogram: {Path(path).name}")
    plt.show()
    
