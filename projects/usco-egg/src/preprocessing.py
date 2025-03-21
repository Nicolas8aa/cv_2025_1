# I need to import the necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from utils import read_image, save_image


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
    

def get_adaptive_thresholds(image, base_lower, base_upper, scale=1.5):
    """
    Adjusts HSV bounds dynamically based on lighting conditions.
    
    :param image: Input RGB image.
    :param base_lower: Manually picked lower HSV bound.
    :param base_upper: Manually picked upper HSV bound.
    :param scale: Factor for standard deviation adjustment.
    :return: Adaptive lower and upper HSV bounds.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    # Compute mean and std deviation of the H & S channels
    mean_H, std_H = np.mean(H), np.std(H)
    mean_S, std_S = np.mean(S), np.std(S)

    # Adjust base bounds using standard deviation
    adaptive_lower = np.array([
        max(0, base_lower[0] - scale * std_H), 
        max(0, base_lower[1] - scale * std_S), 
        base_lower[2]
    ], dtype=np.uint8)

    adaptive_upper = np.array([
        min(180, base_upper[0] + scale * std_H), 
        min(255, base_upper[1] + scale * std_S), 
        base_upper[2]
    ], dtype=np.uint8)

    return adaptive_lower, adaptive_upper
