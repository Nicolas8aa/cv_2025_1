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
    



# Define main function
def main():
    raw_dir = os.path.join(data_dir, 'raw')
    images = os.listdir(raw_dir)


    # Load images and analyze properties
    # image_data = {}
    # for path in images:
    #     if not path.endswith('.tif'):
    #         continue
    #     # Read image
    #     image = cv2.imread(os.path.join(raw_dir,path), cv2.IMREAD_UNCHANGED)  
        
    #     # Analyze image
    #     image_data[path] = analyze_image(image)

    # # Show histograms
    # show_histograms(image_data)
    # return

    for image_filename in images:
        
        # Filter only .tif files
        if not image_filename.endswith('.tif'):
            # print('skipping', image)
            continue
        
        # Read the image
        image = read_image(f'raw/{image_filename}')
        
        # I need to preprocess the image
        preprocessed_image = apply_clahe_lab(image)
        
        # I need to save the preprocessed image
        save_image(preprocessed_image, f'preprocessed/{image_filename}')



# main()