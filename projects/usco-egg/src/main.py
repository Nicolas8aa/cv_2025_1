import cv2
import os
import sys
from utils import read_image, save_image, data_dir
from main_hsv import main_hsv
from main_lab import main_lab
from results import display_results
from utils import plot_lab_histogram
from glob import glob

if len(sys.argv) != 2 or sys.argv[1] not in ["HSV", "LAB"]:
    print("Usage: python tune_bounds.py [HSV|LAB]")
    sys.exit(1)

color_space = sys.argv[1]


# Read image in RGB format
IMAGE = read_image('raw/0019.tif', cv2.IMREAD_COLOR_RGB)
REAL_COUNT = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12,
              14, 16, 18, 20, 20, 20, 20, 20, 20, 20]


def main():

    # Read all images from the dataset (raw folder)
    image_files = glob("../data/raw/*.tif")  # Adjust dataset path

    # Initialize list to store results
    results = []

    for image_file_path in image_files:

        image_file = os.path.basename(image_file_path)
        # Read image
        image = cv2.imread(image_file_path, cv2.IMREAD_COLOR_RGB)
        filename = image_file.split('.')[0]

        # Select color space
        if color_space == "HSV":
            egg_count, markers = main_hsv(image, show_results=False)
        elif color_space == "LAB":
            path_to_save_hist = os.path.join(
                data_dir, f"histograms/lab/{filename}.png")
            egg_count, markers, binary_mask = main_lab(
                image, show_results=False)
            # plot_lab_histogram(
            #     cv2.cvtColor(image, cv2.COLOR_RGB2BGR), a_range=[120, 160], b_range=[120, 180], path=path_to_save_hist,
            #     title=f"LAB Histogram for {image_file}")

        else:
            print("Invalid color space")
            sys.exit(1)

        # Save mask
        # save_image(binary_mask, f"masks/{color_space.lower()}/{image_file}")
        save_image(markers, f"masks/{color_space.lower()}/{image_file}")

        # Append results
        results.append(egg_count)

    # Print results
    display_results(REAL_COUNT, results, color_space)


main()
