import cv2
import os
from utils import read_image, save_image, data_dir
from main_hsv import main_hsv
from main_lab import main_lab
from results import display_results


IMAGE = read_image('raw/0000.tif', cv2.IMREAD_COLOR_RGB) # Read image in RGB format
REAL_COUNT = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 20, 20 , 20, 20, 20, 20]	

def main():

  # count = main_hsv(IMAGE, show_results=True)

  # return
  
  # Read all images from the dataset (raw folder)
  raw_dir = os.path.join(data_dir, 'raw')
  image_files = os.listdir(raw_dir)

  # Initialize list to store results
  results = []

  # Loop over all images
  for image_file in image_files:
    # Read image


    if not image_file.endswith('.tif'):
      continue

    image = read_image(f'raw/{image_file}', cv2.IMREAD_COLOR_RGB)

    # Count eggs
    # egg_count, markers = main_hsv(image, show_results=False)
    egg_count, markers = main_lab(image, show_results=False)

    # Save mask
    save_image(markers, f"masks/{image_file}")

    # print(f"Predicted Count: {egg_count}")

    # Append results
    results.append(egg_count)

  # Print results
  display_results(REAL_COUNT, results)




main()