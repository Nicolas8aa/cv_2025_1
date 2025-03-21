import cv2
import matplotlib.pyplot as plt
from utils import read_image, save_image, data_dir
from preprocessing import  normalize_egg_color_hsv, get_adaptive_thresholds
from segmentation import hsv_segmentation
from counting import  count_eggs
import os
import time
import numpy as np




def display_results(image, binary_mask, egg_count):
  # Display results
  fig, ax = plt.subplots(1, 2, figsize=(10, 5))
  ax[0].imshow(binary_mask, cmap="gray")
  ax[0].set_title("Binary Mask (Thresholded)")
  ax[0].axis("off")

  ax[1].imshow(image)
  ax[1].set_title(f"Detected Eggs: {egg_count}")
  ax[1].axis("off")

  plt.show()

  print(f"Total Eggs Counted: {egg_count}")



def handle_results(image, binary_mask, egg_count):
  # Save results
  save_image(image, "output/adjusted_image.jpg")
  save_image(binary_mask, "output/binary_mask.jpg")

  # Display results
  display_results(image, binary_mask, egg_count)



def main_hsv(image, show_results=False):

#   Start timer
  adjusted = normalize_egg_color_hsv(image)


  # Define HSV color bounds for segmentation

  adaptive_lower, adaptive_upper = get_adaptive_thresholds(adjusted, [9, 106, 113], [13, 172, 255])

  bounds_list = [

    ([18, 49, 151], [24, 95, 255]),  # White eggs +
    ([9, 106, 113], [13, 172, 255]),  # Normal eggs +
    ([13, 59, 175], [17, 82, 255]),  # Normal Light eggs +

    # (adaptive_lower, adaptive_upper), # Normal eggs
]
  


  # Apply segmentation
  binary_mask =  hsv_segmentation(adjusted, bounds_list)


  # Count eggs
  egg_count, markers = count_eggs(binary_mask, threshold_area=4000) 


  # Save results
  if show_results:
    handle_results(adjusted, markers, egg_count)

  return egg_count



IMAGE = read_image('raw/0009.tif', cv2.IMREAD_COLOR_RGB) # Read image in RGB format
REAL_COUNT = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 20, 20 , 20, 20, 20]	

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
    egg_count = main_hsv(image, show_results=False)

    # print(f"Predicted Count: {egg_count}")

    # Append results
    results.append(egg_count)

  # Print results and generate confusion matrix
  print("Predicted Counts:", results)
  print("Real Counts:", REAL_COUNT)

  # Calculate accuracy
  correct = 0
  for pred, real in zip(results, REAL_COUNT):
    if pred == real:
      correct += 1
  
  accuracy = correct / len(REAL_COUNT) * 100
  print(f"Accuracy: {accuracy:.2f}%")




main()