
import cv2
from segmentation import lab_segmentation
from counting import  count_eggs
from preprocessing import normalize_egg_color_lab

def main_lab(image):
  """
  Main function for egg counting using the LAB color space.
    :param image: Input RGB image.
    :return: Number of detected eggs.
  """


  # Apply preprocessing (CLAHE and lighting adjustment)
  adjusted = normalize_egg_color_lab(image)

  
  # Define LAB color bounds for segmentation
  bounds = [140, 170]
  L_bound = [66, 255]
  lower_bound = [L_bound[0], bounds[0], bounds[0]]
  upper_bound = [L_bound[1], bounds[1], bounds[1]]

  bounds = [
    (lower_bound, upper_bound), # Normal eggs (working)
    # ([96, 42, 129],[103, 42, 255])

  ]

  # Apply segmentation
  binary_mask = lab_segmentation(adjusted, bounds)


  # Count eggs
  egg_count = count_eggs(binary_mask, threshold_area=5000)

  # Draw detected contours

  # Save results
  # handle_results(adjusted, binary_mask, egg_count)

  return egg_count

