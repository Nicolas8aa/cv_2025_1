from utils import save_image, display_results
from segmentation import hsv_segmentation
from counting import  count_eggs





def handle_results(image, binary_mask, egg_count):
  # Save results
  save_image(image, "output/adjusted_image.jpg")
  save_image(binary_mask, "output/binary_mask.jpg")

  # Display results
  display_results(image, binary_mask, egg_count)


def main_hsv(image, show_results=False):

#   Start timer
  adjusted = image
  # adjusted = normalize_brightness(image)


  # Define HSV color bounds for segmentation
  bounds_list = [

    ([18, 49, 151], [24, 95, 255]),  # White eggs +
    ([9, 106, 113], [13, 172, 255]),  # Normal eggs +
    ([13, 59, 175], [17, 82, 255]),  # Normal Light eggs +
  ]
  
 

  # Apply segmentation
  binary_mask =  hsv_segmentation(adjusted, bounds_list)


  # Count eggs
  egg_count, markers = count_eggs(binary_mask, area_range=(10000, 55000))  # From 30000 to 52000


  # Save results
  if show_results:
    handle_results(adjusted, markers, egg_count)

  return egg_count, markers


