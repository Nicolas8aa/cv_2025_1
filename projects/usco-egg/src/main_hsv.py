from segmentation import hsv_segmentation
from counting import  count_eggs
from results import save_and_display_results



def main_hsv(image, show_results=False):
  """
  Main function for egg counting using the HSV color space."
    :param image: Input RGB image.
    :return: Number of detected eggs and markers.
  """

  # Adjust image 
  adjusted = image


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
    save_and_display_results(adjusted, markers, egg_count)

  return egg_count, markers


