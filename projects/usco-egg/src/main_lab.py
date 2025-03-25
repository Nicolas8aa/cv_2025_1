
from segmentation import lab_segmentation
from counting import  count_eggs
from results import save_and_display_results


def main_lab(image, show_results=False):
  """
  Main function for egg counting using the LAB color space.
    :param image: Input RGB image.
    :return: Number of detected eggs.
  """

  # Adjust image 
  adjusted = image

  
  # Define LAB color bounds for segmentation
  bounds = [140, 170]
  L_bound = [66, 255]
  lower_bound = [L_bound[0], bounds[0], bounds[0]]
  upper_bound = [L_bound[1], bounds[1], bounds[1]]

  bounds = [
    (lower_bound, upper_bound), # Normal eggs (working)
    # ([96, 42, 129],[103, 42, 255]),
    # ([ 0 ,71 ,65], [255, 122, 120]),
    # ([  0 ,118, 104], [255, 125, 113]),
    # ([  0, 120 , 91], [255, 141, 105])

  ]

  # Apply segmentation
  binary_mask = lab_segmentation(adjusted, bounds)


  # Count eggs
  egg_count, markers = count_eggs(binary_mask, area_range=(10000, 55000))

  # Draw detected contours

  if show_results:
    save_and_display_results(adjusted, markers, egg_count)

  return egg_count, markers

