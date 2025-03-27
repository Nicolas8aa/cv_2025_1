
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
 
  lower_bound = [66, 140, 140]
  upper_bound = [255, 170, 170]

  bounds = [
    (lower_bound, upper_bound), # Normal eggs (working)
    # ([ 68, 126, 131], [185, 141, 158])
  ]

  # Apply segmentation
  binary_mask = lab_segmentation(adjusted, bounds)


  # Count eggs
  egg_count, markers = count_eggs(binary_mask, area_range=(10000, 55000))

  # Draw detected contours

  if show_results:
    save_and_display_results(adjusted, markers, egg_count)

  return egg_count, markers, binary_mask

