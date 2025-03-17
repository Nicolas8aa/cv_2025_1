import cv2
import matplotlib.pyplot as plt
from utils import read_image, save_image, resize_image
from preprocessing import apply_clahe_lab
from illumination import adjust_lighting
from segmentation import hsv_segmentation, lab_segmentation
from counting import filter_eggs_by_similarity, count_eggs


REFERENCE_EGG = read_image('masks/reference-egg.png', cv2.IMREAD_GRAYSCALE)
IMAGE = read_image('raw/0002.tif')


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


def main_hsv():

  # Apply preprocessing (CLAHE and lighting adjustment)
  processed = apply_clahe_lab(IMAGE)
  adjusted = adjust_lighting(processed)

  # plot_lab_histogram(cv2.cvtColor(adjusted, cv2.COLOR_RGB2BGR), a_range=[115,150],
  #                    b_range=[115,160], suffix="adjusted")


  # Define HSV color bounds for segmentation

  bounds_list = [
    ([8, 115, 103], [18, 170, 221]), # Normal eggs
    ([8, 86, 129], [14, 161, 218]),  # Normal eggs +
    ([19, 58, 130], [25, 108, 195]), # Light Green Eggs +
    
    # ([18, 57, 122], [26, 110, 195]), # Light Green Eggs
    # ([9, 40, 59], [19, 138, 166]), # Basket
]

  # Apply segmentation
  binary_mask =  hsv_segmentation(adjusted, bounds_list)


  # Count eggs
  # egg_count, grid_image = filter_eggs_by_similarity(binary_mask, REFERENCE_EGG)
  egg_count = count_eggs(binary_mask, threshold_area=1000) 

 
  # Draw detected contours
  output_image = adjusted.copy()


  # Save results
  save_image(adjusted, "output/adjusted_image.jpg")
  save_image(output_image, "output/counted_eggs.jpg")
  save_image(binary_mask, "output/binary_mask.jpg")

  # Display results
  display_results(output_image, binary_mask, egg_count)



def main_lab():

  # Apply preprocessing (CLAHE and lighting adjustment)
  processed = apply_clahe_lab(IMAGE)
  adjusted = adjust_lighting(processed)

  # plot_lab_histogram(cv2.cvtColor(adjusted, cv2.COLOR_RGB2BGR), a_range=[115,150],
  #                    b_range=[115,160], suffix="adjusted")


  
  # Define LAB color bounds for segmentation
  bounds = [140, 170]
  L_bound = [66, 255]
 
  lower_bound = [L_bound[0], bounds[0], bounds[0]]
  upper_bound = [L_bound[1], bounds[1], bounds[1]]

  # Apply segmentation
  binary_mask = lab_segmentation(adjusted, [(lower_bound, upper_bound)])


  # Count eggs
  egg_count = count_eggs(binary_mask, threshold_area=1000)

  # Draw detected contours
  output_image = adjusted.copy()

  # Save results
  save_image(adjusted, "output/adjusted_image.jpg")
  save_image(output_image, "output/counted_eggs.jpg")
  save_image(binary_mask, "output/binary_mask.jpg")

  # Display results
  display_results(output_image, binary_mask, egg_count)



main_hsv()
# main_lab()