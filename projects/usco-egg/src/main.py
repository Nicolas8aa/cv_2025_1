import cv2
import matplotlib.pyplot as plt
from utils import read_image, save_image
from preprocessing import apply_clahe_lab
from illumination import adjust_lighting
from segmentation import segment_eggs
from counting import count_eggs


def main():


  # Load image
  image_path = "raw/0000.tif"
  image = read_image(image_path)

  # Apply preprocessing
  processed = apply_clahe_lab(image)
  adjusted = adjust_lighting(processed)

  save_image(adjusted, "output/adjusted_image.jpg")

  # Apply segmentation
  binary_mask = segment_eggs(adjusted)

  save_image(binary_mask, "output/binary_mask.jpg")


  return

  # Count eggs
  egg_count, contours = count_eggs(binary_mask)

  # Draw detected contours
  output_image = adjusted.copy()
  cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

  # Save results
  save_image(output_image, "output/counted_eggs.jpg")

  # Display results
  fig, ax = plt.subplots(1, 2, figsize=(10, 5))
  ax[0].imshow(binary_mask, cmap="gray")
  ax[0].set_title("Binary Mask (Thresholded)")
  ax[0].axis("off")

  ax[1].imshow(output_image)
  ax[1].set_title(f"Detected Eggs: {egg_count}")
  ax[1].axis("off")

  plt.show()

  print(f"Total Eggs Counted: {egg_count}")


main()