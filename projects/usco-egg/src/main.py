import cv2
import matplotlib.pyplot as plt
from utils import read_image, save_image
from preprocessing import apply_clahe_lab
from illumination import adjust_lighting
from segmentation import bilateral_segmentation, remove_background
from counting import count_eggs


def plot_lab_histogram(image , a_range = [0, 256], b_range = [0, 256], l_range = [0, 256], suffix = ""):
    """
    Plots the histogram of L, A, B channels to analyze color distribution.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Plot histograms
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(L.ravel(), bins=256, range=[0, 256], color='black')
    plt.title("L Channel Histogram")
    # Apply range
    plt.xlim(l_range)

    plt.subplot(1, 3, 2)
    plt.hist(A.ravel(), bins=256, range=[0, 256], color='red')
    plt.title("A Channel Histogram")
    plt.xlim(a_range)

    plt.subplot(1, 3, 3)
    plt.hist(B.ravel(), bins=256, range=[0, 256], color='blue')
    plt.title("B Channel Histogram")
    plt.xlim(b_range)

    # plt.show()

    # Save histograms
    plt.savefig(f"lab_histogram_{suffix}.png", dpi=300)

def main():


  # Load image
  image_path = "raw/0000.tif"
  image = read_image(image_path)

  # Apply preprocessing (CLAHE and lighting adjustment)
  processed = apply_clahe_lab(image)
  adjusted = adjust_lighting(processed)

  # plot_lab_histogram(cv2.cvtColor(adjusted, cv2.COLOR_RGB2BGR), a_range=[115,150],
  #                    b_range=[115,160], suffix="adjusted")


  # return
  save_image(adjusted, "output/adjusted_image.jpg")

  # Define LAB color bounds for segmentation
  bounds = [140, 170]
  lower_bound = [66, bounds[0], bounds[0]]
  upper_bound = [255, bounds[1], bounds[1]]

  # Apply segmentation
  binary_mask = bilateral_segmentation(adjusted, lower_bound, upper_bound)

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