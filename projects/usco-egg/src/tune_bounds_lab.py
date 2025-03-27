import cv2
import numpy as np
from utils import read_image, resize_image
from segmentation import lab_segmentation

# Load and resize the image
image_original = read_image("raw/0019.tif", cv2.IMREAD_COLOR)
image = resize_image(image_original, 40)

# Convert to LAB (Correct format: BGR → LAB)
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Global list to store saved bounds
bounds_list = []

def print_bounds():
    print("\nCurrent Bounds List:")
    for lower, upper in bounds_list:
        print(f"({lower.tolist()}, {upper.tolist()})")

def update_segmentation(_=None):
    """
    Updates segmentation in real-time based on trackbar values.
    """
    # Read trackbar values (Corrected for LAB ranges)
    lower_bound = np.array([
        cv2.getTrackbarPos("L Min", "Segmentation"),
        cv2.getTrackbarPos("A Min", "Segmentation"),
        cv2.getTrackbarPos("B Min", "Segmentation")
    ], dtype=np.uint8)

    upper_bound = np.array([
        cv2.getTrackbarPos("L Max", "Segmentation"),
        cv2.getTrackbarPos("A Max", "Segmentation"),
        cv2.getTrackbarPos("B Max", "Segmentation")
    ], dtype=np.uint8)

    # Save bounds when "Save" is pressed
    if cv2.getTrackbarPos("Save", "Segmentation") == 1:
        bounds_list.append((lower_bound.copy(), upper_bound.copy()))
        print_bounds()
        cv2.setTrackbarPos("Save", "Segmentation", 0)  # Reset save button

    # Apply segmentation with saved and live bounds
    bounds_list_buffer = bounds_list.copy()
    bounds_list_buffer.append((lower_bound, upper_bound))
    mask = lab_segmentation(image_lab, bounds_list_buffer)

    # Debugging output
    # print(f"Lower: {lower_bound}, Upper: {upper_bound}, Mask Unique Values: {np.unique(mask)}")

    # Show segmentation mask
    cv2.imshow("Mask", mask)

# Create UI window
cv2.namedWindow("Segmentation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Segmentation", 400, 300)

# lower_initial = [66, 140, 140]
# upper_initial = [255, 170, 170]

# Set **initial reasonable values** for segmentation to be visible
cv2.createTrackbar("L Min", "Segmentation", 0, 255, update_segmentation)
cv2.createTrackbar("L Max", "Segmentation", 255, 255, update_segmentation)
cv2.createTrackbar("A Min", "Segmentation", 0, 255, update_segmentation)  # Adjusted
cv2.createTrackbar("A Max", "Segmentation", 255, 255, update_segmentation)
cv2.createTrackbar("B Min", "Segmentation", 0, 255, update_segmentation)  # Adjusted
cv2.createTrackbar("B Max", "Segmentation", 255, 255, update_segmentation)

# Trackbar for saving bounds
cv2.createTrackbar("Save", "Segmentation", 0, 1, update_segmentation)

# Initial segmentation display
update_segmentation()

# Main loop
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Exit on ESC
        break

cv2.destroyAllWindows()
