import cv2
import numpy as np
import sys
from segmentation import hsv_segmentation, lab_segmentation
from utils import read_image, resize_image

# Check for color space flag
if len(sys.argv) != 2 or sys.argv[1] not in ["HSV", "LAB"]:
    print("Usage: python tune_bounds.py [HSV|LAB]")
    sys.exit(1)

color_space = sys.argv[1]

# Load the image
image_original = read_image("raw/0008.tif", cv2.IMREAD_COLOR)
image = resize_image(image_original, 40)

# Function to update segmentation
bounds_list = []

def print_bounds():
    for lower_bound, upper_bound in bounds_list:
        print(f"({lower_bound}, {upper_bound}),")

def update_segmentation(_=None):
    if color_space == "HSV":
        lower_bound = np.array([cv2.getTrackbarPos("H Min", "Segmentation"), 
                                cv2.getTrackbarPos("S Min", "Segmentation"), 
                                cv2.getTrackbarPos("V Min", "Segmentation")], dtype=np.uint8)
        
        upper_bound = np.array([cv2.getTrackbarPos("H Max", "Segmentation"), 
                                cv2.getTrackbarPos("S Max", "Segmentation"), 
                                cv2.getTrackbarPos("V Max", "Segmentation")], dtype=np.uint8)
    else:
        lower_bound = np.array([cv2.getTrackbarPos("L Min", "Segmentation"), 
                                cv2.getTrackbarPos("A Min", "Segmentation"), 
                                cv2.getTrackbarPos("B Min", "Segmentation")], dtype=np.uint8)
        
        upper_bound = np.array([cv2.getTrackbarPos("L Max", "Segmentation"), 
                                cv2.getTrackbarPos("A Max", "Segmentation"), 
                                cv2.getTrackbarPos("B Max", "Segmentation")], dtype=np.uint8)

    # Save bounds
    if cv2.getTrackbarPos("Save", "Segmentation") == 1:
        bounds_list.append((lower_bound.copy(), upper_bound.copy()))
        # Clear the console
        print("\033[H\033[J")
        print_bounds()
        cv2.setTrackbarPos("Save", "Segmentation", 0)

    # Apply segmentation
    bounds_list_buffer = bounds_list.copy()
    bounds_list_buffer.append((lower_bound, upper_bound))

    if color_space == "HSV":
        mask = hsv_segmentation(image, bounds_list_buffer)
    else:
        mask = lab_segmentation(image, bounds_list_buffer)

    # Show results
    cv2.imshow("Mask", mask)

# Create a window
cv2.namedWindow("Segmentation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Segmentation", 400, 280)

# Create trackbars for bounds
if color_space == "HSV":
    cv2.createTrackbar("H Min", "Segmentation", 0, 180, update_segmentation)
    cv2.createTrackbar("H Max", "Segmentation", 180, 180, update_segmentation)
    cv2.createTrackbar("S Min", "Segmentation", 0, 255, update_segmentation)
    cv2.createTrackbar("S Max", "Segmentation", 255, 255, update_segmentation)
    cv2.createTrackbar("V Min", "Segmentation", 0, 255, update_segmentation)
    cv2.createTrackbar("V Max", "Segmentation", 255, 255, update_segmentation)
else:
    cv2.createTrackbar("L Min", "Segmentation", 0, 255, update_segmentation)
    cv2.createTrackbar("L Max", "Segmentation", 255, 255, update_segmentation)
    cv2.createTrackbar("A Min", "Segmentation", 0, 255, update_segmentation)
    cv2.createTrackbar("A Max", "Segmentation", 255, 255, update_segmentation)
    cv2.createTrackbar("B Min", "Segmentation", 0, 255, update_segmentation)
    cv2.createTrackbar("B Max", "Segmentation", 255, 255, update_segmentation)

cv2.createTrackbar("Save", "Segmentation", 0, 1, update_segmentation)

# Initial call to display segmentation
update_segmentation()

# Wait for user to exit
cv2.waitKey(0)
cv2.destroyAllWindows()