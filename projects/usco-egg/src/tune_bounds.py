import cv2
import numpy as np
from segmentation import hsv_segmentation
from utils import read_image, resize_image

# Load the image
image_original = read_image("raw/0019.tif", cv2.IMREAD_COLOR_RGB)


image = resize_image(image_original, 40)


# Function to update segmentation
# Add a button to save the current segmentation bounds into a list

# Accumulate bounds
bounds_list = []


# Iterate over the bounds list and print as [(lower, upper), ...]
def print_bounds():
  for lower_bound, upper_bound in bounds_list:
    print(f"({lower_bound}, {upper_bound}),")



def update_segmentation(_=None):
    lower_bound = np.array([cv2.getTrackbarPos("H Min", "Segmentation"), 
                            cv2.getTrackbarPos("S Min", "Segmentation"), 
                            cv2.getTrackbarPos("V Min", "Segmentation")], dtype=np.uint8)
    
    upper_bound = np.array([cv2.getTrackbarPos("H Max", "Segmentation"), 
                            cv2.getTrackbarPos("S Max", "Segmentation"), 
                            cv2.getTrackbarPos("V Max", "Segmentation")], dtype=np.uint8)

    # Save bounds
    if cv2.getTrackbarPos("Save", "Segmentation") == 1:
        bounds_list.append((lower_bound.copy(), upper_bound.copy()))
        # Clear the console
        print("\033[H\033[J")
        print_bounds()
        cv2.setTrackbarPos("Save", "Segmentation", 0)


    # Convert to HSV and apply segmentation

    # Apply segmentation as a buffer

    bounds_list_buffer = bounds_list.copy()
    bounds_list_buffer.append((lower_bound, upper_bound))

    mask = hsv_segmentation(image, bounds_list_buffer)

    # Show results
    cv2.imshow("Mask", mask)

# Create a window
cv2.namedWindow("Segmentation")


# Create trackbars for HSV bounds
cv2.createTrackbar("H Min", "Segmentation", 0, 180, update_segmentation)
cv2.createTrackbar("H Max", "Segmentation", 180, 180, update_segmentation)
cv2.createTrackbar("S Min", "Segmentation", 0, 255, update_segmentation)
cv2.createTrackbar("S Max", "Segmentation", 255, 255, update_segmentation)
cv2.createTrackbar("V Min", "Segmentation", 0, 255, update_segmentation)
cv2.createTrackbar("V Max", "Segmentation", 255, 255, update_segmentation)
cv2.createTrackbar("Save", "Segmentation", 0, 1, update_segmentation)



# Initial call to display segmentation
update_segmentation()

# Wait for user to exit
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Exit on ESC
        break

cv2.destroyAllWindows()
