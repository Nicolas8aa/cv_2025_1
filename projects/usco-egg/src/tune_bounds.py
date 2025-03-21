import cv2
import numpy as np
from segmentation import hsv_segmentation
from utils import read_image, resize_image

# Load the image
image_original = read_image("raw/0009.tif", cv2.IMREAD_COLOR_RGB)


image = resize_image(image_original, 40)


# Function to update segmentation
# Add a button to save the current segmentation bounds into a list

bounds_list = []


def update_segmentation(_=None):
    lower_bound = np.array([cv2.getTrackbarPos("H Min", "Segmentation"), 
                            cv2.getTrackbarPos("S Min", "Segmentation"), 
                            cv2.getTrackbarPos("V Min", "Segmentation")], dtype=np.uint8)
    
    upper_bound = np.array([cv2.getTrackbarPos("H Max", "Segmentation"), 
                            cv2.getTrackbarPos("S Max", "Segmentation"), 
                            cv2.getTrackbarPos("V Max", "Segmentation")], dtype=np.uint8)

    if cv2.getTrackbarPos("Save", "Segmentation"):
        bounds_list.append((lower_bound, upper_bound))
        cv2.setTrackbarPos("Save", "Segmentation", 0)
        print(f"Bounds saved: {bounds_list}")


    # Convert to HSV and apply segmentation
    mask = hsv_segmentation(image, [(lower_bound, upper_bound)])

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
cv2.waitKey(0)
cv2.destroyAllWindows()
