import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img_2 = cv2.imread('./images/test_pattern_blurring_orig.tif', cv2.IMREAD_GRAYSCALE)

# Callback function for the trackbar
def update_kernel_size(val):
    # Ensure the kernel size is odd and at least 1
    kernel_size = max(1, val | 1)  # Make sure it's odd
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    
    # Apply the filter
    filtered_img = cv2.filter2D(img_2, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    
    # Display the filtered image
    cv2.imshow('Filtered Image', filtered_img)

# Create a window
cv2.namedWindow('Filtered Image')

# Create a trackbar for kernel size
cv2.createTrackbar('Kernel Size', 'Filtered Image', 1, 31, update_kernel_size)  # Max kernel size is 31

# Initial display
update_kernel_size(1)

# Wait for user to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()