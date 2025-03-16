import cv2
import numpy as np
from utils import read_image, save_image
# Load an image
image = read_image("raw/0000.tif")
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


# Resize to fit the screen (keeping aspect ratio)
scale_percent = 30  # Reduce size to 30% of the original
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# Adjust LAB image size accordingly
resized_lab = cv2.resize(lab_image, (width, height), interpolation=cv2.INTER_AREA)

# Click on pixels and print LAB values
# Get average LAB values for the entire image
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        L, A, B = lab_image[y, x]  # Get LAB values at the clicked point
        print(f"Clicked at ({x}, {y}) - L: {L}, A: {A}, B: {B}")

cv2.imshow("Click to get LAB values", resized_image)
cv2.setMouseCallback("Click to get LAB values", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
