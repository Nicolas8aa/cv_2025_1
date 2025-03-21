import cv2
import numpy as np
from utils import read_image, save_image, resize_image

# Load an image
# image = read_image("raw/0002.tif")
image = read_image("output/adjusted_image.jpg")

# Resize to fit the screen (keeping aspect ratio)
scale_percent = 40  # Reduce size to 50% of the original
resized_image = resize_image(image, scale_percent)
resized_lab = resize_image(cv2.cvtColor(image, cv2.COLOR_BGR2LAB), scale_percent)
resized_hsv = resize_image(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), scale_percent)

# Variables to store the starting and ending points of the circle
start_point = None
end_point = None
drawing = False


def print_stats(selected_region):
    L, A, B = cv2.split(selected_region)
    L_min, L_max = L[L > 0].min(), L[L > 0].max()
    A_min, A_max = A[A > 0].min(), A[A > 0].max()
    B_min, B_max = B[B > 0].min(), B[B > 0].max()

    average_L = np.mean(L[L > 0]).round()
    average_A = np.mean(A[A > 0]).round()
    average_B = np.mean(B[B > 0]).round()
    
    print(f"min: [{L_min}, {A_min}, {B_min}]")
    print(f"max: [{L_max}, {A_max}, {B_max}]")
    print(f"avg: [{average_L}, {average_A}, {average_B}]\n")
    

def click_event(event, x, y, flags, param):
    global start_point, end_point, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            temp_image = resized_image.copy()
            radius = int(np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2))
            cv2.circle(temp_image, start_point, radius, (0, 255, 0), 2)
            cv2.imshow("Click and drag to select region", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        drawing = False
        radius = int(np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2))
        cv2.circle(resized_image, start_point, radius, (0, 255, 0), 2)
        cv2.imshow("Click and drag to select region", resized_image)
        
        # print("\nLAB RESULTS")
        # print("--------------------------------------")
        # get_lab_range(resized_lab, start_point, radius)

        print("\nHSV RESULTS")
        print("--------------------------------------")
        get_hsv_range(resized_hsv, start_point, radius)

def get_lab_range(image, center, radius):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    selected_region = cv2.bitwise_and(image, image, mask=mask)
    print_stats(selected_region)


def get_hsv_range(image, center, radius):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    selected_region = cv2.bitwise_and(image, image, mask=mask)
    print_stats(selected_region)

cv2.imshow("Click and drag to select region", resized_image)
cv2.setMouseCallback("Click and drag to select region", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()