import cv2
import numpy as np
from utils import read_image, save_image

# Load an image
image = read_image("raw/0002.tif")
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Resize to fit the screen (keeping aspect ratio)
scale_percent = 50  # Reduce size to 50% of the original
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
resized_lab = cv2.resize(lab_image, (width, height), interpolation=cv2.INTER_AREA)
resized_hsv = cv2.resize(hsv_image, (width, height), interpolation=cv2.INTER_AREA)

# Variables to store the starting and ending points of the circle
start_point = None
end_point = None
drawing = False

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
        get_lab_range(resized_lab, start_point, radius)
        # get_hsv_range(resized_hsv, start_point, radius)

def get_lab_range(image, center, radius):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    selected_region = cv2.bitwise_and(image, image, mask=mask)
    L, A, B = cv2.split(selected_region)
    L_min, L_max = L[L > 0].min(), L[L > 0].max()
    A_min, A_max = A[A > 0].min(), A[A > 0].max()
    B_min, B_max = B[B > 0].min(), B[B > 0].max()

    average_L = np.mean(L[L > 0])
    average_A = np.mean(A[A > 0])
    average_B = np.mean(B[B > 0])

    print(f"min LAB: [{L_min}, {A_min}, {B_min}]")
    print(f"max LAB: [{L_max}, {A_max}, {B_max}]")
    print(f"average LAB: [{average_L}, {average_A}, {average_B}]\n")

def get_hsv_range(image, center, radius):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    selected_region = cv2.bitwise_and(image, image, mask=mask)
    H, S, V = cv2.split(selected_region)
    H_min, H_max = H[H > 0].min(), H[H > 0].max()
    S_min, S_max = S[S > 0].min(), S[S > 0].max()
    V_min, V_max = V[V > 0].min(), V[V > 0].max()

    average_H = np.mean(H[H > 0])
    average_S = np.mean(S[S > 0])
    average_V = np.mean(V[V > 0])

    print(f"min HSV: [{H_min}, {S_min}, {V_min}]")
    print(f"max HSV: [{H_max}, {S_max}, {V_max}]\n")
    print(f"average HSV: [{average_H}, {average_S}, {average_V}]\n")

cv2.imshow("Click and drag to select region", resized_image)
cv2.setMouseCallback("Click and drag to select region", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()