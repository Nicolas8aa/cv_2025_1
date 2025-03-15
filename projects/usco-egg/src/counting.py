import cv2

def count_eggs(binary_mask):
    """Counts the number of eggs based on detected contours."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size
    min_area, max_area = 500, 5000  # Adjust based on egg sizes
    egg_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    return len(egg_contours), egg_contours
