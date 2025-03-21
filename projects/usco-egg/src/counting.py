import cv2
import matplotlib.pyplot as plt
import numpy as np


def analyze_egg_sizes(binary_mask):
    """Analyze contour areas to determine appropriate egg size range."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate areas
    areas = [cv2.contourArea(cnt) for cnt in contours]

    # Plot histogram
    plt.hist(areas, bins=20, color='blue', alpha=0.7)
    plt.xlabel("Contour Area")
    plt.ylabel("Frequency")
    plt.title("Egg Size Distribution")
    plt.show()

    return areas




def jaccard_index(mask1, mask2):
    """Compute Jaccard Index (IoU) between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

def filter_eggs_by_similarity(binary_mask, reference_egg, rows=5, cols=6, margin=70, fill_threshold=0.6, jaccard_threshold=0.5):
    """
    Grid-based egg detection with Jaccard similarity filtering.
    
    - binary_mask: Binary image where eggs are white (255).
    - reference_egg: Binary mask of a typical egg shape.
    - rows, cols: Grid size for dividing the basket.
    - margin: Padding to avoid including the image background.
    - fill_threshold: % of filled pixels required to consider as a candidate.
    - jaccard_threshold: Minimum Jaccard index to classify as an egg.

    Returns:
    - egg_count: Number of detected eggs.
    - grid_visualization: Image with grid overlay.
    """
    h, w = binary_mask.shape

    # Adjust dimensions considering margin
    new_w = w - 2 * margin
    new_h = h - 2 * margin
    start_x = margin
    start_y = margin

    # Compute cell size
    grid_w = new_w // cols
    grid_h = new_h // rows

    egg_count = 0
    grid_visualization = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    for i in range(rows):
        for j in range(cols):
            x_start, y_start = start_x + j * grid_w, start_y + i * grid_h
            x_end, y_end = x_start + grid_w, y_start + grid_h

            cell = binary_mask[y_start:y_end, x_start:x_end]
            total_pixels = cell.size
            filled_pixels = np.count_nonzero(cell)
            fill_ratio = filled_pixels / total_pixels

            # Step 1: Check if the socket is filled enough
            if fill_ratio >= fill_threshold:
                # Step 2: Resize reference egg to match the cell size
                resized_egg = cv2.resize(reference_egg, (grid_w, grid_h))

                # Step 3: Compute Jaccard Index
                similarity = jaccard_index(cell > 0, resized_egg > 0)  # Convert to boolean masks
                
                # Step 4: Validate egg detection using Jaccard Index
                if similarity >= jaccard_threshold:
                    egg_count += 1
                    color = (0, 255, 0)  # Green box for detected eggs
                else:
                    color = (0, 255, 255)  # Yellow for rejected objects (not eggs)
            else:
                color = (0, 0, 255)  # Red for empty sockets

            # Draw grid visualization
            cv2.rectangle(grid_visualization, (x_start, y_start), (x_end, y_end), color, 2)

    return egg_count, grid_visualization



def count_eggs(binary_mask, threshold_area=1000):
    """
    Basic egg counting algorithm based on objects presents from binary mask.
    
    - binary_mask: Binary image where eggs are white (255).
    - threshold_area: Minimum area to consider as an egg.

    Returns:
    - egg_count: Number of detected eggs.
    - markers: Binary mask with detected eggs.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    egg_count = 0

    # Draw a rectangle around each egg

    markers = np.zeros_like(binary_mask)


    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > threshold_area:
            egg_count += 1
            cv2.drawContours(markers, [cnt], -1, 255, -1)

            # Draw bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(markers, (x, y), (x + w, y + h
            ), 255, 2)
            

    return egg_count, markers