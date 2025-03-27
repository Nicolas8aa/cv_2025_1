import numpy as np
import cv2
import glob

def compute_normalized_histograms(image_paths, bins=256):
    """
    Compute normalized histograms for L, A, and B channels.
    Returns cumulative histograms for all images.
    """
    hist_L = np.zeros(bins, dtype=np.float32)
    hist_A = np.zeros(bins, dtype=np.float32)
    hist_B = np.zeros(bins, dtype=np.float32)

    for path in image_paths:
        image = cv2.imread(path)
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        L, A, B = cv2.split(image_lab)

        # Compute histograms and normalize
        hist_L += cv2.calcHist([L], [0], None, [bins], [0, 256]).flatten()
        hist_A += cv2.calcHist([A], [0], None, [bins], [0, 256]).flatten()
        hist_B += cv2.calcHist([B], [0], None, [bins], [0, 256]).flatten()

    # Normalize histograms (sum to 1)
    hist_L /= hist_L.sum()
    hist_A /= hist_A.sum()
    hist_B /= hist_B.sum()

    return hist_L, hist_A, hist_B

def get_percentile_bounds(hist, percentile=5):
    """
    Computes the lower and upper bounds for a given percentile in a normalized histogram.
    """
    cdf = np.cumsum(hist)  # Compute cumulative distribution function (CDF)
    
    lower_bound = np.searchsorted(cdf, percentile / 100.0)  # Lower bound (e.g., 5%)
    upper_bound = np.searchsorted(cdf, 1 - (percentile / 100.0))  # Upper bound (e.g., 95%)

    return lower_bound, upper_bound, cdf



# Load dataset images
image_paths = glob.glob("../data/raw/*.tif")  # Adjust dataset path

# Compute normalized histograms
hist_L, hist_A, hist_B = compute_normalized_histograms(image_paths)


# Compute the best LAB bounds (one way to do it)
L_min, L_max = get_percentile_bounds(hist_L, percentile=10)
A_min, A_max = get_percentile_bounds(hist_A, percentile=5)
B_min, B_max = get_percentile_bounds(hist_B, percentile=5)

optimal_bounds = (np.array([L_min, A_min, B_min], dtype=np.uint8),
                  np.array([L_max, A_max, B_max], dtype=np.uint8))

print("Optimal LAB bounds for egg segmentation:", optimal_bounds)