import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import save_image
import os
from utils import data_dir

def compute_confusion_matrix(predicted_count, real_count):
    """
    Computes a confusion matrix for a single image.
    
    - predicted_count: Count of eggs detected by the system.
    - real_count: Actual number of eggs in the image.

    Returns:
    - Confusion matrix as a NumPy array.
    """
    TP = min(predicted_count, real_count)  # Correctly detected eggs
    FP = max(0, predicted_count - real_count)  # Overcounted (extra) eggs
    FN = max(0, real_count - predicted_count)  # Missed eggs
    TN = 0  # Not applicable in this context

    return np.array([[TP, FP], [FN, TN]])

def analyze_dataset(predictions, ground_truths):
    """
    Computes the total confusion matrix for the entire dataset.
    
    - predictions: List of predicted egg counts for each image.
    - ground_truths: List of actual egg counts for each image.

    Returns:
    - Global confusion matrix
    - Precision, Recall, F1-score, and Accuracy
    """
    assert len(predictions) == len(ground_truths), "Mismatch in dataset size"

    # Initialize a global confusion matrix
    total_cm = np.zeros((2, 2), dtype=int)

    # Process each image's confusion matrix and accumulate
    for pred, real in zip(predictions, ground_truths):
        total_cm += compute_confusion_matrix(pred, real)

    # Extract values
    TP, FP = total_cm[0]
    FN, TN = total_cm[1]

    # Compute performance metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return total_cm, precision, recall, f1_score, accuracy



def save_confusion_matrix(cm, path, show=True):
    """
    Saves the confusion matrix as an image.
    """
    labels = ["Correctly Counted (TP)", "Overcounted (FP)"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=["Missed (FN)", "N/A"])
    plt.xlabel("Predicted Count")
    plt.ylabel("Actual Count")
    plt.title("Global Confusion Matrix for Dataset")
    plt.savefig(path, dpi=300)
    if show:
      plt.show()
    plt.close()


def display_results(predictions, ground_truths, color_space):
    """
    Displays the results of the egg counting system.
    
    - predictions: List of predicted egg counts for each image.
    - ground_truths: List of actual egg counts for each image.
    """
    # Analyze the dataset
    cm, precision, recall, f1_score, accuracy = analyze_dataset(predictions, ground_truths)

    # Display results
    print(f"Global Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    # Plot and save confusion matrix

    path_to_save = os.path.join(data_dir, f"output/confusion_matrix_{color_space.lower()}.png")

    save_confusion_matrix(cm, path_to_save)




def save_and_display_results(image, binary_mask, egg_count):
    # Save results
    save_image(image, "output/adjusted_image.jpg")
    save_image(binary_mask, "output/binary_mask.jpg")

    # Display results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(binary_mask, cmap="gray")
    ax[0].set_title("Binary Mask (Thresholded)")
    ax[0].axis("off")

    ax[1].imshow(image)
    ax[1].set_title(f"Detected Eggs: {egg_count}")
    ax[1].axis("off")

    plt.show()

    print(f"Total Eggs Counted: {egg_count}")