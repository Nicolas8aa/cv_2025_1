
import os
import cv2
import matplotlib.pyplot as plt


dirname =  os.path.dirname(__file__)
data_dir = os.path.join(dirname, '../data')

# I need to define the function that will save the preprocessed image
def save_image(image, path):
    path_to_save = os.path.join(data_dir, path)

    # print(f'saving image to {path_to_save}')
    cv2.imwrite(path_to_save, image)



def read_image(image_path, flag=cv2.IMREAD_COLOR):
    path_to_read = os.path.join(data_dir, image_path)
    return cv2.imread(path_to_read, flag)


def display_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def plot_lab_histogram(image , a_range = [0, 256], b_range = [0, 256], l_range = [0, 256], suffix = ""):
    """
    Plots the histogram of L, A, B channels to analyze color distribution.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Plot histograms
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(L.ravel(), bins=256, range=[0, 256], color='black')
    plt.title("L Channel Histogram")
    # Apply range
    plt.xlim(l_range)

    plt.subplot(1, 3, 2)
    plt.hist(A.ravel(), bins=256, range=[0, 256], color='red')
    plt.title("A Channel Histogram")
    plt.xlim(a_range)

    plt.subplot(1, 3, 3)
    plt.hist(B.ravel(), bins=256, range=[0, 256], color='blue')
    plt.title("B Channel Histogram")
    plt.xlim(b_range)

    # plt.show()

    # Save histograms
    plt.savefig(f"lab_histogram_{suffix}.png", dpi=300)

def resize_image(image, scale_percent):
    """
    Resize the image to a given scale percentage while keeping the aspect ratio.
    
      :parameters:
      image (numpy.ndarray): The input image to resize.
      scale_percent (float): The scale percentage to resize the image.
    
      :returns:
      numpy.ndarray: The resized image.
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image



