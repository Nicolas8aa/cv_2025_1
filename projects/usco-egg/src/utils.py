
import os
import cv2
import matplotlib as plt


dirname =  os.path.dirname(__file__)
data_dir = os.path.join(dirname, '../data')

# I need to define the function that will save the preprocessed image
def save_image(image, path):
    path_to_save = os.path.join(data_dir, path)

    print(f'saving image to {path_to_save}')
    cv2.imwrite(path_to_save, image)



def read_image(image_path):
    path_to_read = os.path.join(data_dir, image_path)
    return cv2.imread(path_to_read)


def display_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()




