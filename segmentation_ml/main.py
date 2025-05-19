import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import norm

from plotting import plot_gaussian_models

def gmm_from_histogram(histograma, n_componentes):
    # Asegurarse de que el histograma es un arreglo 1D
    histograma = histograma.flatten()
    
    # Crear un arreglo de valores de intensidad (0 a 255)
    bins = np.arange(len(histograma))

    # Expandir el histograma en datos repetidos según la cuenta de cada bin
    datos = np.repeat(bins, histograma.astype(int))

    # Reestructurar los datos para que sean compatibles con scikit-learn
    datos = datos.reshape(-1, 1)

    # Verificar si hay datos suficientes para ajustar el GMM
    if len(datos) < n_componentes:
        raise ValueError("No hay suficientes datos para el número de componentes especificado.")

    # Ajustar el Modelo de Mezcla Gaussiana
    gmm = GaussianMixture(n_components=n_componentes, covariance_type='diag', random_state=0)
    gmm.fit(datos)

    # Obtener las medias y desviaciones estándar
    means = gmm.means_.flatten()
    std_devs  = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    return means, std_devs, weights


def segmentar_gm(imagen, mu, sigma, multiplicador_sigma=1.0):
    """
    Segmenta una imagen en escala de grises dejando en blanco los píxeles cuya
    intensidad esté dentro del rango [mu - k*sigma, mu + k*sigma].

    Parámetros:
    - imagen: ndarray, imagen en escala de grises.
    - mu: float, valor medio (media) de la intensidad.
    - sigma: float, desviación estándar.
    - multiplicador_sigma: float, número de desviaciones estándar a considerar (default=1.0).

    Retorna:
    - imagen_segmentada: ndarray, imagen binaria con píxeles blancos dentro del rango.
    """
    if len(imagen.shape) != 2:
        raise ValueError("La imagen debe estar en escala de grises (2D)")

    # Calcular límites
    rango_inferior = mu - multiplicador_sigma * sigma
    rango_superior = mu + multiplicador_sigma * sigma

    # Crear máscara
    mascara = cv2.inRange(imagen, int(rango_inferior), int(rango_superior))

    # Crear imagen segmentada
    imagen_segmentada = np.zeros_like(imagen)
    imagen_segmentada[mascara > 0] = 255

    return imagen_segmentada


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Error al cargar la imagen: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess_image(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_blur = image_gray.copy()
    #image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    image_blur = cv2.bilateralFilter(image_gray, 5, 75, 75, borderType= cv2.BORDER_REPLICATE)
    return image_blur



def fill_holes_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    for i, c in enumerate(contours):
        cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)
    return filled


def postprocess_mask(mask, kernel_size=5  ):
    # Aplicar operaciones morfológicas para limpiar la máscara
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

    return mask_cleaned

def main(image_dir_path, output_dir_path, file_name):
    # Cargar la imagen
    image_path =  f'{image_dir_path}/{file_name}'
    output_path = f'{output_dir_path}/{file_name}'

    image_raw = load_image(image_path )

    # Preprocesar la imagen
    imagen = preprocess_image(image_raw)

    # get histogram of the image
    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])
    
    
    #  # Ajustar el GMM al histograma
    n_componentes = 3  # Número de componentes GMM
    means, std_devs, weights = gmm_from_histogram(histograma, n_componentes)
   
    #plot_gaussian_models(means, std_devs, weights, save_path=f'./segmentation_ml/g_models/{file_name}')
    

    cluster_select = np.argmin(means)  # Seleccionar el cluster con la media más baja

    mask = segmentar_gm(imagen, means[cluster_select], std_devs[cluster_select], 1.2)

    mask = postprocess_mask(mask)

    mask = fill_holes_contours(mask)

    # Save the mask
    cv2.imwrite(output_path, mask)
    


for i in range(0, 4):
    IMAGE = f'NORMOBLASTOS_0{i}.jpg'
    main('./segmentation_ml/dataset', './segmentation_ml/masks', IMAGE)
