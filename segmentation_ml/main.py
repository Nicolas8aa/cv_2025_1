import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import norm

from plotting import plot_gaussian_models


def calcular_histograma(imagen, canal):
    """
    Calcula y grafica el histograma del canal especificado en la imagen.

    Parámetros:
    - imagen: ndarray, imagen cargada con cv2.imread().
    - canal: str, uno de 'gray', 'b', 'g', 'r'.

    Retorna:
    - hist: ndarray, histograma del canal seleccionado.
    """
    # Diccionario de índices para los canales
    canales_dict = {'b': 0, 'g': 1, 'r': 2, 'gray': 0}

    # Verificar número de canales en la imagen
    num_canales = 1 if len(imagen.shape) == 2 else imagen.shape[2]

    # Verificar validez del canal
    if canal == 'gray' and num_canales > 1:
        hist = cv2.calcHist([imagen], [0], None, [256], [0, 256])
    elif canal in ('b', 'g', 'r') and num_canales == 3:
        idx = canales_dict[canal]
        hist = cv2.calcHist([imagen], [idx], None, [256], [0, 256])
    else:
        raise ValueError("Canal no válido para esta imagen.")

    return hist


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
    #image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    return image_gray


def postprocess_mask(mask):
    # Aplicar operaciones morfológicas para limpiar la máscara
    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

    return mask_cleaned

def main(image_path, output_path):
    # Cargar la imagen
    image_raw = load_image(image_path)

    # Preprocesar la imagen
    imagen = preprocess_image(image_raw)

    # get histogram of the image
    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])
    
    
    #  # Ajustar el GMM al histograma
    n_componentes = 2  # Número de componentes GMM
    means, std_devs, weights = gmm_from_histogram(histograma, n_componentes)
   
    #plot_gaussian_models(means, std_devs, weights)
    

    cluster_select =1  # Seleccionar el primer cluster (0 o 1)
    mask = segmentar_gm(imagen, means[cluster_select], std_devs[cluster_select], 1)

    mask = postprocess_mask(mask)

    # Save the mask
    cv2.imwrite(output_path, mask)
    


for i in range(0, 1):
    IMAGE = f'NORMOBLASTOS_0{i}.jpg'
    main(f'./segmentation_ml/dataset/{IMAGE}', f'./segmentation_ml/masks/{IMAGE}')
