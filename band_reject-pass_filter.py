import cv2 
import numpy as np
import matplotlib.pyplot as plt


def create_brf_mask_base(dft_shift, r_out=60, r_in=30):
    """
    Genera una máscara de filtro atenua de banda (notch)

    Parámetros:
    -----------
    dft_shift : ndarray
        Transformada de Fourier desplazada (2D).
    
    r_out : int, optional
        Radio exterior del filtro (por defecto es 60).

    r_in : int, optional
        Radio interior del filtro (por defecto es 30).

    Retorna:
    --------
    H : ndarray
        Máscara de filtro atenua banda de las mismas dimensiones que dft_shift.
    """
    # tamaño de la mascara del filtro
    rows, cols = dft_shift.shape[:2]
    crow, ccol = rows // 2 , cols // 2

    mask = np.ones((rows, cols, 2), np.uint8)


    # Crear el filtro 
    for u in range(rows):
      for v in range(cols):
          D = np.sqrt((u - crow)**2 + (v - ccol)**2)
          if r_in < D < r_out:
              mask[u, v] = 0

    # Filtro pasa altas usando el filtro gaussiano

    return mask.astype(np.float32)



def create_brf_mask(dft_shift, radius, width):
    
    """
    Generates a band-reject filter mask using Gaussian function.

    Args:
        dft_shift: The DFT of the image (shifted).
        radius: The radius of the band-reject filter.
        width: The width of the band-reject filter.

    Returns:
        The band-reject filter mask
    """

    rows, cols = dft_shift.shape[:2]
    center = (cols // 2, rows // 2)
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    mask = 1 - np.exp(-((distance - radius)**2) / (2 * width**2))

    mask = np.stack([mask, mask], axis=2)

    return mask.astype(np.float32)



def create_bpf_mask ( dft_shift,radius, width):
    """
    Generates a band-pass filter mask for the DFT of an image.

    Args:
        dft_shift: The DFT of the image (shifted).
        radius: The radius of the band-pass filter.
        width: The width of the band-pass filter.

    Returns:
        The band-pass filter mask.
    """

    return 1 - create_brf_mask(dft_shift, radius, width)

def filter_image(dft_shift, mask):
    """
    Applies a band-reject filter the image in the frequency domain.
    This function takes the DFT of an image and a band-reject filter mask,

    Args:
        dft_shift: The DFT of the image (shifted).
        mask: The band-reject filter mask.

    Returns:
        The filtered image as uint8.
    """
    filtered_dft = dft_shift * mask
    filtered_image = np.fft.ifftshift( filtered_dft)
    filtered_image = cv2.idft(filtered_image)
    filtered_image = cv2.magnitude(filtered_image[:, :, 0], filtered_image[:, :, 1])

    # Normalizar la imagen filtrada para que esté en el rango [0, 255]
    return  cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), filtered_dft





def main():
    # Cargar la imagen en escala de grises

    image = cv2.imread('./images/lena_gray.tif', cv2.IMREAD_GRAYSCALE)

    # Create a synthetic image (for demonstration purposes)
    # image = np.ones((256, 256), dtype=np.uint8)

    if image is None:
        print("Error: No se pudo cargar la imagen.")
        return 
    


    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)



    # Crear la máscara de filtro atenua banda
    # mask = create_brf_mask(dft_shift, r_out=60, r_in=30)
    # mask = gaussian_band_reject_mask(dft_shift, radius=30, width=10)
    mask = create_bpf_mask(dft_shift, radius=30, width=10)

    # Aplicar el filtro atenua banda
    filtered_image, filtered_dft= filter_image(dft_shift, mask)


    # Mostrar la imagen original y la imagen filtrada
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Imagen Original')
    plt.subplot(1, 3, 2),    plt.imshow(filtered_image, cmap='gray'), plt.title('Imagen Filtrada con Filtro Atenua Banda')
    plt.subplot(1, 3, 3), plt.imshow(mask[:, :, 0], cmap='gray'), plt.title('Mascara de Filtro Atenua Banda')
    plt.tight_layout()
    plt.show()


    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude_filtered = cv2.magnitude(filtered_dft[:, :, 0], filtered_dft[:, :, 1])
    magnitude_log = np.log1p(magnitude)
    magnitude_filtered_log = np.log1p(magnitude_filtered)


    # Plot original and filtered magnitude spectra (3D)
    rows, cols = image.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, magnitude_log, cmap='viridis')
    ax1.set_title('Original Frequency Magnitude (log scale)')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, magnitude_filtered_log, cmap='plasma')
    ax2.set_title('Filtered Frequency Magnitude (log scale)')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()