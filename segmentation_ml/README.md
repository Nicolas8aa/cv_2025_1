# Segmentación y machine learning

## Objetivo
Aplicar técnicas de segmentación no supervisada mediante modelos de mezclas gaussianas (Gaussian Mixture Model, GMM) para identificar y extraer los núcleos celulares en imágenes microscópicas.

## Descripción

Cada grupo deberá implementar un sistema en Python con OpenCV y scikit-learn que permita:
- Cargar una imagen microscópica en color.
- Aplicar preprocesamiento básico necesario (por ejemplo, filtrado, cambio de espacio de color, normalización, u otro).
- Implementar un algoritmo de segmentación utilizando GMM, clasificando los píxeles en al menos dos clases: fondo y núcleo.
- Generar y guardar la imagen segmentada, resaltando únicamente los núcleos celulares (máscaras binarias obtenidas).
- Se debe usar la implementación de GaussianMixture de sklearn.mixture (en Python).
- El sistema debe ser modular (dividir el código en funciones).
Suministrar sus respuestas utilizando el formulario adjunto (este se utilizara para la evaluación).