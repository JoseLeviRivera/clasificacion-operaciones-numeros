import cv2
import os


def preprocesar_imagen(imagen):
    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # Aplicar un umbral para binarizar la imagen
    _, binarizada = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY_INV)
    return binarizada

def encontrar_contornos(binarizada):
    contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos


def extraer_y_guardar_caracteres(imagen_original, contornos, directorio_salida):
    indice = 0
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        if w > 10 and h > 10:  # Filtrar contornos muy pequeños
            caracter = imagen_original[y:y+h, x:x+w]
            cv2.imwrite(f"{directorio_salida}/caracter_{indice}.png", caracter)
            indice += 1


def dividir_imagen(imagen_ruta, directorio_salida):
    # Crear el directorio de salida si no existe
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)

    # Leer la imagen
    imagen = cv2.imread(imagen_ruta)

    # Preprocesar la imagen
    binarizada = preprocesar_imagen(imagen)

    # Encontrar los contornos
    contornos = encontrar_contornos(binarizada)

    # Ordenar los contornos por su posición en x (de izquierda a derecha)
    contornos_ordenados = sorted(contornos, key=lambda c: cv2.boundingRect(c)[0])

    # Extraer y guardar los caracteres individuales
    extraer_y_guardar_caracteres(imagen, contornos_ordenados, directorio_salida)


# Ejemplo de uso
dividir_imagen('image_test/image.png', 'image_output')
