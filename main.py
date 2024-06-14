import os

import cv2
import joblib
import pandas as pd
import sympy as sp
from train_model import calcular_momentos_hu

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
            caracter = imagen_original[y:y + h, x:x + w]
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


def concatenar_numero_lista(lista):
    mapeo = {
        "uno": "1",
        "dos": "2",
        "tres": "3",
        "cuatro": "4",
        "cinco": "5",
        "seis": "6",
        "siete": "7",
        "ocho": "8",
        "nueve": "9",
        "division": "/",
        "resta": "-",
        "suma": "+",
        "multiplicacion": "*"
    }

    cadena = "".join(mapeo[elemento] for elemento in lista if elemento in mapeo)
    return cadena


def calcular_expresion(expresion):
    resultado = sp.sympify(expresion)
    print(resultado)
    return resultado

if __name__ == '__main__':

    ruta_imagen = "test/p1.png"
    carpeta = "image_output"

    # Cargar el modelo y el scaler
    model = joblib.load('model/model.pkl')
    scaler = joblib.load('model/scaler.pkl')

    # Ejemplo de uso
    dividir_imagen(ruta_imagen, 'image_output')

    lista_prediccion = []

    # Obtener la lista de archivos en el directorio y ordenarlos alfabéticamente
    archivos = sorted(os.listdir(carpeta))

    for archivo in archivos:
        ruta_imagen = os.path.join(carpeta, archivo)
        if os.path.isfile(ruta_imagen):
            print("Procesando imagen: ", ruta_imagen)
            imagen = cv2.imread(ruta_imagen)
            momentos_hu = calcular_momentos_hu(imagen)
            # Convertir los momentos de Hu a un DataFrame para que tenga nombres de características
            momento_image_df = pd.DataFrame([momentos_hu], columns=[f'MomentoHu{i + 1}' for i in range(7)])
            # Escalar los momentos de Hu
            momento_image_scaled = scaler.transform(momento_image_df)
            # Realizar la predicción
            prediction = model.predict(momento_image_scaled)
            print("prediction: ", prediction[0])
            lista_prediccion.append(prediction[0])
    print(lista_prediccion)
    expression = concatenar_numero_lista(lista_prediccion)
    print("El resultado es: ", calcular_expresion(expression))
