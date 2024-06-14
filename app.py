import os
import shutil

from flask import Flask, request, render_template, redirect, url_for
import cv2
import joblib
import pandas as pd
import sympy as sp
from train_model import calcular_momentos_hu

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Cargar el modelo y el scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')


def preprocesar_imagen(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binarizada = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY_INV)
    return binarizada


def encontrar_contornos(binarizada):
    contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos


def extraer_y_guardar_caracteres(imagen_original, contornos, directorio_salida):
    indice = 0
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        if w > 10 and h > 10:
            caracter = imagen_original[y:y + h, x:x + w]
            cv2.imwrite(f"{directorio_salida}/caracter_{indice}.png", caracter)
            indice += 1


def dividir_imagen(imagen_ruta, directorio_salida):
    imagen = cv2.imread(imagen_ruta)
    binarizada = preprocesar_imagen(imagen)
    contornos = encontrar_contornos(binarizada)
    contornos_ordenados = sorted(contornos, key=lambda c: cv2.boundingRect(c)[0])
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
    return resultado


def limpiar_directorio(directorio):
    for archivo in os.listdir(directorio):
        archivo_path = os.path.join(directorio, archivo)
        try:
            if os.path.isfile(archivo_path) or os.path.islink(archivo_path):
                os.unlink(archivo_path)
            elif os.path.isdir(archivo_path):
                shutil.rmtree(archivo_path)
        except Exception as e:
            print(f'Fallo al eliminar {archivo_path}. Motivo: {e}')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            carpeta = "image_output"
            dividir_imagen(filepath, carpeta)

            lista_prediccion = []
            archivos = sorted(os.listdir(carpeta))

            for archivo in archivos:
                ruta_imagen = os.path.join(carpeta, archivo)
                if os.path.isfile(ruta_imagen):
                    imagen = cv2.imread(ruta_imagen)
                    momentos_hu = calcular_momentos_hu(imagen)
                    momento_image_df = pd.DataFrame([momentos_hu], columns=[f'MomentoHu{i + 1}' for i in range(7)])
                    momento_image_scaled = scaler.transform(momento_image_df)
                    prediction = model.predict(momento_image_scaled)
                    lista_prediccion.append(prediction[0])

            expression = concatenar_numero_lista(lista_prediccion)
            resultado = calcular_expresion(expression)

            # Limpiar el directorio después de la clasificación
            limpiar_directorio(carpeta)

            return render_template('index.html', resultado=resultado, expression=expression)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
