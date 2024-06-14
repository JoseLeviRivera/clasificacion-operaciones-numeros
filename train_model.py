import csv
import os
import cv2
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

def agregar_fila_momentos_hu(nombre, momento, etiqueta):
    with open(nombre, "a", newline='') as csvfile:  # Modo 'a' para agregar filas
        writer = csv.writer(csvfile)
        writer.writerow(momento.tolist() + [etiqueta])  # Añadir la etiqueta al final de la fila

def procesar_carpeta(carpeta, archivo_csv, label):
    archivos = os.listdir(carpeta)
    for archivo in archivos:
        ruta_imagen = os.path.join(carpeta, archivo)
        if os.path.isfile(ruta_imagen):
            print("Procesando imagen: ", ruta_imagen)
            imagen = cv2.imread(ruta_imagen)
            momentos_hu = calcular_momentos_hu(imagen)
            print(momentos_hu)
            agregar_fila_momentos_hu(archivo_csv, momentos_hu, label)

def calcular_momentos_hu(imagen):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    momentos = cv2.moments(binaria)
    momentos_hu = cv2.HuMoments(momentos).flatten()
    return momentos_hu

def print_hi():
    print("Procesando imagenes ...")

if __name__ == '__main__':
    print_hi()
    archivo_csv = 'db.csv'
    if os.path.exists(archivo_csv):
        os.remove(archivo_csv)
    with open(archivo_csv, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f'MomentoHu{i + 1}' for i in range(7)] + ['Etiqueta'])

    etiquetas = {
        '1': "uno", '2': "dos", '3': "tres", '4': "cuatro", '5': "cinco",
        '6': "seis", '7': "siete", '8': "ocho", '9': "nueve", 'div': "division",
        '-': "resta", '+': "suma", 'mult': "multiplicacion"
    }

    for carpeta, etiqueta in etiquetas.items():
        procesar_carpeta(f'imagenes_numeros/{carpeta}', archivo_csv, etiqueta)

    data = pd.read_csv("db.csv")
    print(data.head())

    features = ["MomentoHu1","MomentoHu2","MomentoHu3","MomentoHu4","MomentoHu5","MomentoHu6","MomentoHu7"]
    clase = ["Etiqueta"]

    X = data[features]
    y = data[clase]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluar el modelo
    conf_matrix = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=0)

    print("Matriz de Confusión:")
    print(conf_matrix)
    print("\nAccuracy Score:")
    print(acc_score)
    print("\nClassification Report:")
    print(class_report)

    joblib.dump(model, 'model/model.pkl')
    # Guardar el scaler junto con el modelo
    joblib.dump(scaler, 'model/scaler.pkl')

    

