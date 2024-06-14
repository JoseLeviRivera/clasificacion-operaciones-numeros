import joblib
import m2cgen as m2c

# Cargar el modelo
model = joblib.load('../../model/model.pkl')

# Convertir el modelo a Java
java_code = m2c.export_to_java(model)

# Guardar el código Java en un archivo
with open('model/RandomForestClassifier.java', 'w') as f:
    f.write(java_code)

