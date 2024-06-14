import pandas as pd
import matplotlib.pyplot as plt

# Leer los datos del archivo CSV
data = pd.read_csv("../../db.csv")

# Contar las etiquetas
etiquetas = data['Etiqueta'].value_counts()
print(etiquetas)

# Graficar la distribución de las etiquetas
etiquetas.plot(kind='bar')
plt.xlabel('Etiqueta')
plt.ylabel('Frecuencia')
plt.title('Distribución de las Etiquetas')
plt.show()
