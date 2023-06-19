#### K Means

# STEP 2. ALGORITHM.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ruta=  "C:/Users/ACER/Desktop/DEEP LEARNING CLASS/"
BaseAumentada = pd.read_csv("C:/Users/Usuario/Downloads/BaseLD.csv")

m = BaseAumentada.shape[0]
n = BaseAumentada.shape[1]
pertenencia = np.zeros(m)
neuronas = np.zeros((3, n))
promedio = np.zeros((1, n))
minimo = np.zeros((1, n))
maximo = np.zeros((1, n))

# Initialize the locations.
for i in range(n):
    minimo[0, i] = np.min(BaseAumentada.iloc[:, i])

for i in range(n):
    maximo[0, i] = np.max(BaseAumentada.iloc[:, i])

for i in range(n):
    promedio[0, i] = np.mean(BaseAumentada.iloc[:, i])

neuronas[0, :] = minimo[0, :] # 3 x30
neuronas[1, :] = promedio[0, :]
neuronas[2, :] = maximo[0, :]

# Synaptic Potential
h = np.zeros((1, 3)) # 1 * 3
base = BaseAumentada.values #27751 * 30
accum = np.zeros((3, n)) # 3 * 30

# Cluster
cluster = np.zeros((1, 3))

for k in range(3):
    cluster = np.zeros((1, 3))

    for i in range(m): # 0 hasta 27750
        for j in range(3): # 0 hasta 2
            h[0, j] = np.dot(base[i, :], neuronas[j, :]) - 0.5 * (np.dot(neuronas[j, :], neuronas[j, :])) #distancia euclidiana

        un = 0.5 * (1 - i / m) #tasa de aprendizaje n(t), donde n(0) = 0.5 y (1 - t/T) = (1 - i / m)
        max_val = np.max(h[0, :])

        if h[0, 0] == max_val:
            neuronas[0, :] = neuronas[0, :]  # + un * (base[i, :] - neuronas[0, :])
            cluster[0, 0] += 1
            pertenencia[i] = 1
            accum[0, :] = (accum[0, :] + base[i, :]) / cluster[0, 0]

        if h[0, 1] == max_val:
            neuronas[1, :] = neuronas[1, :]  # + un * (base[i, :] - neuronas[1, :])
            cluster[0, 1] += 1
            pertenencia[i] = 2
            accum[1, :] = (accum[1, :] + base[i, :]) / cluster[0, 1]

        if h[0, 2] == max_val:
            neuronas[2, :] = neuronas[2, :]  # + un * (base[i, :] - neuronas[2, :])
            cluster[0, 2] += 1
            pertenencia[i] = 3
            accum[2, :] = (accum[2, :] + base[i, :]) / cluster[0, 2]

    k += 1


###
resultado = pd.Series(pertenencia).value_counts()

print(resultado)
# 1.0    13793
# 2.0    10845
# 3.0     3113