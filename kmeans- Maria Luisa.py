import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from numpy.linalg import norm

ruta=  "C:/Users/ACER/Desktop/DEEP LEARNING CLASS/"
BaseAumentada = pd.read_csv("C:/Users/Usuario/Downloads/BaseLD.csv")
base = BaseAumentada.values
m = BaseAumentada.shape[0] #27751
n = BaseAumentada.shape[1] #30
pertenencia = np.zeros(m) #(27751,)
neuronas = np.zeros((3, n)) #3*30
num_neu = 3
h = np.zeros((1,3)) #1*3 matriz donde guardare las distancias
neuronas = pd.concat([BaseAumentada.min(), BaseAumentada.mean(),
                      BaseAumentada.max()], axis=1).T.values #3*30
eta_0 = 0.5
count_cluster0 = 0
count_cluster1 = 0
count_cluster2 = 0

for k in range(3):
    for i in range(0,  m) : #i = 0 #
        for j in range(0, 1): #j = 0 #
            vector_xi = base[i,:]
            neurona_yj = neuronas[j,:]

            # dist = distance.euclidean(vector_xi , neurona_yj)
            dist = norm(neuronas-vector_xi, axis=1, ord=2)
            eta = eta_0 * (1 - i / m)
            min_val = np.argmin(dist)

            if 0 == min_val :
                neuronas[0, :] = neuronas[0, :] #+ eta * (base[i, :] - neuronas[0, :])
                count_cluster0 = count_cluster0 +1
                pertenencia[i] = 0

            if 1 == min_val :
                neuronas[1, :] = neuronas[1, :] #+ eta * (base[i, :] - neuronas[1, :])
                count_cluster1 = count_cluster1 +1
                pertenencia[i] = 1

            if 2 == min_val :
                neuronas[2, :] = neuronas[2, :] #+ eta * (base[i, :] - neuronas[2, :])
                count_cluster2 = count_cluster2 +1
                pertenencia[i] = 2
    k = k+1

print('pertenencia', pertenencia)
print('count_cluster0', count_cluster0 )
print('count_cluster1', count_cluster1 )
print('count_cluster2', count_cluster2 )

resultado = pd.Series(pertenencia).value_counts()
print(resultado)

# 0.0    13793
# 1.0    10845
# 2.0     3113