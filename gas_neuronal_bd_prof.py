import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.metrics import confusion_matrix

ruta=  "C:/Users/ACER/Desktop/DEEP LEARNING CLASS/"
BaseAumentada = pd.read_csv("C:/Users/Usuario/Downloads/BaseLD.csv")

#----------------------------------------------------------------Seteo de parametros iniciales
base = BaseAumentada.values # 30162 * 14
m = BaseAumentada.shape[0] #30162
n = BaseAumentada.shape[1] #14
num_neu = 3 # son dos neuronas, los que ganan mas de 50K y los que no
pertenencia = np.zeros(m) #(30162,)
# neuronas = np.zeros((num_neu, n)) #2*14

num_epochs = 10
h = np.zeros((1,num_neu)) #1*2 matriz donde guardare las distancias
neuronas = pd.concat([BaseAumentada.min(), BaseAumentada.max(),BaseAumentada.mean() ], axis=1).T.values #2*30
eta_0 = 1
eta_f = 0.0001

rho_0 = 1
rho_f = 0.0001

count_cluster0 = 0
count_cluster1 = 0
count_cluster2 = 0

for k in range(num_epochs):
    for i in range(0,  m):
        for j in range(0, 3): #num de nueronas

            vector_xi = base[i,:]
            neurona_yj = neuronas[j,:]

            # Calculamos distancias y obtenemos el indice de la distancia menor
            dist = norm(neuronas - vector_xi, axis=1, ord=2)
            temp = dist.argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(dist))
            dist_rankeadas = ranks
            min_dist = np.argmin(dist)

            #calculo de funciones que necesitamos para actualizar la formula que actualiza las neuronas
            eta_t = eta_0 * (eta_f / eta_0)**(i / m)  # learning rate
            rho_t = rho_0 * (rho_f / rho_0)**(i / m)  # neighborhood width

            lst_hr = []
            for r_i in dist_rankeadas:
                h_r = np.exp(- r_i / rho_t)
                lst_hr.append(h_r)

            # distancia, distancias ordenadas, valor de hr con base a los valores anteriores
            valores_neurona = np.matrix([dist, dist_rankeadas, np.array(lst_hr)])

            h_r_neu_0 = valores_neurona[-1,0]
            neuronas[0, :] = neuronas[0, :] + eta_t * h_r_neu_0 * (base[i, :] - neuronas[0, :])
            if min_dist == 0 :
                count_cluster0 = count_cluster0 +1
                pertenencia[i] = 0

            h_r_neu_1 = valores_neurona[-1, 1]
            neuronas[1, :] = neuronas[1, :] + eta_t * h_r_neu_1 * (base[i, :] - neuronas[1, :])
            if min_dist == 1 :
                count_cluster1 = count_cluster1 +1
                pertenencia[i] = 1

            h_r_neu_2 = valores_neurona[-1, 2]
            neuronas[2, :] = neuronas[2, :] + eta_t * h_r_neu_2 * (base[i, :] - neuronas[2, :])
            if min_dist == 2:
                count_cluster2 = count_cluster2 + 1
                pertenencia[i] = 2

    k = k+1

print('pertenencia', pertenencia)
print('count_cluster0', count_cluster0 )
print('count_cluster1', count_cluster1 )
print('count_cluster2', count_cluster2 )
resultado = pd.Series(pertenencia).value_counts()
print(resultado)

