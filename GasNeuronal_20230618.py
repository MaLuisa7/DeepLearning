import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.metrics import confusion_matrix

'''
Dataset : Kohavi,Ron. (1996). Census Income. UCI Machine Learning Repository. https://doi.org/10.24432/C5GP7S.


'''
# ----------------------------------------------------------------lectura de inputs
ruta=  "C:/Users/Usuario/Documents/Deep Learning/adult.csv"
BaseAumentada = pd.read_csv(ruta)

# ----------------------------------------------------------------data preprocessing
BaseAumentada.info()
# df = BaseAumentada
''' No nans
'age'          -ok-   ok numerica cont 
'workclass',   -nok-   '?', 'Private', 'State-gov', 'Federal-gov', 'Self-emp-not-inc',
                   'Self-emp-inc', 'Local-gov', 'Without-pay', 'Never-worked'               -- dropare los ?
'fnlwgt',      -ok-  ok numerica cont 
'education',   -ok-  'HS-grad', 'Some-college', '7th-8th', '10th', 'Doctorate',
                   'Prof-school', 'Bachelors', 'Masters', '11th', 'Assoc-acdm',
                   'Assoc-voc', '1st-4th', '5th-6th', '12th', '9th', 'Preschool'
'education_num',-ok- ok num discrete
'marital_status', - ok- ['Widowed', 'Divorced', 'Separated', 'Never-married',
                             'Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']
    'occupation', -nok- ['?', 'Exec-managerial', 'Machine-op-inspct', 'Prof-specialty',
                       'Other-service', 'Adm-clerical', 'Craft-repair',
                       'Transport-moving', 'Handlers-cleaners', 'Sales',
                       'Farming-fishing', 'Tech-support', 'Protective-serv',
                       'Armed-Forces', 'Priv-house-serv']                                     -- dropare los ?
'relationship', -ok - ['Not-in-family', 'Unmarried', 'Own-child', 'Other-relative',
                     'Husband', 'Wife'
'race', -ok - ['White', 'Black', 'Asian-Pac-Islander', 'Other',
                 'Amer-Indian-Eskimo']
'sex', -ok-['Female', 'Male']
'capital_gain', -ok- num disc - creo
'capital_loss',-ok- num disc - creo
'hours_per_week', - ok - num disc
 'native_country', -            ['United-States', '?', 'Mexico', 'Greece', 'Vietnam', 'China',
                               'Taiwan', 'India', 'Philippines', 'Trinadad&Tobago', 'Canada',
                               'South', 'Holand-Netherlands', 'Puerto-Rico', 'Poland', 'Iran',
                               'England', 'Germany', 'Italy', 'Japan', 'Hong', 'Honduras', 'Cuba',
                               'Ireland', 'Cambodia', 'Peru', 'Nicaragua', 'Dominican-Republic',
                               'Haiti', 'El-Salvador', 'Hungary', 'Columbia', 'Guatemala',
                               'Jamaica', 'Ecuador', 'France', 'Yugoslavia', 'Scotland',
                               'Portugal', 'Laos', 'Thailand', 'Outlying-US(Guam-USVI-etc)'], --- drop ?
'income'  - ok - ['<=50K', '>50K'] -- label so i'll drop it 
  '''
# cambiar categorias por numeros
# quitar ?
# quitar etiquetas

BaseAumentada.replace('?', np.nan, inplace=True)
BaseAumentada.dropna(inplace=True)


col_cat = ['workclass', 'education', 'marital_status',
           'occupation','relationship','race', 'sex',  'native_country', 'income']
for name_col in col_cat:
    BaseAumentada[name_col] =  BaseAumentada.loc[:,name_col].astype('category').cat.codes
y = BaseAumentada.iloc[:,-1]
BaseAumentada = BaseAumentada.iloc[:,:-1] #le quito la ultima que seria la label

#----------------------------------------------------------------Seteo de parametros iniciales
base = BaseAumentada.values # 30162 * 14
m = BaseAumentada.shape[0] #30162
n = BaseAumentada.shape[1] #1
num_neu = 2 # son dos neuronas, los que ganan mas de 50K y los que no
pertenencia = np.zeros(m) #(30162,)
neuronas = np.zeros((num_neu, n)) #2*14

num_epochs = 10
h = np.zeros((1,num_neu)) #1*2 matriz donde guardare las distancias
neuronas = pd.concat([BaseAumentada.quantile(.25), BaseAumentada.quantile(.75)], axis=1).T.values #2*30
eta_0 = 0.1
eta_f = 0.0001

rho_0 = 1
rho_f = 0.0001#num_neu / num_epochs / 2  #0.375

count_cluster0 = 0
count_cluster1 = 0
count_cluster2 = 0

for k in range(num_epochs):
    for i in range(0,  m) :
        for j in range(0, 1):
            vector_xi = base[i,:]
            neurona_yj = neuronas[j,:]

            # Calculamos distnacias y obtenemos el indice de la distancia menor
            dist = norm(neuronas-vector_xi, axis=1, ord=2)
            min_val = np.argmin(dist)

            #calculo de funciones que necesitamos para actualizar la formula que actualiza las neuronas
            eta_t = eta_0 * (eta_f / eta_0)**(i / m) #learning rate
            rho_t = rho_0 * (rho_f / rho_0)**(i / m) #neighborhood width
            r = k
            h_r = np.exp(- r / rho_t) # h(r) = exp(- r / rho_t) , donde r_k(t)

            if 0 == min_val :
                neuronas[0, :] = neuronas[0, :] + eta_t * h_r * (base[i, :] - neuronas[0, :])
                count_cluster0 = count_cluster0 +1
                pertenencia[i] = 0

            if 1 == min_val :
                neuronas[1, :] = neuronas[1, :] + eta_t * h_r * (base[i, :] - neuronas[1, :])
                count_cluster1 = count_cluster1 +1
                pertenencia[i] = 1

            # if 2 == min_val :
            #     neuronas[2, :] = neuronas[2, :] + eta_t * h_r * (base[i, :] - neuronas[2, :])
            #     count_cluster2 = count_cluster2 +1
            #     pertenencia[i] = 2
    k = k+1

print('pertenencia', pertenencia)
print('count_cluster0', count_cluster0 )
print('count_cluster1', count_cluster1 )
print('count_cluster2', count_cluster2 )

resultado = pd.Series(pertenencia).value_counts()
print(resultado)
# 0.0    22237
# 1.0     7925

cf = confusion_matrix(y, pertenencia)
print(cf)
# [[16582  6072]
#  [ 5655  1853]]

tn, fp, fn, tp =  cf.ravel()
acc = (tn + tp) / (m)

# 61% con num_epochs = 4
# 61% con num_epochs = 10