import numpy as np
import pandas as pd
from keras.layers import LSTM, SimpleRNN, Dense, Activation
from keras.models import Sequential
from __future__ import print_function
import matplotlib.pyplot as plt
# %matplotlib in inline
from sklearn import preprocessing
import math ,time

path = "C:/Users/Usuario/Documents/Deep Learning/PrecioPetroleoDEF.csv"
sales = pd.read_csv(path ,sep =";")

plt.plot(sales.Fecha,sales.Mezcla_Mexicana)
plt.show()

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
SALESM = min_max_scaler.fit_transform(sales['Mezcla_Mexicana'].values.reshape(-1, 1))
SALESM.shape

#Separar conjunto de prueba y entrenamiento
train_size = int(len(SALESM) * 0.7)
test_size = len(SALESM) - train_size
train, test = SALESM[0:train_size,:], SALESM[train_size:len(SALESM),:]
print(len(train), len(test))

print(len(sales))
print(122 + 53)

# convert an array of values into a SALESM matrix
def create_matrix(SALESM, look_back=15):
    dataX, dataY = [], []
    for i in range(len(SALESM)-look_back-1):
        a = SALESM[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(SALESM[i + look_back,0])
    return np.array(dataX), np.array(dataY)

x_train, y_train = create_matrix(train, look_back=15)
x_test, y_test = create_matrix(test, look_back=15)
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

# create and fit the LSTM network
look_back = 15
model = Sequential()
model.add(LSTM(20, input_shape=(1, look_back))) # 20 neuronas
model.add(Dense(1)) #relu is the activation function default
model.compile(loss='mape', optimizer='adam')
model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)

trainPredict = model.predict(x_train)
testPredict = model.predict(x_test)


# invert predictions
trainPredict = min_max_scaler.inverse_transform(trainPredict)
trainY = min_max_scaler.inverse_transform([y_train])
testPredict = min_max_scaler.inverse_transform(testPredict)
testY = min_max_scaler.inverse_transform([y_test])

# shift train predictions for plotting
trainPredictPlot = np.empty_like(SALESM)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back,:] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(SALESM)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(SALESM)-1, :] = testPredict

# plot baseline and predictions
plt.plot(min_max_scaler.inverse_transform(SALESM))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


trainPredictPlot