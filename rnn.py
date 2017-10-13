import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

training_data = pd.read_csv("Google_Stock_Price_Train.csv")
training_open = training_data.iloc[:,1:2] #needed to transform back for sklearn
#training_open = training_open.iloc[:,:].values
sc = preprocessing.MinMaxScaler((0,1)) #smashes it between 0 and 1
training_open = sc.fit_transform(training_open)

#input = stock price at time t
#output = stock price at time t+1

x_train=training_open[0:1257,]
y_train=training_open[1:1258,] #predicted time shifted by 1


timestamp=np.arange(0,1257)
timestamp = timestamp.reshape(1257,1)
x_train = np.reshape(x_train,(1257,1,1))
#3d, open price, timestamp, and feature-or stock time at time t


########
#RNN, need further optimization
model = Sequential()
#units=4, standard practice
model.add(LSTM(units=4,activation="tanh",input_shape=(None,1))) #units = number of memory cells
#(None,1) represents any timestep, 1 feature.

model.add(Dense(units=1, activation="sigmoid"))
model.compile(optimizer="rmsprop",loss="MSE",metrics=["MAE"])
model.fit(x_train,y_train, batch_size=20, epochs=200)


test_data=pd.read_csv('Google_Stock_Price_Test.csv')
test_open=test_data.iloc[:,1:2]
sc = preprocessing.MinMaxScaler((0,1))
