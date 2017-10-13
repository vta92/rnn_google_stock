import numpy as np
import matplotlib.pyplot as plt
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
test_open = sc.fit_transform(test_open)

x_test=test_open[0:1257,]
y_test=test_open[1:1258,]

x_test=np.reshape(x_test,(20,1,1))
y_pred = model.predict(x_test,batch_size=20)
#remember, we can only predict from the previous days...therefore the first/second day doesn't have much data


########################
#to graph

def graph(real_open, y_pred):
    y_pred = sc.inverse_transform(y_pred) #important, getting back the stock price from the standardscale
    real_open = sc.inverse_transform(real_open)
    plt.plot(real_open, color="black", label="Real Google Price")
    plt.plot(y_pred, color="blue", label="predicted")
    plt.title("prediction by lstm")
    plt.xlabel("dates")
    plt.ylabel("price")
    plt.legend(loc="upper left")
    plt.savefig('benchmark')
    plt.show()
    
graph(test_open,y_pred) #bad naming scheme for test open, since this is all data, not just a test set
