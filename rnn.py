#changing from tanh to sigmoid, or using adam instead of rmsprop doesn't seem to change the result much
#it doesn't seem that benchmarking with epochs really help. Mean values don't help

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
model.compile(optimizer="rmsprop",loss="MSE")
model.fit(x_train,y_train, batch_size=20, epochs=300)

'''
test_data=pd.read_csv('Google_Stock_Price_Test.csv')
test_open=test_data.iloc[:,1:2]
test_open = sc.fit_transform(test_open)

x_test=test_open[0:1257,]
y_test=test_open[1:1258,]

x_test=np.reshape(x_test,(20,1,1))
y_pred = model.predict(x_test)
'''
#remember, we can only predict from the previous days...therefore the first/second day doesn't have much data
###########################################################################################
#previously we only predicted 20 data points. Now, it's over 1200+ days
full_y_pred = model.predict(x_train)
x_train = np.reshape(x_train,(1257,1))
########################
#to graph

def graph(real_open, y_pred):
    y_pred = sc.inverse_transform(y_pred) #important, getting back the stock price from the standardscale
    real_open = sc.inverse_transform(real_open)
    plt.plot(real_open, color="black", label="Real Google Price")
    plt.plot(y_pred, color="green", label="predicted")
    plt.title("prediction with lstm")
    plt.xlabel("days")
    plt.ylabel("price")
    plt.legend(loc="upper left")
    plt.savefig('result')
    plt.show()
'''    
def benchmark(key,val_key,history_dict, title):
    plt.plot(history_dict[key])
    plt.plot(history_dict[val_key])
    
    plt.title(title)
    plt.yscale("log")
    plt.ylabel("mean abs % error")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.savefig(title)
    plt.show()
'''    
graph(x_train,full_y_pred)
#graph(test_open,y_pred) #bad naming scheme for test open, since this is all data, not just a test set
#rnn_accuracy_plt=benchmark("mean_absolute_percentage_error", "loss", model_history.history, title="rnn accuracy")