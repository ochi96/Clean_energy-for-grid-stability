import os
import numpy as np 
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Flatten,Dense,Embedding
from keras import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle

import pandas as pd 
from sklearn.preprocessing import StandardScaler

def read_train_data():
    train_df = pd.read_csv('power_train.csv')
    print('Training data')
    print(train_df.iloc[:10])
    return train_df

def read_test_data(train_df,test_df = pd.read_csv('power_test.csv')):
    print('Test data')
    print(test_df.iloc[:10])
    data = [train_df,test_df]
    return data


def normalize(data):
    [train_df,test_df] = data
    train_data = train_df.drop(['PE'],axis=1)
    train_targets = train_df['PE']
    test_data = test_df.drop(['PE'],axis=1)
    test_targets = test_df['PE']
    data2=[train_data,test_data] 
    data3=[]
    for dataset in data2:
        sc = StandardScaler()
        dataset_numerical_feaures = list(dataset.select_dtypes(include=['int64','float64','int32','float32']).columns)
        dataset_scaled = pd.DataFrame(data=dataset)
        dataset_scaled[dataset_numerical_feaures] = sc.fit_transform(dataset_scaled[dataset_numerical_feaures])
        data3.append(dataset)
    
    [train_data,test_data] = data3
    return train_data,train_targets,test_data,test_targets

def build_model(train_data):
    #building the achitecture
    model = models.Sequential()
    model.add(layers.Dense(16,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(1))

    #compiling...ie showing the loss function(depending on output), how it will be minimised(optimiser)
    model.compile(optimizer='rmsprop',loss='mae',metrics=['mae','mse'])
    #mse=mean squared error,mae = mean absolute error
    return model

def kfoldvalidation(data):
    train_data,train_targets,test_data,test_targets = data
    model = build_model(train_data)
    history = model.fit(train_data,train_targets,epochs=100,batch_size=25,verbose=0)
    loss,test_mae, test_mse = model.evaluate(test_data,test_targets)
    print(test_mae)
    print(test_mse)
    
    test_predictions = model.predict(test_data).flatten()
    pickle.dump(model,open('model.pkl','wb'))
    error = test_predictions-test_targets
    
    plt.hist(error,bins=15)
    plt.xlabel('prediction error')
    _ = plt.ylabel('count')
    plt.show()
    
    plt.scatter(test_targets, test_predictions)
    plt.xlabel('true values')
    plt.ylabel('predictions')
    plt.xlim([420,500])
    plt.ylim([420,500])
    _ = plt.plot([-2,510],[-2,510])
    plt.show()

    return history

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('mean absolute error')
    plt.plot(hist['epoch'],hist['mean_absolute_error'],label='train_error')
    plt.ylim([0,25])
    plt.legend()

    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('mean squared error')
    plt.plot(hist['epoch'],hist['mean_squared_error'],label='train_error')
    plt.ylim([0,60])
    plt.legend()
    graph = plt.show()

    return graph

training_set = read_train_data()
training_and_test_set = read_test_data(training_set)
norm = normalize(training_and_test_set)
results = kfoldvalidation(norm)
training_progress = plot_history(results)
