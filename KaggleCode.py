# -*- coding: utf-8 -*-
#%% Importing Libraries
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, Adamax, Nadam

#%% Data Preprocessing
#Reading the data from the respective csv files
TrainingSet = pd.read_csv('train.csv', sep = ',')
ValidationSet = pd.read_csv('Dig-MNIST.csv', sep=',')
#The Xtrain is a greyscale image (28x28 matrix) of the number in kannada 
Xtrain = TrainingSet.iloc[:, 1:]
Xtrain = np.asarray(Xtrain)
#Ytrain contains the information of the value (number value in kannada)
Ytrain = TrainingSet.iloc[:, 0:1]

XValid = ValidationSet.iloc[:, 1:]
XValid = np.asarray(XValid)

YValid = ValidationSet.iloc[:, 0:1]

#We want the Y dataset to take the form of categorical data
# if the number is 1, then the dataset is [1,0,0,0,0,0,0,0,0,0]
#To create the data in this format we follow the steps below, for both training and validation data
TempVector = [0 for i in range(0, 10)]
Templist = [[] for i in range(len(Ytrain))]

for j in range(len(Ytrain)):
    Templist[j] = TempVector
Templist = np.asarray(Templist)
for j in range(len(Templist)):
    Templist[j, Ytrain.iloc[j,0]] = 1

Ytrain = Templist

Templist2 = [[] for i in range(len(YValid))]
for j in range(len(YValid)):
    Templist2[j] = TempVector
Templist2 = np.asarray(Templist2)
for j in range(len(Templist2)):
    Templist2[j, YValid.iloc[j,0]] = 1
    
YValid = Templist2

#%% Building the deep learning model
model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #Stocastic Gradient Descent Algorithm# Accuracy = 0.1
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False) #Adam Optimiser, similar to SGD but Accuracy = 0.97
adama = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999) #It is a variant of Adam based on the infinity norm, Accuracy = 0.98
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999) #Much like Adam is essentially RMSprop with momentum, Nadam is RMSprop with Nesterov momentum, Accuracy = 0.993
model.compile(optimizer= adama, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(Xtrain, Ytrain, epochs=25, batch_size=32)
score = model.evaluate(XValid, YValid, batch_size=32)
#%% Testing the Model
TestSet = pd.read_csv('test.csv', sep = ',', index_col=['id']) 

results = [[] for i in range(len(TestSet))]
for i in range(len(TestSet)):
    results[i] = model.predict(TestSet.iloc[i:i+1,:].values)