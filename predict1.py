# weather prediction algorithm
# author: Ranvir

import numpy as np
import math
import os

from keras import backend as K
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers import TimeDistributed, Flatten
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.wrappers import  Bidirectional
from keras.callbacks import LearningRateScheduler
from keras import regularizers
import pickle

import scipy.io as sio
import h5py
import sys
import time
import logging

from keras.models import model_from_json

def scheduler(epoch):

    if epoch > 3 and epoch <=4:
        print('changing by /10 lr')
        lr = learning_rate/10.0
    elif epoch >4 and epoch <=5:
        print('changing by /100 lr')
        lr = learning_rate/100.0
    elif epoch >5 and epoch <=7:
        print('changing by /1000 lr')
        lr = learning_rate/1000.0
    elif epoch > 7:
        print('changing by /10000 lr')
        lr = learning_rate/10000.0
    else:
        lr = learning_rate

    return lr

change_lr = LearningRateScheduler(scheduler)

def errors(y_true, y_pred):
    myOtherTensor = K.not_equal(K.round(y_true), K.round(y_pred))
    return K.mean(tf.cast(myOtherTensor, tf.float32))

regularizer = regularizers.l2(0.000)
#print 'We are using regularizer ', regularizer.get_config()
dropout_rate = 0.1
learning_rate = 1e-2
train_batch_size = 10
num_epoch = 10

input_array = np.load('input_weather.npy')
output_array = np.load('output_rain.npy')
output_array = np.squeeze(output_array)


inputs = Input(shape=(40, 5))
Rx_received = inputs
x = Rx_received
x = Bidirectional(LSTM(units=100, activation='tanh',
                       kernel_regularizer=regularizer,recurrent_regularizer=regularizer,
                       return_sequences=True, dropout=dropout_rate))(x)
x = BatchNormalization()(x)

predictions_intermediate = TimeDistributed(Dense(8, activation='tanh', kernel_regularizer=regularizer))(x)
predictions_intermediate2 = Flatten()(predictions_intermediate)
predictions = Dense(8,activation='linear', kernel_regularizer=regularizer)(predictions_intermediate2)

model = Model(inputs=inputs, outputs=predictions)
optimizer= keras.optimizers.adam(lr=learning_rate)
model.compile(optimizer=optimizer,loss='mean_squared_error', metrics=['accuracy'])
print(model.summary())

model.fit(x=input_array, y=output_array, batch_size=train_batch_size,
          callbacks=[change_lr],
          epochs=num_epoch, validation_split = 0.01) # starts training

#model.save_weights('./temperature_train.h5')
#print ('watch out!')

model_json = model.to_json()
with open("model2_rain.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model2_rain.h5")
print("Saved model to disk")

json_file = open('model2_rain.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model2_rain.h5")
print("Loaded model from disk")

temp_predict = loaded_model.predict(input_array[0].reshape(1,40,5))
print(temp_predict)
print(output_array[0])



