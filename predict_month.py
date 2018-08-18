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
from tqdm import tqdm

from keras.models import model_from_json

json_file1 = open('model2_temp.json', 'r')
loaded_model_json1 = json_file1.read()
json_file1.close()
loaded_model_temp = model_from_json(loaded_model_json1)
# load weights into new model
loaded_model_temp.load_weights("model2_temp.h5")
print("Loaded temperature model from disk")

json_file2 = open('model2_pressure.json', 'r')
loaded_model_json2 = json_file2.read()
json_file2.close()
loaded_model_pressure = model_from_json(loaded_model_json2)
# load weights into new model
loaded_model_pressure.load_weights("model2_pressure.h5")
print("Loaded pressure model from disk")

json_file3 = open('model2_hum.json', 'r')
loaded_model_json3 = json_file3.read()
json_file3.close()
loaded_model_hum = model_from_json(loaded_model_json3)
# load weights into new model
loaded_model_hum.load_weights("model2_hum.h5")
print("Loaded humidity model from disk")

json_file4 = open('model2_main.json', 'r')
loaded_model_json4 = json_file4.read()
json_file4.close()
loaded_model_main = model_from_json(loaded_model_json4)
# load weights into new model
loaded_model_main.load_weights("model2_main.h5")
print("Loaded main model from disk")

json_file5 = open('model2_rain.json', 'r')
loaded_model_json5 = json_file5.read()
json_file5.close()
loaded_model_rain = model_from_json(loaded_model_json5)
# load weights into new model
loaded_model_rain.load_weights("model2_rain.h5")
print("Loaded rain model from disk")

weather = np.load("weather.npy")

main_dict = {}
main_dict['Clear'] = 0
main_dict['Clouds'] = 1
main_dict['Rain'] = 2

predict_input_len = 40
predict_output_len = 8

weather_cache_list = []


#Prepping weather_cache
for i in range(len(weather)):
	input_list = []
	for j in range(len(weather[i])-predict_input_len,len(weather[i])):
		temp_list = [weather[i][j][3],weather[i][j][6],weather[i][j][7],main_dict[weather[i][j][9]],weather[i][j][13]]
		input_list.append(temp_list)
	weather_cache_list.append(input_list)

for future in tqdm(range(240)):
	for i in range(len(weather_cache_list)):
		input_list = []
		for j in range(len(weather_cache_list[i])-predict_input_len,len(weather_cache_list[i])):
			temp_list = [weather_cache_list[i][j][0]/50,weather_cache_list[i][j][1]/1000,weather_cache_list[i][j][2]/50,weather_cache_list[i][j][3]/2,weather_cache_list[i][j][4]]
			input_list.append(temp_list)		
		temp_nn_input = np.array(input_list)
		temp_nn_input = np.expand_dims(temp_nn_input,axis=0)
		temp_output = loaded_model_temp.predict(temp_nn_input)*50
		pressure_output = loaded_model_pressure.predict(temp_nn_input)*1000
		hum_output = loaded_model_hum.predict(temp_nn_input)*50
		main_output = loaded_model_main.predict(temp_nn_input)*2
		rain_output = loaded_model_rain.predict(temp_nn_input)
		for k in range(predict_output_len):
			temp1 = main_output[0][k]
			temp1 = round(temp1)
			if (temp1 <0):
				temp1 =0
			elif(temp1>2):
				temp1 =2
			main_output[0][k] = temp1
		for k in range(predict_output_len):
			temp1 = [temp_output[0][k],pressure_output[0][k],hum_output[0][k],main_output[0][k],rain_output[0][k]]
			weather_cache_list[i].append(temp1)

print(len(weather_cache_list))
print(len(weather_cache_list[0]))
print(len(weather_cache_list[0][0]))
print("done")

store_array = np.array(weather_cache_list)
np.save('weather_predict1.npy', store_array)

