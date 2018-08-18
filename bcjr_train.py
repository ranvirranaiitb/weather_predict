
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
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.wrappers import  Bidirectional
from keras.callbacks import LearningRateScheduler
from keras import regularizers
import pickle
import commpy.channelcoding.convcode as cc
from commpy.utilities import *

import scipy.io as sio
import h5py

import sys

import time

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print 'use python -W ignore xxx.py to ignore warnings'

################################
# First Run: Run bcjr_trEx.py to Generate Training Examples
################################


with open('bcjr_trEX_0.00136885636175_SNRidx_1_BL_100_BN_1000.pickle') as f:
    step_of_history, k, iterations_number, ts_train, ts_test, nb_errors, BER = pickle.load(f)

# Network Parameters
num_epoch = 1
learning_rate = 1e-2                # one suggestion is LR 0.001 and batch size 10
regularizer = regularizers.l2(0.000)

train_batch_size = 400              # Many2Many better use small batch (10) # 200 good. 
test_batch_size  = 100
dropout_rate = 1.0                   # Dropout !=1.0 doesn't work!

code_rate   =  3
N               = k*code_rate        # length of coded bit


# Rx Decoder Parameters
rx_direction          = 'bd'         #'bd', 'sd'
rx_type               = 'rnn-gru'#'rnn-lstm'   #'cnn', 'rnn-lstm', 'rnn-gru', 'fc-nn'
num_rx_layer          = 2            # >= 1
num_hunit_rnn_rx      = 200#0#200#200          # rnn param
rx_kernel_size        = 7            # cnn param
rx_num_filter         = 100            # cnn param

# print parameters
print '*'*100
print 'Message bit is ', k
print 'learning rate is ', learning_rate
print 'batch size is ', train_batch_size
print 'step of history is ', step_of_history
print 'dropout is ', dropout_rate
print 'The RNN has ', rx_direction, rx_type, 'with ',num_rx_layer, 'layers with ', num_hunit_rnn_rx, ' unit'
print 'We are using regularizer ', regularizer.get_config()
print '*'*100

# Setup LR decay
def scheduler(epoch):

    if epoch > 3 and epoch <=4:
        print 'changing by /10 lr'
        lr = learning_rate/10.0
    elif epoch >4 and epoch <=5:
        print 'changing by /100 lr'
        lr = learning_rate/100.0
    elif epoch >5 and epoch <=7:
        print 'changing by /1000 lr'
        lr = learning_rate/1000.0
    elif epoch > 7:
        print 'changing by /10000 lr'
        lr = learning_rate/10000.0
    else:
        lr = learning_rate

    return lr

change_lr = LearningRateScheduler(scheduler)

# Hyeji's SNR Setup.
SNR_dB_start_Eb = -2
SNR_dB_stop_Eb = 3
SNR_points = 5


id = float(np.random.random())

SNR_dB_start_Eb = -2
SNR_dB_stop_Eb = 3
SNR_points = 5

SNR_dB_start_Es = SNR_dB_start_Eb + 10*np.log10(float(k)/float(2*k))
SNR_dB_stop_Es = SNR_dB_stop_Eb + 10*np.log10(float(k)/float(2*k))

sigma_start = np.sqrt(1/(2*10**(float(SNR_dB_start_Es)/float(10))))
sigma_stop = np.sqrt(1/(2*10**(float(SNR_dB_stop_Es)/float(10))))


test_sigmas = np.linspace(sigma_start, sigma_stop, SNR_points, dtype = 'float32')
SNRS = -10*np.log10(test_sigmas**2)


##########################################
# Data Preprocessing and Data Generation
#########################################

'''
tic = time.time()

#execfile('bcjr_turbo.py')
with open('0dB_bcjr100_train.pickle') as f:  # Python 3: open(..., 'rb')
    ts_train, ts_test = pickle.load(f)


print ts_train.shape
print ts_test.shape

t_totalc = k/step_of_history
trSNR = 0
if trSNR<=0:
    matfile = 'IT'+ str(6)+'_NB'+ str(t_totalc) + '_BL' + str(step_of_history)+'_SNR_neg'+str(abs(trSNR))+'.mat'
else:
    matfile = 'IT'+ str(6)+'_NB'+ str(t_totalc) + '_BL' + str(step_of_history)+'_SNR'+str(abs(trSNR))+'.mat'
    


#mat_load = sio.loadmat(matfile)
mat_load = h5py.File(matfile)
ts_train = mat_load['ts_train']
ts_train = np.transpose(ts_train)

ts_test = mat_load['ts_test']
ts_test = np.transpose(ts_test)
ts_test = ts_test.reshape([ts_test.shape[0],ts_test.shape[1],ts_test.shape[2],1])

tb_ber = mat_load['tb12_ber']
print 'train ber (0dB)', tb_ber


# ts_train = 

# ts_test =


toc = time.time()

print('time to generate codewords:', toc-tic)

'''

#########################################
# Train on low snr
#########################################
inputs = Input(shape=(step_of_history, code_rate)) # LL1, sys, par1

# AWGN between Tx and Rx
#def awgn_channel_tr(x):
#    res = x + 1.0*tf.random_normal(tf.shape(x),dtype=tf.float32, mean=0., stddev=1.0)
#    return res
#Rx_direct_received = Lambda(awgn_channel_tr, name = 'Direct_Link_channel')(inputs)

Rx_received = inputs #Rx_direct_received

# Rx Decoder
if rx_type == 'cnn':
    x = Rx_received
    for layer in range(num_rx_layer):
        x = Conv1D(filters=rx_num_filter, kernel_size=rx_kernel_size, strides=1, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

elif rx_type == 'rnn-lstm' or rx_type == 'rnn':
    x = Rx_received
    for layer in range(num_rx_layer):
        if rx_direction == 'bd':
            x = Bidirectional(LSTM(units=num_hunit_rnn_rx, activation='tanh',
                                   kernel_regularizer=regularizer,recurrent_regularizer=regularizer,
                                   return_sequences=True, dropout=dropout_rate))(x)
            x = BatchNormalization()(x)
        else:
            x = LSTM(units=num_hunit_rnn_rx, activation='tanh',
                     kernel_regularizer=regularizer,recurrent_regularizer=regularizer,
                     return_sequences=True, dropout=dropout_rate)(x)
            x = BatchNormalization()(x)

elif rx_type == 'rnn-gru':
    x = Rx_received
    for layer in range(num_rx_layer):
        if rx_direction == 'bd':
            x = Bidirectional(GRU(units=num_hunit_rnn_rx, activation='tanh',
                                  kernel_regularizer=regularizer,recurrent_regularizer=regularizer,
                                  return_sequences=True, dropout=dropout_rate))(x)
            x = BatchNormalization()(x)
        else:
            x = GRU(units=num_hunit_rnn_rx, activation='tanh',
                    kernel_regularizer=regularizer,recurrent_regularizer=regularizer,
                    return_sequences=True, dropout=dropout_rate)(x)
            x = BatchNormalization()(x)

else:
    print 'not supported'


def errors(y_true, y_pred):
    myOtherTensor = K.not_equal(K.round(y_true), K.round(y_pred))
    return K.mean(tf.cast(myOtherTensor, tf.float32))

predictions = TimeDistributed(Dense(1, activation='sigmoid', kernel_regularizer=regularizer))(x)

model = Model(inputs=inputs, outputs=predictions)
optimizer= keras.optimizers.adam(lr=learning_rate)
model.compile(optimizer=optimizer,loss='mean_squared_error', metrics=[errors])
print(model.summary())

# Build Data Format

input_prob = False # Input is LOG LIKELIHOOD, so False
output_prob = True # Output is Probability, so True

select_all = True
select_layer = 1 # used only if select_all = False

out_sum = True # Whether to use true Posterior LL or Sum of Prior + Posterior. 

t_iter = 12 # How many layers to use ( ex. 12: 6 iterations of Turbo)

if select_all == True: #Shrink all (turbo) layer examples
    ts_train_select = ts_train[0:t_iter,:,:,:]
    ts_test_select = ts_test[0:t_iter,:,:,:]
    ts_train_select = ts_train_select.reshape((int(t_iter*k/step_of_history),step_of_history,3))
    ts_test_select = ts_test_select.reshape((int(t_iter*k/step_of_history),step_of_history,1))
    
else:
    ts_train_select = ts_train[select_layer,:,:,:]
    ts_test_select = ts_test[select_layer,:,:,:]

if input_prob == True:
    ts_train_select[:,:,2] = math.e**ts_train_select[:,:,2]*1.0/(1+math.e**ts_train_select[:,:,2])

if out_sum == True:
    X_train_select = ts_test_select[:,:,0] + ts_train_select[:,:,2]
else:
    X_train_select = ts_test_select[:,:,0]
    

if output_prob == True:
    X_train_select[:,:] = math.e**X_train_select[:,:]*1.0/(1+math.e**X_train_select[:,:])


train_tx = ts_train_select.reshape(int(t_iter*k/step_of_history),step_of_history,3)
X_train = X_train_select.reshape(int(t_iter*k/step_of_history),step_of_history,1)


print train_tx.shape
print X_train.shape

###########################
# Start training!
###########################

id = str(np.random.random())


model.fit(x=train_tx, y=X_train, batch_size=train_batch_size,
          callbacks=[change_lr],
          epochs=num_epoch, validation_split = 0.01)#validation_data=(test_tx, X_test))  # starts training

#model.save_weights('./bcjr_train100_layer'+str(select_layer)+'_'+id+'.h5')
model.save_weights('./bcjr_train100_truePostLL_'+id+'_1.h5')
print 'watch out!'
print './'+ matfile + id+'_1.h5'



model.fit(x=train_tx, y=X_train, batch_size=train_batch_size,
          callbacks=[change_lr],
          epochs=num_epoch, validation_split = 0.01)#validation_data=(test_tx, X_test))  # starts training

#model.save_weights('./bcjr_train100_layer'+str(select_layer)+'_'+id+'.h5')
model.save_weights('./bcjr_train100_truePostLL_'+id+'_2.h5')
print 'watch out!'
print './'+ matfile + id+'_2.h5'




model.fit(x=train_tx, y=X_train, batch_size=train_batch_size,
          callbacks=[change_lr],
          epochs=num_epoch, validation_split = 0.01)#validation_data=(test_tx, X_test))  # starts training

#model.save_weights('./bcjr_train100_layer'+str(select_layer)+'_'+id+'.h5')
model.save_weights('./bcjr_train100_truePostLL_'+id+'_3.h5')
print 'watch out!'
print './'+ matfile + id+'_3.h5'




def scheduler2(epoch):

    if epoch >= 0 and epoch <=4:
        print 'changing by /10 lr'
        lr = learning_rate/10.0
    elif epoch >4 and epoch <=5:
        print 'changing by /100 lr'
        lr = learning_rate/100.0
    elif epoch >5 and epoch <=7:
        print 'changing by /1000 lr'
        lr = learning_rate/1000.0
    elif epoch > 7:
        print 'changing by /10000 lr'
        lr = learning_rate/10000.0
    else:
        lr = learning_rate

    return lr

change_lr2 = LearningRateScheduler(scheduler2)


model.fit(x=train_tx, y=X_train, batch_size=train_batch_size,
          callbacks=[change_lr2],
          epochs=num_epoch, validation_split = 0.01)#validation_data=(test_tx, X_test))  # starts training

#model.save_weights('./bcjr_train100_layer'+str(select_layer)+'_'+id+'.h5')
model.save_weights('./bcjr_train100_truePostLL_'+id+'_4.h5')
print 'watch out!'
print './'+ matfile + id+'_4.h5'




model.fit(x=train_tx, y=X_train, batch_size=train_batch_size,
          callbacks=[change_lr2],
          epochs=num_epoch, validation_split = 0.01)#validation_data=(test_tx, X_test))  # starts training

#model.save_weights('./bcjr_train100_layer'+str(select_layer)+'_'+id+'.h5')
model.save_weights('./bcjr_train100_truePostLL_'+id+'_5.h5')
print 'watch out!'
print './'+ matfile + id+'_5.h5'

