import matplotlib
# matplotlib.use('Agg')
import numpy as np
import os
from keras import Model, models
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from matplotlib import cm
from tensorboard.plugins.hparams import keras



def create_model(n):
    model = Sequential()
    model.add(Dense(units=2048, activation='tanh', input_dim=n))
    model.add(Dense(units=2048, activation='tanh'))
    model.add(Dense(units=1, activation='relu'))
    return model


def load(opname, n):
    model = create_model(n)
    model.load_weights('../../res/model_parameter/' + opname + '_model.hdf5', by_name=True, skip_mismatch=True)
    return model


load_list = ['convolution_2d_forward_VALID', 'convolution_backward_filter_2d_VALID',
             'convolution_backward_data_2d_VALID',
             'convolution_2d_forward_SAME', 'convolution_backward_filter_2d_SAME', 'convolution_backward_data_2d_SAME',
             'dropout_forward', 'dropout_backward', 'broadcast_to_NHWC',
             'broadcast_to_NCHW', 'reduce_sum_new_NHWC', 'reduce_sum_new_NCHW',
             'bn_forward_pre_activation', 'bn_backward_pre_activation', 'activation_forward_relu',
             'activation_backward_relu', 'activation_forward_softmax', 'activation_backward_softmax',
             'pooling_2d_forward_max', 'pooling_2d_backward_max', 'pooling_2d_forward_mean',
             'pooling_2d_backward_mean', 'matrix_multiply', 'matrix_elementwise_multiply_by_const',
             'matrix_elementwise_add',
             'array_set', 'concat_forward', 'concat_a_backward',
             'concat_b_backward', 'sgd_update', 'cross', 'cross_backward', 'adam_mv', 'adam_compute']

for opname in load_list:
    print(opname, end=" ")
    file_handle = open('../../res/data_bn/' + opname + '_mean_and_std.txt', mode='r')
    print(len(file_handle.readlines()) + 1)
    # model = load(opname, len(file_handle.readlines()) + 1)
    file_handle.close()
print('finish')
