import matplotlib
# matplotlib.use('Agg')
from pynvml import *
import numpy as np
import os
from keras import Model, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from matplotlib import cm
from tensorboard.plugins.hparams import keras
import numpy as np
from keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
config = tf.ConfigProto(allow_soft_placement=True,
                        device_count={'CPU': 1,
                                      'GPU': 0}
                        )

session = tf.Session(config=config)
K.set_session(session)
# 第几块gpu
inited = False
handle = None
load_list = ['convolution_2d_forward_VALID', 'convolution_backward_filter_2d_VALID',
             'convolution_backward_data_2d_VALID',
             'convolution_2d_forward_SAME', 'convolution_backward_filter_2d_SAME', 'convolution_backward_data_2d_SAME',
             'dropout_forward', 'dropout_backward', 'bn_forward_pre_activation',
             'bn_backward_pre_activation', 'pooling_2d_forward_max', 'pooling_2d_backward_max',
             'pooling_2d_forward_mean',
             'pooling_2d_backward_mean', 'broadcast_to_NHWC',
             'broadcast_to_NCHW', 'reduce_sum_new_NHWC', 'reduce_sum_new_NCHW', 'cross', 'cross_backward', 'adam_mv',
             'adam_compute',
             'activation_forward_relu', 'activation_backward_relu', 'activation_forward_softmax',
             'activation_backward_softmax',
             'matrix_multiply', 'matrix_elementwise_add', 'concat_forward', 'concat_a_backward',
             'concat_b_backward', 'sgd_update', 'matrix_elementwise_multiply_by_const', 'array_set']


# inputsshape是[],对应new_node.inputs = [node_A, node_B]的shape
def getinputsofmodel(node, inputsshape):
    if node.name == "Convolution2DForward":
        if node.padding == "VALID":
            opname = 'convolution_2d_forward_VALID'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], inputsshape[1][0],
                             inputsshape[1][2], node.u]
        if node.padding == "SAME":
            opname = 'convolution_2d_forward_SAME'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], inputsshape[1][0],
                             inputsshape[1][2], node.u]
    if node.name == "Convolution2DBackward":
        if node.padding == "VALID":
            if node.type == 0:
                opname = 'convolution_backward_data_2d_VALID'
                inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], inputsshape[1][0],
                                 inputsshape[1][2], node.u]
            if node.type == 1:
                opname = 'convolution_backward_filter_2d_VALID'
                inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], inputsshape[1][0],
                                 inputsshape[1][2], node.u]
        if node.padding == "SAME":
            if node.type == 0:
                opname = 'convolution_backward_data_2d_SAME'
                inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], inputsshape[1][0],
                                 inputsshape[1][2], node.u]
            if node.type == 1:
                opname = 'convolution_backward_filter_2d_SAME'
                inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], inputsshape[1][0],
                                 inputsshape[1][2], node.u]
    if node.name == "DropoutForward":
        opname = 'dropout_forward'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], node.dropout]
    if node.name == "FullyDropoutForward":
        opname = 'dropout_forward'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1], 1, node.dropout]
    if node.name == "DropoutBackward":
        opname = 'dropout_backward'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], node.dropout]
    if node.name == "FullyDropoutBackward":
        opname = 'dropout_backward'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1], 1, node.dropout]
    if node.name == "BNForward":
        opname = 'bn_forward_pre_activation'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2]]
    if node.name == "FullyBNForward":
        opname = 'bn_forward_pre_activation'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1], 1]
    if node.name == "BNBackward":
        opname = 'bn_backward_pre_activation'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2]]
    if node.name == "FullyBNBackward":
        opname = 'bn_backward_pre_activation'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1], 1]
    if node.name == "Pooling2DForward":
        if node.poolingMode == "max":
            opname = 'pooling_2d_forward_max'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], node.filter_h, node.u]
        if node.poolingMode == "mean":
            opname = 'pooling_2d_forward_mean'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], node.filter_h, node.u]
    if node.name == "Pooling2DBackward":
        if node.inputs[2].poolingMode == "max":
            opname = 'pooling_2d_backward_max'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], node.inputs[2].filter_h,
                             node.inputs[2].u]
        if node.inputs[2].poolingMode == "mean":
            opname = 'pooling_2d_backward_mean'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2], node.inputs[2].filter_h,
                             node.inputs[2].u]
    if node.name == "BroadcastTo":
        if node.type == "NHWC":
            opname = 'broadcast_to_NHWC'
            inputsofmodel = [inputsshape[1][0], inputsshape[1][1]]
        if node.type == "NCHW":
            opname = 'broadcast_to_NCHW'
            inputsofmodel = [inputsshape[1][0], inputsshape[1][1], inputsshape[1][2]]
    if node.name == "BroadcastToGradient":
        if node.type == "NHWC":
            opname = 'reduce_sum_new_NHWC'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1]]
        if node.type == "NCHW":
            opname = 'reduce_sum_new_NCHW'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2]]
    if node.name == 'ReduceSumOp':
        opname = 'reduce_sum_new_NHWC'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1]]
    if node.name == "CrossOp":
        opname = 'cross'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1]]
    if node.name == "CrossBackwardOp":
        opname = 'cross_backward'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1]]
    # 这个node的计算用了两个gpuop
    if node.name == "AdamOp":
        opname0 = 'adam_mv'
        if (len(inputsshape[0]) == 2):
            inputsofmodel0 = [inputsshape[0][1], inputsshape[0][0], 1]
        elif (len(inputsshape[0]) == 1):
            inputsofmodel0 = [1, inputsshape[0][0], 1]
        else:
            inputsofmodel0 = [inputsshape[0][1], inputsshape[0][0], inputsshape[0][2]]
        opname1 = 'adam_compute'
        return opname0, opname1, inputsofmodel0
    if node.name == "ActivationForward":
        if node.activationMode == "relu":
            opname = 'activation_forward_relu'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2]]
        if node.activationMode == "softmax":
            opname = 'activation_forward_softmax'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1] * inputsshape[0][2]]
    if node.name == "FullyActivationForward":
        if node.activationMode == "relu":
            opname = 'activation_forward_relu'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], 1]
        if node.activationMode == "softmax":
            opname = 'activation_forward_softmax'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1]]
    if node.name == "ActivationBackward":
        if node.activationMode == "relu":
            opname = 'activation_backward_relu'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], inputsshape[0][2]]
        if node.activationMode == "softmax":
            opname = 'activation_backward_softmax'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1] * inputsshape[0][2]]
    if node.name == "FullyActivationBackward":
        if node.activationMode == "relu":
            opname = 'activation_backward_relu'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1], 1]
        if node.activationMode == "softmax":
            opname = 'activation_backward_softmax'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1]]
    if node.name == "MatMul":
        opname = 'matrix_multiply'
        n1 = inputsshape[0][0]
        n2 = inputsshape[0][1]
        n3 = inputsshape[1][1]
        if node.matmul_attr_trans_A == True:
            n1 = inputsshape[0][1]
            n2 = inputsshape[0][0]
        if node.matmul_attr_trans_B == True:
            n3 = inputsshape[1][0]
        inputsofmodel = [n1, n2, n3]
    if node.name == "+":
        opname = 'matrix_elementwise_add'
        n = inputsshape[0][0]
        m = inputsshape[0][1]
        if len(inputsshape[0]) == 4:
            n = inputsshape[0][0] * inputsshape[0][1]
            m = inputsshape[0][2] * inputsshape[0][3]
        inputsofmodel = [n, m]
    if node.name == "ConcatForward":
        opname = 'concat_forward'
        inputsofmodel = [inputsshape[0][0], inputsshape[0][1]]
    if node.name == "Concatbackward":
        if node.type == 0:
            opname = 'concat_a_backward'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1]]
        if node.type == 1:
            opname = 'concat_b_backward'
            inputsofmodel = [inputsshape[0][0], inputsshape[0][1]]
    if node.name == "SgdOp":
        opname = 'sgd_update'
        inputsofmodel = [inputsshape[0][1], inputsshape[0][0], inputsshape[0][2]]
    if node.name == "*" or node.name == "Flatten" or node.name == "FlattenGradient" or node.name == "Squeeze" \
            or node.name == "SqueezeGradient":
        opname = 'matrix_elementwise_multiply_by_const'
        n = inputsshape[0][0]
        m = inputsshape[0][1]
        if len(inputsshape[0]) == 4:
            n = inputsshape[0][0] * inputsshape[0][1]
            m = inputsshape[0][2] * inputsshape[0][3]
        inputsofmodel = [m, n]
    if node.name == "Zeroslike" or node.name == "Oneslike":
        opname = 'array_set'
        n = inputsshape[0][0]
        m = 1
        if len(inputsshape[0]) == 2:
            m = inputsshape[0][1]
        if len(inputsshape[0]) == 4:
            n = inputsshape[0][0] * inputsshape[0][1]
            m = inputsshape[0][2] * inputsshape[0][3]
        inputsofmodel = [m, n]
    return opname, inputsofmodel


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


def gettime(node, inputsshape):
    global inited
    global handle
    if not inited:
        i = int(os.environ['CUDA_VISIBLE_DEVICES'])
        print("Now on GPU" + str(i))
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(i)
        inited = True
    tmp = nvmlDeviceGetUtilizationRates(handle)
    tmp = float(tmp.gpu)
    list = getinputsofmodel(node, inputsshape)
    if len(list) == 2:
        opname = list[0]
        list[1].insert(0, tmp)
        inputsofmodel = np.array(list[1])
        file_handle = open('../../res/data_bn/' + opname + '_mean_and_std.txt', mode='r')
        model = load(opname, len(file_handle.readlines()) + 1)
        file_handle.close()
        time = model.predict(inputsofmodel.reshape(1, inputsofmodel.shape[0]), verbose=1)
    if len(list) == 3:
        opname1 = list[0]
        opname2 = list[1]
        list[2].insert(0, tmp)
        inputsofmodel = np.array(list[2])
        file_handle1 = open('../../res/data_bn/' + opname1 + '_mean_and_std.txt', mode='r')
        model1 = load(opname1, len(file_handle1.readlines()) + 1)
        file_handle1.close()
        time1 = model1.predict(inputsofmodel.reshape(1, inputsofmodel.shape[0]), verbose=1)
        file_handle2 = open('../../res/data_bn/' + opname2 + '_mean_and_std.txt', mode='r')
        model2 = load(opname2, len(file_handle2.readlines()) + 1)
        file_handle2.close()
        time2 = model2.predict(inputsofmodel.reshape(1, inputsofmodel.shape[0]), verbose=1)
        time = time1 + time2
    return time[0][0]
