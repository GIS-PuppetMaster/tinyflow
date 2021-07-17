import datetime
from multiprocessing import Process

import pynvml

from pycode.tinyflow import ndarray, gpu_op
from pycode.tinyflow import autodiff as ad
from pycode.tinyflow import mainV2 as mp
import six.moves.cPickle as pickle
import gzip
import numpy as np
import os
import queue
import multiprocessing
import threading
import time

from tools import *
from agetinputsofmodel import *
import read_model



def test1():
    # 测试从node转化到opname
    W1 = ad.Variable(name="W1")
    W2 = ad.Variable(name="W2")
    W3 = ad.Variable(name="W3")
    W4 = ad.Variable(name="W4")
    W5 = ad.Variable(name="W5")
    W6 = ad.Variable(name="W6")
    b1 = ad.Variable(name="b1")
    b2 = ad.Variable(name="b2")
    b3 = ad.Variable(name="b3")
    b4 = ad.Variable(name="b4")
    b5 = ad.Variable(name="b5")
    b6 = ad.Variable(name="b6")
    X = ad.Placeholder(name="X")
    y_ = ad.Placeholder(name="y_")

    # 下面是三层网络的激活函数，两个relu和一个softmax

    # relu(X W1+b1)
    z2 = ad.dense(X, W1, b1)
    z3 = ad.fullyactivation_forward_op(z2, "NCHW", "relu")

    # relu(z3 W2+b2)
    z5 = ad.dense(z3, W2, b2)
    z6 = ad.fullyactivation_forward_op(z5, "NCHW", "relu")
    z7 = ad.dense(z6, W3, b3)
    z8 = ad.fullyactivation_forward_op(z7, "NCHW", "relu")
    z9 = ad.dense(z8, W4, b4)
    z10 = ad.fullyactivation_forward_op(z9, "NCHW", "relu")
    z11 = ad.dense(z10, W5, b5)
    z12 = ad.fullyactivation_forward_op(z11, "NCHW", "relu")
    # softmax(z5 W2+b2)
    z13 = ad.dense(z12, W6, b6)
    bn1 = ad.fullybn_forward_op(z13, "NCHW")
    y = ad.fullyactivation_forward_op(bn1, "NCHW", "softmax")
    loss = ad.crossEntropy_loss(y, y_)

    inputs = [[10, 10], [10, 10]]


def test2():
    # 测试从opname 得到结果
    tmp = ('dropout_forward', [32, 2048, 1, 0.8])
    opname = tmp[0]
    inputs_of_model = tmp[1]
    model = read_model.load(opname, 5)
    model.predict(inputs_of_model, verbose=1)


if __name__ == '__main__':
    test2()

