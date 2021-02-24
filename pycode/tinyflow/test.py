""" library to take autodiff and execute a computation graph """
from __future__ import absolute_import

import threading
import time
import numpy as np
import random
import queue
import datetime
import time
import numpy as np
from pycode.tinyflow import ndarray, gpu_op, memoryManager, memoryManagerController
import random
import queue

# rand = np.random.RandomState(seed=123)
# W1_val = np.ones((78400, 4560))
# ctx_gpu = ndarray.gpu(0)
# ctx_cpu = ndarray.cpu(0)
# list_w1 = []
#
# w1 = ndarray.array(W1_val, ctx_cpu)
# w2 = ndarray.empty(w1.shape, ctx_gpu)
# w1.copyto(w2)
# print(w2.asnumpy())
# t1 = time.time()
#
# for i in range(100):
#     w1.copyto(w2)
#     t2 = time.time()
#     print(t2 - t1)
# print("success")
# for i in range(100):
#     list_w1.append(ndarray.array(W1_val, ctx_gpu))
#     del list_w1[i]

# time.sleep(10)


ctx_cpu = ndarray.cpu(0)
ctx_gpu = ndarray.gpu(0)
w1 = ndarray.empty((10000, 100000), ctx_cpu)
w2 = ndarray.empty((10000, 100000), ctx_gpu)


t1 = datetime.datetime.now()
w1.copyto(w2, None)
t2 = datetime.datetime.now()
print((t2 - t1).microseconds)

