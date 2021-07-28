import os
gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
from pycode.tinyflow import ndarray
from pycode.tinyflow.gpu_op import *
import numpy as np
import time
SetCudaMemoryLimit(1000)
# s = create_cudaStream()
# cudnn = create_cudnnHandle(s)
# cblas = create_cublasHandle(s)
ctx = ndarray.gpu(0)
a = []
for _ in range(500):
    a.append(ndarray.array(np.ones(10000000,),ctx=ctx))
print('finish')
time.sleep(100)