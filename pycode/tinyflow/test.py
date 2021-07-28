import os
gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
from pycode.tinyflow import ndarray
from pycode.tinyflow.gpu_op import *
import numpy as np
import time
s = create_cudaStream()
cudnn = create_cudnnHandle(s)
cblas = create_cublasHandle(s)
SetCudaMemoryLimit(1000)
ctx = ndarray.gpu(0)
a = []
for _ in range(500):
    a.append(ndarray.array(np.ones(1000000,),ctx=ctx))
print('finish')
time.sleep(100)