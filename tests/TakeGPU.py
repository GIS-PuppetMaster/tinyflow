import os
import yagmail
from multiprocessing import *
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from pynvml import *
import time
import numpy as np
nvmlInit()


def run(gpu):
    import time
    import torch
    import setproctitle
    setproctitle.setproctitle('myProc')
    gpu = f'cuda:{gpu}'
    a = torch.ones((100000, 100), device=gpu)
    b = torch.ones((100, 12000), device=gpu)
    d = []
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    # cublasHandle = gpu_op.create_cublasHandle()
    # W1_val = np.ones((10000, 100))
    #
    # X_val = np.ones((10000, 10000))
    # ctx = ndarray.gpu(0)
    #
    # z = ndarray.array(np.ones((100000, 20000)), ctx)
    #
    #
    # W1_val = ndarray.array(W1_val, ctx=ctx)
    #
    # X_val = ndarray.array(X_val, ctx=ctx)
    #
    # out_val = ndarray.empty((10000, 100), ctx=ctx)
    print(f'launching on gpu:{gpu}')
    while True:
        d.append(torch.matmul(a, b))
        if len(d) > 1:
            d.clear()
        # gpu_op.matrix_multiply(X_val, False, W1_val, False, out_val, cublasHandle)
        time.sleep(np.random.random_sample()/10)


already_took = {}
while True:
    try:
        with open('../release.json') as f:
            release = json.load(f)
    except Exception as e:
        release = []
    for i in range(16):
        handle = nvmlDeviceGetHandleByIndex(i)
        flag = []
        for _ in range(5):
            tmp = nvmlDeviceGetUtilizationRates(handle)
            flag.append(tmp.gpu == 0 and tmp.memory <=10)
            time.sleep(1)
        if False not in flag:
            if i not in already_took.keys() and i not in release:
                p = Process(target=run, args=(i,), name=f'GPU:{i}')
                already_took[i] = p
                p.start()
                try:
                    yag = yagmail.SMTP(user='741481546@qq.com', password="sclcmmgzvylkbfic", host='smtp.qq.com')
                    yag.send(to='741481546@qq.com', subject=f'Took GPU:{i}',
                             contents=f'Took GPU:{i}')
                except Exception as e:
                    pass
        elif i in release and i in already_took.keys():
            already_took[i].terminate()
            already_took.pop(i)
    # time.sleep(10)
