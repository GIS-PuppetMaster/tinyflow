import os
import keras
from keras.applications import VGG16
import numpy as np
import multiprocessing
from multiprocessing import Process


def train(i):
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{i}"
    image = np.ones((1000, 224, 224, 3))
    labels = np.ones((1000,1000))
    model = VGG16(weights=None, include_top=True)
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_squared_error)
    model.fit(image,labels,epochs=100000)

pool = []
gpu = [2,4,6]
for i in gpu:
    pool.append(Process(target=train, args=(i,)))
    pool[-1].start()
for p in pool:
    p.join()


