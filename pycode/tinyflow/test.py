import numpy as np
from pycode.tinyflow import ndarray
import time
val = np.random.normal(loc=0, scale=0.1, size=(2, 3, 244, 244))
x = ndarray.array(val, ndarray.gpu(0))
time.sleep(20)