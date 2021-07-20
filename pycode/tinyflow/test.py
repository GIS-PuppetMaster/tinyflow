import os
import sys

import numpy as np
from pycode.tinyflow import ndarray

sys.path.append('../../')
from tests.Experiment import VGG16_test, ResNet50_test, DenseNet_test, InceptionV3_test, InceptionV4_test
import random, time

from tests.Experiment.log.result import get_result, get_vanilla_max_memory
import pickle as pkl
from line_profiler import LineProfiler
from pycode.tinyflow.TrainExecuteAdam_vDNNconv import TrainExecutor as vdnnExecutor
from pycode.tinyflow.TrainExecuteAdam_Capu import TrainExecutor as CapuchinExecutor
from pycode.tinyflow.TrainExecuteAdam import TrainExecutor as VanillaTrainExecutor

gpu = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

t = 1


def main():
    vgg16 = VGG16_test.VGG16(num_step=50, type=t, batch_size=16, gpu_num=gpu, path='test', file_name='test', n_class=1000, need_tosave=0)
    vgg16.run()


if __name__ == '__main__':
    profiler = LineProfiler()
    if t == 0:
        profiler.add_function(VanillaTrainExecutor.run)
    elif t == 1:
        vanilla_max_memory = 3218.875
        bud = vanilla_max_memory * (1 - 0.3520873066327611)
        # 总显存=预算+need_tosave(额外占用空间)
        need_tosave = 11019 - bud
        # if net_id == 2 and i == 3:
        #     need_tosave -= 500
        print(f'need_tosave:{need_tosave}')
        need_tosave_list = []
        need_tosave_list.append(need_tosave)
        outspace = []
        size = need_tosave * 1e6 / 4
        gctx = ndarray.gpu(0)
        while size > 0:
            if size > 10000 * 10000:
                outspace.append(ndarray.array(np.ones((10000, 10000)) * 0.01, ctx=gctx))
                size -= 10000 * 10000
            else:
                need_sqrt = int(pow(size, 0.5))
                if need_sqrt <= 0:
                    break
                outspace.append(ndarray.array(np.ones((need_sqrt, need_sqrt)) * 0.01, ctx=gctx))
                size -= need_sqrt * need_sqrt
        print('finish extra matrix generation')
        profiler.add_function(CapuchinExecutor.run)
        profiler.add_function(CapuchinExecutor.clear)
    else:
        profiler.add_function(vdnnExecutor.run)

    profiler_wrapper = profiler(main)
    res = profiler_wrapper()
    profiler.print_stats()
    # main()
