import multiprocessing
import os
import sys
import time
import traceback
from multiprocessing import Process
import os
import numpy as np

sys.path.append('../../')

from pycode.tinyflow import ndarray
from tests.Experiment import VGG16_test, ResNet50_test, DenseNet_test, InceptionV3_test, InceptionV4_test
from tests.Experiment.result import get_result, get_vanilla_max_memory

gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
net_names = ['VGG', 'InceptionV3', 'InceptionV4', 'ResNet', 'DenseNet']
budget = {
    'VGG': {2: 2642.0,
            16: 4099.333333333333},
    'InceptionV3': {2: 1173.3333333333333,
                    16: 2447.3333333333335},
    'InceptionV4': {
        2: 1444.0,
        16: 3788.6666666666665
    },
    'ResNet': {
        2: 1273.3333333333333, 16: 2212.0},
    'DenseNet': {
        2: 1130.0, 16: 2334.0
    }
}


def generate_job(num_step, net_id, type, batch_size, path, budget, file_name=""):
    if net_id == 0:
        vgg16 = VGG16_test.VGG16(num_step=num_step, type=type, batch_size=batch_size, gpu_num=gpu, path=path,
                                 file_name=file_name, n_class=1000, budget=budget)
        return vgg16
    elif net_id == 1:
        inceptionv3 = InceptionV3_test.Inceptionv3(num_step=num_step, type=type, batch_size=batch_size, gpu_num=gpu,
                                                   path=path, file_name=file_name, budget=budget)
        return inceptionv3
    elif net_id == 2:
        inceptionv4 = InceptionV4_test.Inceptionv4(num_step=num_step, type=type, batch_size=batch_size, gpu_num=gpu,
                                                   path=path, file_name=file_name, budget=budget)
        return inceptionv4
    elif net_id == 3:
        resNet = ResNet50_test.ResNet50(num_step=num_step, type=type, batch_size=batch_size, gpu_num=gpu, path=path,
                                        file_name=file_name, budget=budget)
        return resNet
    elif net_id == 4:
        denseNet = DenseNet_test.DenseNet121(num_step=num_step, type=type, batch_size=batch_size, gpu_num=gpu,
                                             path=path, file_name=file_name, budget=budget)
        return denseNet


def create_extra_matrix(need_tosave, pipe1, pipe2):
    outspace = []
    arr_size = need_tosave * pow(2, 20) / 4
    gctx = ndarray.gpu(0)
    while arr_size > 0:
        if arr_size > 100000000:
            outspace.append(ndarray.array(np.ones((100000000,), dtype=np.float32) * 0.01, ctx=gctx))
            arr_size -= 100000000
        else:
            arr_size = int(arr_size)
            outspace.append(ndarray.array(np.ones((arr_size,), dtype=np.float32) * 0.01, ctx=gctx))
            arr_size -= arr_size
    print('finish extra matrix generation')
    pipe1.put(True)
    while True:
        if not pipe2.empty():
            if pipe2.get():
                break
    for i in range(len(outspace) - 1, -1, -1):
        outspace.pop(i)


def Experiment1():
    for net_id in range(5):
        repeat_times = 3
        print("Experiment1 start")
        net_name = net_names[net_id]
        for i, num_net in enumerate([1, 1, 2, 3]):
            if i == 0:
                batch_size = 16
                net_name_ = net_name
            else:
                batch_size = 2
                net_name_ = net_name + f' x{i}'
            print("batch_size", batch_size)
            path = f'./log/{net_name_}/'
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)
            nets = []
            for _ in range(num_net):
                # net_id = random.randint(0, 4) #net_id随机选取网络种类 0:vgg16, 1:inceptionv3, 2:inceptionv4, 3:resNet, 4:denseNet
                nets.append(net_id)
            print("选取的网络", list(map(lambda x: net_names[x], nets)))
            for t in range(repeat_times):
                print(f'repeat_times:{t}')
                for type in range(3):  # type是调度方式的选择, 0.不调度 1.capuchin 2.vdnn
                    bud = 0
                    if type == 1:
                        bud = budget[net_name][batch_size]
                    job_pool = []
                    for i, net_id in enumerate(nets):
                        job_pool.append(
                            generate_job(num_step=50, net_id=net_id, type=type, batch_size=batch_size, path=path,
                                         file_name=f"_repeat_time={t}_net_order={i}", budget=bud))
                    for job in job_pool:
                        job.start()
                    for job in job_pool:
                        job.join()
            get_result(path, repeat_times=repeat_times)
            print("Experiment1 finish")


if __name__ == "__main__":
    Experiment1()
    # for net_name in net_names[3:]:
    #     for i, num_net in enumerate([1, 1, 2, 3]):
    #         if i == 0:
    #             batch_size = 16
    #             net_name_ = net_name
    #         else:
    #             batch_size = 2
    #             net_name_ = net_name + f' x{i}'
    #         try:
    #             get_result(f'./log/{net_name_}/', repeat_times=3, need_tosave=[])
    #         except Exception as e:
    #             print(net_name_)
    #             traceback.print_exc()