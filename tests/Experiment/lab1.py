import multiprocessing
import os
import sys
import traceback
from multiprocessing import Process

import numpy as np

sys.path.append('../../')
from pycode.tinyflow import ndarray
from tests.Experiment import VGG16_test, ResNet50_test, DenseNet_test, InceptionV3_test, InceptionV4_test
import random, time

from tests.Experiment.log.result import get_result, get_vanilla_max_memory
import pickle as pkl
import time

gpu = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
net_names = ['VGG', 'InceptionV3', 'InceptionV4', 'ResNet', 'DenseNet']
budget = {
    'VGG': {
        1: {2: 0.15293383270911362, 16: 0.3264440911499735},
        2: {2: 0.17408086923144292},
        3: {2: 0.2422420988339721}
    },
    'InceptionV3': {
        1: {2: 0.2967741935483871, 16: 0.5895117540687161},
        2: {2: 0.31210824492341316},
        3: {2: 0.2720903432502497}
    },
    'InceptionV4': {
        1: {2: 0.5022768670309654, 16: 0.647182313192941},
        2: {2: 0.4145428496730023},
        3: {2: 0.422244682222328}
    },
    'ResNet': {
        1: {2: 0.25741399762752076, 16: 0.6708817498291183},
        2: {2: 0.32907537538633647},
        3: {2: 0.2964702300613243}
    },
    'DenseNet': {
        1: {2: 0.3854850474106491, 16: 0.7596731033485211},
        2: {2: 0.27557542588794254},
        3: {2: 0.21244023807335696}
    }
}


def generate_job(num_step, net_id, type, batch_size, path, need_tosave, file_name=""):
    if net_id == 0:
        vgg16 = VGG16_test.VGG16(num_step=num_step, type=type, batch_size=batch_size, gpu_num=gpu, path=path, file_name=file_name, n_class=1000, need_tosave=need_tosave)
        return vgg16
    elif net_id == 1:
        inceptionv3 = InceptionV3_test.Inceptionv3(num_step=num_step, type=type, batch_size=batch_size, gpu_num=gpu, path=path, file_name=file_name, need_tosave=need_tosave)
        return inceptionv3
    elif net_id == 2:
        inceptionv4 = InceptionV4_test.Inceptionv4(num_step=num_step, type=type, batch_size=batch_size, gpu_num=gpu, path=path, file_name=file_name, need_tosave=need_tosave)
        return inceptionv4
    elif net_id == 3:
        resNet = ResNet50_test.ResNet50(num_step=num_step, type=type, batch_size=batch_size, gpu_num=gpu, path=path, file_name=file_name, need_tosave=need_tosave)
        return resNet
    elif net_id == 4:
        denseNet = DenseNet_test.DenseNet121(num_step=num_step, type=type, batch_size=batch_size, gpu_num=gpu, path=path, file_name=file_name, need_tosave=need_tosave)
        return denseNet


def create_extra_matrix(need_tosave, pipe1, pipe2):
    outspace = []
    arr_size = need_tosave * 1e6 / 4
    gctx = ndarray.gpu(0)
    while arr_size > 0:
        if arr_size > 10000 * 10000:
            outspace.append(ndarray.array(np.ones((10000, 10000)) * 0.01, ctx=gctx))
            arr_size -= 10000 * 10000
        else:
            need_sqrt = int(pow(arr_size, 0.5))
            if need_sqrt <= 0:
                break
            outspace.append(ndarray.array(np.ones((need_sqrt, need_sqrt)) * 0.01, ctx=gctx))
            arr_size -= need_sqrt * need_sqrt
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
            # if not ((net_id == 0 and i == 3) or (net_id == 3 and i == 3) or (net_id == 4 and (i == 2 or i == 3))):
            #     continue
            # if not (net_id == 4 and i == 3):
            #     continue
            # if i!=3:
            #     continue
            if i == 0:
                batch_size = 16
                net_name_ = net_name
            else:
                batch_size = 2
                net_name_ = net_name + f' x{i}'
            path = f'./log/{net_name_}/'
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)
            nets = []
            for _ in range(num_net):
                # net_id = random.randint(0, 4) #net_id随机选取网络种类 0:vgg16, 1:inceptionv3, 2:inceptionv4, 3:resNet, 4:denseNet
                nets.append(net_id)
            print("选取的网络", list(map(lambda x: net_names[x], nets)))
            vanilla_max_memory = 0
            need_tosave_list = []
            for t in range(repeat_times):
                print(f'repeat_times:{t}')
                for type in range(3):  # type是调度方式的选择, 0.不调度 1.capuchin 2.vdnn
                    # if type==1:
                    #     continue
                    need_tosave = 0
                    if type == 1:
                        bud = vanilla_max_memory * (1 - budget[net_name][num_net][batch_size])
                        # 总显存=预算+need_tosave(额外占用空间)
                        need_tosave = 11019 - bud
                        print(f'need_tosave:{need_tosave}')
                        need_tosave_list.append(need_tosave)
                        pipe1 = multiprocessing.Queue()
                        pipe2 = multiprocessing.Queue()
                        p = Process(target=create_extra_matrix, args=(need_tosave, pipe1, pipe2))
                        p.start()
                        while True:
                            if not pipe1.empty():
                                if pipe1.get():
                                    break
                    job_pool = []
                    for i, net_id in enumerate(nets):
                        job_pool.append(generate_job(num_step=50, net_id=net_id, type=type, batch_size=batch_size, path=path, file_name=f"_repeat_time={t}_net_order={i}", need_tosave=0))
                    for job in job_pool:
                        job.start()
                    for job in job_pool:
                        job.join()
                    if type == 1:
                        pipe2.put(True)
                        p.join()
                        pipe1.close()
                        pipe2.close()
                    if type == 0:
                        vanilla_max_memory = get_vanilla_max_memory(path, repeat_times=repeat_times)
            try:
                get_result(path, repeat_times=repeat_times, need_tosave=need_tosave_list, skip='capuchin')
            except Exception as e:
                traceback.print_exc()
            print("Experiment1 finish")


if __name__ == "__main__":
    Experiment1()
# get_result('./log/VGG x2/', repeat_times=3, need_tosave=[0,0,0])
