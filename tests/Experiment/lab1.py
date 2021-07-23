import os
import sys

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


def Experiment1():
    net_names = ['VGG', 'InceptionV3', 'InceptionV4', 'ResNet', 'DenseNet']
    budget = {
        'VGG': {
            1: {2: 0.2174494900279348, 16: 0.35648782684120267},
            2: {2: 0.21485818067033183},
            3: {2: 0.19819229733851573}
        },
        'InceptionV3': {
            1: {2: 0.31681040272964184, 16: 0.6304005544632028},
            2: {2: 0.2752777245166674},
            3: {2: 0.27398579361625057}
        },
        'InceptionV4': {
            1: {2: 0.5019070066423826, 16: 0.6893718558970763},
            2: {2: 0.46673445779697914},
            3: {2: 0.4388073557438008}
        },
        'ResNet': {
            1: {2: 0.26281338767444734, 16: 0.7066612562202336},
            2: {2: 0.342654921731748},
            3: {2: 0.28701103881578344}
        },
        'DenseNet': {
            1: {2: 0.39732987036657325, 16: 0.7603964692004176},
            2: {2: 0.24516440969028397},
            3: {2: 0.20844425426073546}
        }
    }
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
                    if type!=2:
                        continue
                    need_tosave = 0
                    if type == 1:
                        bud = vanilla_max_memory * (1 - budget[net_name][num_net][batch_size])
                        # 总显存=预算+need_tosave(额外占用空间)
                        need_tosave = 11019 - bud
                        print(f'need_tosave:{need_tosave}')
                        need_tosave_list.append(need_tosave)
                    job_pool = [
                        generate_job(num_step=50, net_id=net_id, type=type, batch_size=batch_size, path=path, file_name=f"_repeat_time={t}_net_order={i}", need_tosave=need_tosave) for
                        i, net_id in enumerate(nets)]
                    for job in job_pool:
                        job.start()
                    for job in job_pool:
                        job.join()
                    if type == 0:
                        vanilla_max_memory = get_vanilla_max_memory(path, repeat_times=repeat_times)
            get_result(path, repeat_times=repeat_times, need_tosave=need_tosave_list)
            print("Experiment1 finish")
Experiment1()
# get_result('./log/VGG x2/', repeat_times=3, need_tosave=[0,0,0])
