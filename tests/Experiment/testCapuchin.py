import multiprocessing
import os
import sys
import time
import traceback
from multiprocessing import Process
import os
import numpy as np

sys.path.append('../../')

from tests.Experiment import VGG16_test, ResNet50_test, DenseNet_test, InceptionV3_test, InceptionV4_test
from tests.Experiment.result import get_result, get_vanilla_max_memory
from tests.Experiment.lab1 import generate_job
gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
net_names = ['VGG', 'InceptionV3', 'InceptionV4', 'ResNet', 'DenseNet']
repeat_times = 1
ratio = {
    'VGG': 0.4654201822477219,
    'InceptionV3': 0.6161714930424973,
    'InceptionV4': 0.7081164945254665,
    'ResNet': 0.600043020004302,
    'DenseNet': 0.7222270636649722
}





def Experiment1():
    for net_id in range(5):
        print("Experiment1 start")
        net_name = net_names[net_id]
        for i, num_net in enumerate([1]):
            batch_size = 16
            net_name_ = net_name + f' x{i + 1}'
            print("batch_size", batch_size)

            nets = []
            for _ in range(num_net):
                # net_id = random.randint(0, 4) #net_id随机选取网络种类 0:vgg16, 1:inceptionv3, 2:inceptionv4, 3:resNet, 4:denseNet
                nets.append(net_id)
            print("选取的网络", list(map(lambda x: net_names[x], nets)))
            vanilla_capu_max_memory = 0
            for repeat in range(repeat_times):
                p = f'./log/{net_name} x1/type0_repeat_time={repeat}_net_order=0_record_2.txt'
                with open(p, 'r') as f:
                    lines = f.readlines()
                vanilla_max_memory = 0
                for line in lines:
                    try:
                        memory = float(line.split('\t')[1].split(' ')[1])
                        if memory > vanilla_max_memory:
                            vanilla_max_memory = memory
                    except Exception as e:
                        pass
                vanilla_capu_max_memory += vanilla_max_memory
            vanilla_capu_max_memory /= repeat_times
            for ratio in range(0, 10):
                ratio_ = ratio / 10
                path = f'./log/{net_name_} ratio={ratio_}/'
                print(path)
                if not os.path.exists(path):
                    os.makedirs(path)
                for type in [0,1]:  # type是调度方式的选择, 0.capuchin_不调度 1.capuchin 2.vdnn 3. vdnn_不调度
                    print(f'type:{type}')
                    if type != 1:
                        bud = -1
                    else:
                        bud = (1 - ratio_) * vanilla_capu_max_memory
                    print(f'budget:{bud}')
                    for t in range(repeat_times):
                        print(f'repeat_times:{t}')
                        job_pool = []
                        for i, net_id in enumerate(nets):
                            job_pool.append(
                                generate_job(num_step=50, net_id=net_id, type=type, batch_size=batch_size, path=path,
                                             file_name=f"_repeat_time={t}_net_order={i}", budget=bud))
                        for job in job_pool:
                            job.start()
                        for job in job_pool:
                            job.join()
                get_result(path, repeat_times=repeat_times, skip='vdnn')
            print("Experiment1 finish")


if __name__ == "__main__":
    Experiment1()