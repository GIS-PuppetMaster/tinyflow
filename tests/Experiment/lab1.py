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
repeat_times = 5

def get_budget():
    ratio = {
        'VGG': 0.2798349455455591,
        'InceptionV3': 0.49659956269320665,
        'InceptionV4': 0.5757002707861677,
        'ResNet': 0.592396694214876,
        'DenseNet': 0.705503563981962
    }
    budget = {}
    for net_name_ in net_names:
        vanilla_capu_max_memory = 0
        for repeat in range(repeat_times):
            path = f'./log/{net_name_} x1/type0_repeat_time={repeat}_net_order=0_record_2.txt'
            with open(path, 'r') as f:
                lines = f.readlines()
            vanilla_max_memory = 0
            for line in lines:
                memory = float(line.split('\t')[1].split(' ')[1])
                if memory > vanilla_max_memory:
                    vanilla_max_memory = memory
            vanilla_capu_max_memory += vanilla_max_memory
        vanilla_capu_max_memory/=repeat_times
        # MSR = (vanilla-schedule)/vanilla
        # schedule = vanilla - vanilla * MSR
        budget[net_name_] = (1-ratio[net_name_]) * vanilla_capu_max_memory
    return budget



# budget = {
#     'VGG': {2: 2642.0,
#             16: 5527.62687581475},
#     'InceptionV3': {2: 1173.3333333333333,
#                     16: 2447.3333333333335},
#     'InceptionV4': {
#         2: 1444.0,
#         16: 3788.6666666666665
#     },
#     'ResNet': {
#         2: 1273.3333333333333, 16: 2212.0},
#     'DenseNet': {
#         2: 1130.0, 16: 2334.0
#     }
# }
budget = get_budget()

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
        print("Experiment1 start")
        net_name = net_names[net_id]
        for i, num_net in enumerate([1, 2, 3]):
            # if i == 0:
            #     batch_size = 16
            #     net_name_ = net_name
            # else:
            batch_size = 16
            net_name_ = net_name + f' x{i + 1}'
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
                for type in [1,2]:  # type是调度方式的选择, 0.capuchin_不调度 1.capuchin 2.vdnn 3. vdnn_不调度
                    bud = -1
                    if type == 1:
                        bud = budget[net_name]
                        print(f'budget:{bud}')
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
