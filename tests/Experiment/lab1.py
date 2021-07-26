import multiprocessing
import os
import sys
import time
import traceback
from multiprocessing import Process

import numpy as np

sys.path.append('../../')
from pycode.tinyflow import ndarray
from tests.Experiment import VGG16_test, ResNet50_test, DenseNet_test, InceptionV3_test, InceptionV4_test
from tests.Experiment.result import get_result, get_vanilla_max_memory

gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
net_names = ['VGG', 'InceptionV3', 'InceptionV4', 'ResNet', 'DenseNet']
budget = {
    'VGG': {
        1: {2: 0.19225967540574282, 16: 0.35050344462109173},
        2: {2: 0.16605005389841376},
        3: {2: 0.1669419047740918}
    },
    'InceptionV3': {
        1: {2: 0.24301075268817204, 16: 0.5574442435201928},
        2: {2: 0.3043805833602411},
        3: {2: 0.24857190980685862}
    },
    'InceptionV4': {
        1: {2: 0.50480109739369, 16: 0.627173128649216},
        2: {2: 0.515406162464986},
        3: {2: 0.42023152790143836}
    },
    'ResNet': {
        1: {2: 0.24476077500988533, 16: 0.6220095693779905},
        2: {2: 0.32698945827648906},
        3: {2: 0.2767580454161535}
    },
    'DenseNet': {
        1: {2: 0.38183807439824946, 16: 0.746799739639835},
        2: {2: 0.2523519969688825},
        3: {2: 0.210212894494957}
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
    arr_size = need_tosave * pow(2,20) / 4
    gctx = ndarray.gpu(0)
    while arr_size > 0:
        if arr_size > 100000000:
            outspace.append(ndarray.array(np.ones((100000000, ), dtype=np.float32) * 0.01, ctx=gctx))
            arr_size -= 100000000
        else:
            # need_sqrt = int(pow(arr_size, 0.5))
            # if need_sqrt <= 0:
            #     break
            arr_size = int(arr_size)
            outspace.append(ndarray.array(np.ones((arr_size, ), dtype=np.float32) * 0.01, ctx=gctx))
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
            # if not ((net_id == 0 and i == 3) or (net_id == 3 and i == 3) or (net_id == 4 and (i == 2 or i == 3))):
            #     continue
            # if i==1 or i==2 or i==3:
            #     continue

            if i == 0:
                batch_size = 16
                net_name_ = net_name
            else:
                batch_size = 2
                net_name_ = net_name + f' x{i}'
            print("batch_size",batch_size)
            path = f'./log/{net_name_}/'
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)
            nets = []
            for _ in range(num_net):
                # net_id = random.randint(0, 4) #net_id随机选取网络种类 0:vgg16, 1:inceptionv3, 2:inceptionv4, 3:resNet, 4:denseNet
                nets.append(net_id)
            print("选取的网络", list(map(lambda x: net_names[x], nets)))
            # if net_id==0 or net_id==1 or net_id==2:
            #     continue
            if net_id <=1:
                break
            vanilla_max_memory = 0
            need_tosave_list = []
            for t in range(repeat_times):
                print(f'repeat_times:{t}')
                for type in range(3):  # type是调度方式的选择, 0.不调度 1.capuchin 2.vdnn
                    # if type !=2:
                    #     continue
                    if type == 1:
                        bud = vanilla_max_memory * (1 - budget[net_name][num_net][batch_size])
                        # 总显存=预算+need_tosave+cuda开销(额外占用空间)
                        # need_tosave = 11019 - bud - 536
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
            # try:
            get_result(path, repeat_times=repeat_times, need_tosave=need_tosave_list)
            # except Exception as e:
            #     traceback.print_exc()
            print("Experiment1 finish")


if __name__ == "__main__":
    Experiment1()
    # from pycode.tinyflow import gpu_op
    # cudaStream = gpu_op.create_cudaStream()
    # cudnnHandle = gpu_op.create_cudnnHandle(cudaStream)
    # cublasHandle = gpu_op.create_cublasHandle(cudaStream)
    # time.sleep(1000)
    # get_result('./log/InceptionV3 x1/', repeat_times=3, need_tosave=[7480,9060,9848])
