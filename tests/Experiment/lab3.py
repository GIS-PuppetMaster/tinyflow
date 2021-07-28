import datetime
import os
import sys
import threading

import numpy as np
import pynvml

sys.path.append('../../')
from tests.Experiment import VGG16_test, ResNet50_test, DenseNet_test, InceptionV3_test, InceptionV4_test
import random, time
import pickle as pkl
from lab1 import budget, net_names

gpu = 0
methods = ['vanilla', 'capuchin', 'vdnn']


def get_result(log_path, repeat_times, log, need_tosave_list=None):
    res = open(f'{log_path}/res.txt', 'w+')
    all_vanilla_max_memory = []
    all_vanilla_time = []

    all_vdnn_max_memory = []
    all_vdnn_time = []
    all_vdnn_MSR = []
    all_vdnn_EOR = []
    all_vdnn_BCR = []

    all_capuchin_max_memory = []
    all_capuchin_time = []
    all_capuchin_MSR = []
    all_capuchin_EOR = []
    all_capuchin_BCR = []
    for t in range(repeat_times):
        vanilla_max_gpu_memory, vanilla_average_time_cost = log[methods[0]][t]
        all_vanilla_max_memory.append(vanilla_max_gpu_memory)
        all_vanilla_time.append(vanilla_average_time_cost)
        capuchin_max_gpu_memory, capuchin_average_time_cost = log[methods[1]][t]
        if need_tosave_list is not None and len(need_tosave_list) > 0:
            capuchin_max_gpu_memory -= need_tosave_list[t]
        all_capuchin_max_memory.append(capuchin_max_gpu_memory)
        all_capuchin_time.append(capuchin_average_time_cost)
        all_capuchin_MSR.append(1 - capuchin_max_gpu_memory / vanilla_max_gpu_memory)
        all_capuchin_EOR.append(capuchin_average_time_cost / vanilla_average_time_cost - 1)
        all_capuchin_BCR.append(all_capuchin_MSR[-1] / all_capuchin_EOR[-1])
        vdnn_max_gpu_memory, vdnn_average_time_cost = log[methods[2]][t]
        all_vdnn_max_memory.append(vdnn_max_gpu_memory)
        all_vdnn_time.append(vdnn_average_time_cost)
        all_vdnn_MSR.append(1 - vdnn_max_gpu_memory / vanilla_max_gpu_memory)
        all_vdnn_EOR.append(vdnn_average_time_cost / vanilla_average_time_cost - 1)
        all_vdnn_BCR.append(all_vdnn_MSR[-1] / all_vdnn_EOR[-1])
    all_vanilla_max_memory = np.array(all_vanilla_max_memory)
    all_vanilla_time = np.array(all_vanilla_time)
    all_vdnn_max_memory = np.array(all_vdnn_max_memory)
    all_vdnn_time = np.array(all_vdnn_time)
    all_vdnn_MSR = np.array(all_vdnn_MSR)
    all_vdnn_EOR = np.array(all_vdnn_EOR)
    all_vdnn_BCR = np.array(all_vdnn_BCR)
    all_capuchin_max_memory = np.array(all_capuchin_max_memory)
    all_capuchin_time = np.array(all_capuchin_time)
    all_capuchin_MSR = np.array(all_capuchin_MSR)
    all_capuchin_EOR = np.array(all_capuchin_EOR)
    all_capuchin_BCR = np.array(all_capuchin_BCR)
    res.writelines('vanilla:\n')
    res.writelines(f'max_memory:{all_vanilla_max_memory.mean()} +- {all_vanilla_max_memory.std()}\n')
    res.writelines(f'time:{all_vanilla_time.mean()} +- {all_vanilla_time.std()}\n\n')

    res.writelines('vDNN:\n')
    res.writelines(f'max_memory:{all_vdnn_max_memory.mean()} +- {all_vdnn_max_memory.std()}\n')
    res.writelines(f'time:{all_vdnn_time.mean()} +- {all_vdnn_time.std()}\n')
    res.writelines(f'memory_saved:{all_vdnn_MSR.mean()} +- {all_vdnn_MSR.std()}\n')
    res.writelines(f'extra_overhead:{all_vdnn_EOR.mean()} +- {all_vdnn_EOR.std()}\n')
    res.writelines(f'efficiency:{all_vdnn_MSR.mean() / all_vdnn_EOR.mean()}\n\n')

    res.writelines('capuchin:\n')
    res.writelines(f'max_memory:{all_capuchin_max_memory.mean()} +- {all_capuchin_max_memory.std()}\n')
    res.writelines(f'time:{all_capuchin_time.mean()} +- {all_capuchin_time.std()}\n')
    res.writelines(f'memory_saved:{all_capuchin_MSR.mean()} +- {all_capuchin_MSR.std()}\n')
    res.writelines(f'extra_overhead:{all_capuchin_EOR.mean()} +- {all_capuchin_EOR.std()}\n')
    res.writelines(f'efficiency:{all_capuchin_MSR.mean() / all_capuchin_EOR.mean()}\n\n')
    res.flush()
    res.close()


class GPURecord(threading.Thread):
    def __init__(self, log_path, suffix=""):
        threading.Thread.__init__(self)
        pynvml.nvmlInit()
        GPU = int(os.environ['CUDA_VISIBLE_DEVICES'])
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(GPU)
        self.f = open(f"{log_path}/gpu_record{suffix}.txt", "w+")
        # todo 临时用作释放的计数器
        self.times = 0
        self.max_gpu_memory = 0
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.base_used = meminfo.used / 1024 ** 2
        self.flag = True

    def run(self):
        while self.flag:
            # if self.times == 30000:
            #     self.f.close()
            #     break
            self.times += 1
            # time.sleep(0.1)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.memory_used = meminfo.used / 1024 ** 2
            if self.memory_used > self.max_gpu_memory:
                self.max_gpu_memory = self.memory_used
                print("time", datetime.datetime.now(),
                      "\tmemory", self.memory_used,
                      "\tmax_memory_used", self.max_gpu_memory,
                      "\tretained_memory_used", self.memory_used - self.base_used,
                      "\tretained_max_memory_used", self.max_gpu_memory - self.base_used, file=self.f)  # 已用显存大小
                self.f.flush()

    def stop(self):
        self.flag = False
        time.sleep(0.01)
        self.f.close()


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

budget_ratio = 0.20144502280823326


def Experiment3():
    repeat_times = 3
    num_step = 50
    batch_size = 2
    file = open('./log/experiment3_log.txt', 'w+')
    log = {'vanilla': [], 'capuchin': [], 'vdnn': []}
    log_path = f'./log/Experiment3/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    f1 = open(f"{log_path}/log.pkl", "wb")
    # res = open(f'{log_path}/res.txt', 'w+')
    need_tosave_list = []
    vanilla_max_memory = 0
    for t in range(repeat_times):
        print(f'repeat_times:{t}')
        nets = [0, 1, 2, 3, 4]
        np.random.shuffle(nets)
        file.writelines(f'repeat_times:{t}, nets:{nets}')
        file.flush()
        print("Experiment3 start")
        print("选取的网络", nets)
        path = f'./log/Experiment3/repeat_{t}'
        print(path)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # TENSILE中MDW实验的MSR值
        vanilla_max_memory = 0
        for type in range(3):  # type是调度方式的选择, 0.不调度，1.capuchin 2.vdnn
            job_pool = []
            for i, net_id in enumerate(nets):
                if type == 1:
                    # budget_ratio = 1-schedule/vanilla
                    # schedule = (1-budget_ratio) * vanilla
                    budget = vanilla_max_memory*(1-budget_ratio)
                else:
                    budget = 0
                job_pool.append(
                    generate_job(num_step=num_step, net_id=net_id, type=type, batch_size=batch_size, path=path,
                                 file_name=f"_repeat_time={t}_net_order={i}", budget=budget))
            start_time = time.time()
            recorder = GPURecord(log_path)
            recorder.start()
            for job in job_pool:
                job.start()
            for job in job_pool:
                job.join()
            average_time_cost = (time.time() - start_time) / num_step
            recorder.stop()
            log[methods[type]].append([recorder.max_gpu_memory, average_time_cost])
            with open(f"{log_path}/log.pkl", "wb") as f:
                pkl.dump(log, f1)
            if type == 0:
                vanilla_max_memory = recorder.max_gpu_memory
    get_result(log_path, repeat_times, log, need_tosave_list)
    print("Experiment3 finish")
    file.close()


if __name__ == '__main__':
    Experiment3()
    # log_path = f'./log/Experiment3/'
    # repeat_times =3
    # with open(f"{log_path}/log.pkl", "rb") as f:
    #     log = pkl.load(f)
    # get_result(log_path, repeat_times, log)
