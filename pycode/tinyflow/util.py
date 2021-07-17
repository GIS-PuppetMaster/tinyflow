import multiprocessing
import threading
import datetime
import time

import os
from multiprocessing import Process

import numpy as np
import pynvml
from pycode.tinyflow import Scheduler as mp

from pycode.tinyflow import ndarray


class GPURecord(threading.Thread):
    def __init__(self, log_path):
        threading.Thread.__init__(self)
        pynvml.nvmlInit()
        GPU = int(os.environ['CUDA_VISIBLE_DEVICES'])
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(GPU)
        self.f = open(f"{log_path}/gpu_record.txt", "w+")
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


def run_workload(batch_size, num_step, log_path, top_control_queue_list, top_message_queue_list, job_id, executor_ctx, model, **kwargs):
    top_control_queue = multiprocessing.Queue()
    top_control_queue_list.append(top_control_queue)
    top_message_queue = multiprocessing.Queue()
    top_message_queue_list.append(top_message_queue)

    model = model(num_step=num_step, batch_size=batch_size, log_path=log_path, job_id=job_id)
    X_val = np.random.normal(loc=0, scale=0.1, size=(
        batch_size, model.image_channel, model.image_size, model.image_size))  # number = batch_size  channel = 3  image_size = 224*224

    y_val = np.random.normal(loc=0, scale=0.1, size=(batch_size, 1000))  # n_class = 1000

    p = Process(target=model.run,
                args=(executor_ctx, top_control_queue, top_message_queue, 1000, X_val, y_val), kwargs=kwargs)
    return p


def main(raw_log_path, repeat_times, job_number, batch_size, model, **kwargs):
    for t in range(repeat_times):
        print(f'repeat_time:{t}')
        global_message_queue = multiprocessing.Queue()
        global_control_queue = multiprocessing.Queue()

        top_control_queue_list = []
        top_message_queue_list = []
        executor_ctx = ndarray.gpu(0)
        path_list = list(os.path.split(raw_log_path))
        path_list.insert(1, f'repeat_{t}')
        log_path = os.path.join(*path_list)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        num_step = 50
        job_pool = [run_workload(batch_size, num_step, log_path, top_control_queue_list, top_message_queue_list, job_id, executor_ctx, model, **kwargs) for job_id in range(job_number)]
        for job in job_pool:
            job.start()

        if 'schedule' in log_path:
            scheduler = Process(target=mp.multiprocess_init, args=(global_message_queue, global_control_queue, job_number))
            scheduler.start()
            while True in [job.is_alive() for job in job_pool]:
                for i in range(job_number):
                    if not top_message_queue_list[i].empty():
                        # print("job ", i, "message")
                        global_message_queue.put([i, top_message_queue_list[i].get()])
                if not global_control_queue.empty():
                    global_control = global_control_queue.get()
                    for i in range(job_number):
                        if i in global_control:
                            print("job ", i, "control")
                            top_control_queue_list[i].put(global_control[i])
            for q in top_message_queue_list:
                q.close()
            for q in top_control_queue_list:
                q.close()
            scheduler.terminate()
        else:
            while True in [job.is_alive() for job in job_pool]:
                for i in range(job_number):
                    if not top_message_queue_list[i].empty():
                        top_message_queue_list[i].get()
                    if not top_control_queue_list[i].empty():
                        top_control_queue_list[i].get()
        for job in job_pool:
            job.terminate()
        while not global_control_queue.empty():
            global_control_queue.get()
        global_control_queue.close()
        while not global_message_queue.empty():
            global_message_queue.get()
        global_message_queue.close()
