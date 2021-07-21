import os

os.environ['CUDA_VISIBLE_DEVICES'] = f'{0}'
import pickle as pkl
import multiprocessing
import time
import numpy as np
from VGG16_test_leo import VGG16
from Inceptionv3_test_leo import Inceptionv3
from Inceptionv4_test_leo import Inceptionv4
from ResNet50_test_leo import ResNet50
from DenseNet_test_leo import DenseNet121
from multiprocessing import Process
from pycode.tinyflow import Scheduler as mp
from pycode.tinyflow import ndarray
from pycode.tinyflow.get_result import get_result
from util import GPURecord


def run_model(log_path, model: list, top_control_queue_list, top_message_queue_list, num_step, batch_size, **kwargs):
    job_number = len(model)
    global_message_queue = multiprocessing.Queue()
    global_control_queue = multiprocessing.Queue()
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    inited_model = []
    for job_id, m in enumerate(model):
        top_control_queue = multiprocessing.Queue()
        top_control_queue_list.append(top_control_queue)
        top_message_queue = multiprocessing.Queue()
        top_message_queue_list.append(top_message_queue)
        inited_model.append(m(num_step=num_step, batch_size=batch_size, log_path=log_path, job_id=job_id))
    job_pool = [Process(target=m.run,
                        args=(ndarray.gpu(0), top_control_queue_list[i], top_message_queue_list[i], 1000, np.random.normal(loc=0, scale=0.1, size=(m.batch_size, m.image_channel, m.image_size, m.image_size)),
                              np.random.normal(loc=0, scale=0.1, size=(m.batch_size, 1000))), kwargs=dict(predict_results=kwargs['predict_results'][m.__class__.__name__])) for i, m in enumerate(inited_model)]
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


if __name__ == '__main__':
    with open(f'../../res/inferred_shape.pkl', 'rb') as f:
        predict_results = pkl.load(f)
    for t in range(10):
        log_path = f'./log/MDW/repeat_{t}/vanilla'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        # =======================vanilla===================== #
        print(f'vanilla,repeat:{t}')
        recorder = GPURecord(log_path)
        f1 = open(f"{log_path}/gpu_time.txt", "w+")
        top_control_queue_list = []
        top_message_queue_list = []
        model_list = [VGG16, Inceptionv3, Inceptionv4, ResNet50, DenseNet121]
        np.random.shuffle(model_list)
        recorder.start()
        start_time = time.time()
        run_model(log_path, model_list, top_control_queue_list, top_message_queue_list, 50, 2, predict_results=predict_results)
        f1.write(f'time_cost:{time.time() - start_time}')
        f1.flush()
        f1.close()
        recorder.stop()
        # =======================schedule===================== #
        print(f'schedule,repeat:{t}')
        log_path = f'./log/MDW/repeat_{t}/schedule'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        recorder = GPURecord(log_path)
        f1 = open(f"{log_path}/gpu_time.txt", "w+")
        top_control_queue_list = []
        top_message_queue_list = []
        model_list = [VGG16, Inceptionv3, Inceptionv4, ResNet50, DenseNet121]
        np.random.shuffle(model_list)
        res = []
        recorder.start()
        start_time = time.time()
        run_model(log_path, model_list, top_control_queue_list, top_message_queue_list, 50, 2, predict_results=predict_results)
        f1.write(f'time_cost:{time.time() - start_time}')
        f1.flush()
        f1.close()
        recorder.stop()
    get_result(raw_workload='./log/MDW/', repeat_times=10)
