import datetime
from multiprocessing import Process

import pynvml

from pycode.tinyflow import ndarray, gpu_op
from pycode.tinyflow import autodiff as ad
from pycode.tinyflow import mainV2 as mp
import six.moves.cPickle as pickle
import gzip
import numpy as np
import os
import queue
import multiprocessing
import threading
import time

from tools import *

def load_mnist_data(dataset):
    # 加载mnist数据集
    """ Load the dataset
    Code adapted from http://deeplearning.net/tutorial/code/logistic_sgd.py

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    """
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('Loading data...')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix), np.float32
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector), np.int64 that has the same length
    # as the number of rows in the input. It should give the target
    # to the example with the same index in the input.
    return train_set, valid_set, test_set

def convert_to_one_hot(vals):
    """Helper method to convert label array to one-hot array."""
    one_hot_vals = np.zeros((vals.size, vals.max()+1))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals


def sgd_update_gpu(param, grad_param, learning_rate, cuda_stream):
    """Helper GPU SGD update method. Avoids copying NDArray to cpu."""
    assert isinstance(param, ndarray.NDArray)
    assert isinstance(grad_param, ndarray.NDArray)
    gpu_op.matrix_elementwise_multiply_by_const(
        grad_param, -learning_rate, grad_param, cuda_stream)
    gpu_op.matrix_elementwise_add(param, grad_param, param, cuda_stream)


def mnist_mlp(executor_ctx, num_epochs, print_loss_val_each_epoch, top_control_queue, top_message_queue):
    # 训练一个三层感知机模型
    print("Build 3-layer MLP model...")

    cuda_stream = gpu_op.create_cudaStream()
    gpu_record = GPURecord()


    W1 = ad.Variable(name="W1")
    W2 = ad.Variable(name="W2")
    W3 = ad.Variable(name="W3")
    W4 = ad.Variable(name="W4")
    W5 = ad.Variable(name="W5")
    W6 = ad.Variable(name="W6")
    b1 = ad.Variable(name="b1")
    b2 = ad.Variable(name="b2")
    b3 = ad.Variable(name="b3")
    b4 = ad.Variable(name="b4")
    b5 = ad.Variable(name="b5")
    b6 = ad.Variable(name="b6")
    X = ad.Placeholder(name="X")
    y_ = ad.Placeholder(name="y_")

    # 下面是三层网络的激活函数，两个relu和一个softmax

    # relu(X W1+b1)
    z2 = ad.dense(X, W1, b1)
    z3 = ad.fullyactivation_forward_op(z2, "NCHW", "relu")

    # relu(z3 W2+b2)
    z5 = ad.dense(z3, W2, b2)
    z6 = ad.fullyactivation_forward_op(z5, "NCHW", "relu")
    z7 = ad.dense(z6, W3, b3)
    z8 = ad.fullyactivation_forward_op(z7, "NCHW", "relu")
    z9 = ad.dense(z8, W4, b4)
    z10 = ad.fullyactivation_forward_op(z9, "NCHW", "relu")
    z11 = ad.dense(z10, W5, b5)
    z12 = ad.fullyactivation_forward_op(z11, "NCHW", "relu")
    # softmax(z5 W2+b2)
    z13 = ad.dense(z12, W6, b6)
    bn1 = ad.fullybn_forward_op(z13, "NCHW")
    y = ad.fullyactivation_forward_op(bn1, "NCHW", "softmax")
    loss = ad.crossEntropy_loss(y, y_)


    # 此处向前为符号定义
    # 只声明，不操作
    executor = ad.Executor(loss,y,0.001, top_control_queue=top_control_queue, top_message_queue=top_message_queue)

    # Read input data
    datasets = load_mnist_data("mnist.pkl.gz")
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]


    # Set up minibatch
    batch_size = 1000
    n_train_batches = train_set_x.shape[0] // batch_size
    n_valid_batches = valid_set_x.shape[0] // batch_size

    print("Start training loop...")

    # Initialize parameters
    # 随机初始化网络中的w和b
    rand = np.random.RandomState(seed=123)
    W1_val = rand.normal(scale=0.1, size=(784, 4096))
    W2_val = rand.normal(scale=0.1, size=(4096, 1024))
    W3_val = rand.normal(scale=0.1, size=(1024, 1024))
    W4_val = rand.normal(scale=0.1, size=(1024, 1024))
    W5_val = rand.normal(scale=0.1, size=(1024, 1024))
    W6_val = rand.normal(scale=0.1, size=(1024, 10))

    b1_val = rand.normal(scale=0.1, size=(4096))
    b2_val = rand.normal(scale=0.1, size=(1024))
    b3_val = rand.normal(scale=0.1, size=(1024))
    b4_val = rand.normal(scale=0.1, size=(1024))
    b5_val = rand.normal(scale=0.1, size=(1024))
    b6_val = rand.normal(scale=0.1, size=(10))


    W1_val_m = np.zeros(shape=(784, 4096), dtype=np.float32)
    W2_val_m = np.zeros(shape=(4096, 1024), dtype=np.float32)
    W3_val_m = np.zeros(shape=(1024, 1024), dtype=np.float32)
    W4_val_m = np.zeros(shape=(1024, 1024), dtype=np.float32)
    W5_val_m = np.zeros(shape=(1024, 1024), dtype=np.float32)
    W6_val_m = np.zeros(shape=(1024, 10), dtype=np.float32)
    b1_val_m = np.zeros(shape=(4096), dtype=np.float32)
    b2_val_m = np.zeros(shape=(1024), dtype=np.float32)
    b3_val_m = np.zeros(shape=(1024), dtype=np.float32)
    b4_val_m = np.zeros(shape=(1024), dtype=np.float32)
    b5_val_m = np.zeros(shape=(1024), dtype=np.float32)
    b6_val_m = np.zeros(shape=(10), dtype=np.float32)
    W1_val_v = np.zeros(shape=(784, 4096), dtype=np.float32)
    W2_val_v = np.zeros(shape=(4096, 1024), dtype=np.float32)
    W3_val_v = np.zeros(shape=(1024, 1024), dtype=np.float32)
    W4_val_v = np.zeros(shape=(1024, 1024), dtype=np.float32)
    W5_val_v = np.zeros(shape=(1024, 1024), dtype=np.float32)
    W6_val_v = np.zeros(shape=(1024, 10), dtype=np.float32)
    b1_val_v = np.zeros(shape=(4096), dtype=np.float32)
    b2_val_v = np.zeros(shape=(1024), dtype=np.float32)
    b3_val_v = np.zeros(shape=(1024), dtype=np.float32)
    b4_val_v = np.zeros(shape=(1024), dtype=np.float32)
    b5_val_v = np.zeros(shape=(1024), dtype=np.float32)
    b6_val_v = np.zeros(shape=(10), dtype=np.float32)
    X_val = np.zeros(shape=(batch_size, 784), dtype=np.float32)
    y_val = np.zeros(shape=(batch_size, 10), dtype=np.float32)
    valid_X_val = np.zeros(shape=(batch_size, 784), dtype=np.float32)
    valid_y_val = np.zeros(shape=(batch_size, 10), dtype=np.float32)


    # todo 此处修改回gpu
    W1_val = ndarray.array(W1_val, ctx=executor_ctx)
    W2_val = ndarray.array(W2_val, ctx=executor_ctx)
    W3_val = ndarray.array(W3_val, ctx=executor_ctx)
    W4_val = ndarray.array(W4_val, ctx=executor_ctx)
    W5_val = ndarray.array(W5_val, ctx=executor_ctx)
    W6_val = ndarray.array(W6_val, ctx=executor_ctx)
    b1_val = ndarray.array(b1_val, ctx=executor_ctx)
    b2_val = ndarray.array(b2_val, ctx=executor_ctx)
    b3_val = ndarray.array(b3_val, ctx=executor_ctx)
    b4_val = ndarray.array(b4_val, ctx=executor_ctx)
    b5_val = ndarray.array(b5_val, ctx=executor_ctx)
    b6_val = ndarray.array(b6_val, ctx=executor_ctx)
    W1_val_m = ndarray.array(W1_val_m, ctx=executor_ctx)
    W2_val_m = ndarray.array(W2_val_m, ctx=executor_ctx)
    W3_val_m = ndarray.array(W3_val_m, ctx=executor_ctx)
    W4_val_m = ndarray.array(W4_val_m, ctx=executor_ctx)
    W5_val_m = ndarray.array(W5_val_m, ctx=executor_ctx)
    W6_val_m = ndarray.array(W6_val_m, ctx=executor_ctx)
    b1_val_m = ndarray.array(b1_val_m, ctx=executor_ctx)
    b2_val_m = ndarray.array(b2_val_m, ctx=executor_ctx)
    b3_val_m = ndarray.array(b3_val_m, ctx=executor_ctx)
    b4_val_m = ndarray.array(b4_val_m, ctx=executor_ctx)
    b5_val_m = ndarray.array(b5_val_m, ctx=executor_ctx)
    b6_val_m = ndarray.array(b6_val_m, ctx=executor_ctx)
    W1_val_v = ndarray.array(W1_val_v, ctx=executor_ctx)
    W2_val_v = ndarray.array(W2_val_v, ctx=executor_ctx)
    W3_val_v = ndarray.array(W3_val_v, ctx=executor_ctx)
    W4_val_v = ndarray.array(W4_val_v, ctx=executor_ctx)
    W5_val_v = ndarray.array(W5_val_v, ctx=executor_ctx)
    W6_val_v = ndarray.array(W6_val_v, ctx=executor_ctx)
    b1_val_v = ndarray.array(b1_val_v, ctx=executor_ctx)
    b2_val_v = ndarray.array(b2_val_v, ctx=executor_ctx)
    b3_val_v = ndarray.array(b3_val_v, ctx=executor_ctx)
    b4_val_v = ndarray.array(b4_val_v, ctx=executor_ctx)
    b5_val_v = ndarray.array(b5_val_v, ctx=executor_ctx)
    b6_val_v = ndarray.array(b6_val_v, ctx=executor_ctx)
    X_val = ndarray.array(X_val, ctx=executor_ctx)
    y_val = ndarray.array(y_val, ctx=executor_ctx)

    # 此处以上将数据分别转化为cpu和gpu两种格式

    feed_dict = {
        W1: W1_val,
        W2: W2_val,
        W3: W3_val,
        W4: W4_val,
        W5: W5_val,
        W6: W6_val,
        b1: b1_val,
        b2: b2_val,
        b3: b3_val,
        b4: b4_val,
        b5: b5_val,
        b6: b6_val,
        executor.Variable_node_to_mv[W1][0]: W1_val_m,
        executor.Variable_node_to_mv[W2][0]: W2_val_m,
        executor.Variable_node_to_mv[W3][0]: W3_val_m,
        executor.Variable_node_to_mv[W4][0]: W4_val_m,
        executor.Variable_node_to_mv[W5][0]: W5_val_m,
        executor.Variable_node_to_mv[W6][0]: W6_val_m,
        executor.Variable_node_to_mv[b1][0]: b1_val_m,
        executor.Variable_node_to_mv[b2][0]: b2_val_m,
        executor.Variable_node_to_mv[b3][0]: b3_val_m,
        executor.Variable_node_to_mv[b4][0]: b4_val_m,
        executor.Variable_node_to_mv[b5][0]: b5_val_m,
        executor.Variable_node_to_mv[b6][0]: b6_val_m,
        executor.Variable_node_to_mv[W1][1]: W1_val_v,
        executor.Variable_node_to_mv[W2][1]: W2_val_v,
        executor.Variable_node_to_mv[W3][1]: W3_val_v,
        executor.Variable_node_to_mv[W4][1]: W4_val_v,
        executor.Variable_node_to_mv[W5][1]: W5_val_v,
        executor.Variable_node_to_mv[W6][1]: W6_val_v,
        executor.Variable_node_to_mv[b1][1]: b1_val_v,
        executor.Variable_node_to_mv[b2][1]: b2_val_v,
        executor.Variable_node_to_mv[b3][1]: b3_val_v,
        executor.Variable_node_to_mv[b4][1]: b4_val_v,
        executor.Variable_node_to_mv[b5][1]: b5_val_v,
        executor.Variable_node_to_mv[b6][1]: b6_val_v}




    lr = 1.0e-3
    f1 = open("log/VGG x2 new/scheduled/gpu_time.txt", "w+")
    for i in range(num_epochs):
        print("epoch %d" % i)
        if i==1:
            gpu_record.start()
        if i==11:
            gpu_record.stop()
        for minibatch_index in range(n_train_batches):
            minibatch_start = minibatch_index * batch_size
            minibatch_end = (minibatch_index + 1) * batch_size

            # 修改引入数据的方式为手动引入
            x_numpy_val = train_set_x[minibatch_start:minibatch_end]
            y_numpy_val = convert_to_one_hot(
                train_set_y[minibatch_start:minibatch_end])

            # X_val[:] = train_set_x[minibatch_start:minibatch_end]
            # y_val[:] = convert_to_one_hot(
            #     train_set_y[minibatch_start:minibatch_end])

            feed_dict[X] = ndarray.array(x_numpy_val, ctx=executor_ctx)
            feed_dict[y_] = ndarray.array(y_numpy_val, ctx=executor_ctx)


            # 计算单步的梯度
            time1 = datetime.datetime.now()

            res = executor.run(feed_dict=feed_dict)

            time2 = datetime.datetime.now()
            print(str((time2 - time1).microseconds), file=f1)
            # print(loss_val.asnumpy())
            loss_val = res[0]
            feed_dict = res[1]
            # print(loss_val.asnumpy())
            # return



            # todo 更新sgd_update_gpu_on_cpu
            # def sgd_update_cpu(w1, w2, w3):
            #     w1_gpu = ndarray.empty(w1.shape, executor_ctx)
            #     w1.copyto(w1_gpu)
            #     w2_gpu = ndarray.empty(w2.shape, executor_ctx)
            #     w2.copyto(w2_gpu)
            #     sgd_update_gpu(w1_gpu, w2_gpu, w3)
            #     w1_gpu.copyto(w1)
            #     w2_gpu.copyto(w2)
            #
            # sgd_update_cpu(W1_val, grad_W1_val, lr)
            # sgd_update_cpu(W2_val, grad_W2_val, lr)
            # sgd_update_cpu(W3_val, grad_W3_val, lr)
            # sgd_update_cpu(b1_val, grad_b1_val, lr)
            # sgd_update_cpu(b2_val, grad_b2_val, lr)
            # sgd_update_cpu(b3_val, grad_b3_val, lr)

            # print(loss_val.asnumpy())
        if print_loss_val_each_epoch:
            if isinstance(loss_val, ndarray.NDArray):
                print(loss_val.asnumpy())
            else:
                print(loss_val)
    f1.close()
    print("success")
    return

    correct_predictions = []
    for minibatch_index in range(n_valid_batches):
        minibatch_start = minibatch_index * batch_size
        minibatch_end = (minibatch_index + 1) * batch_size
        valid_X_val[:] = valid_set_x[minibatch_start:minibatch_end]
        valid_y_val[:] = convert_to_one_hot(
            valid_set_y[minibatch_start:minibatch_end])
        res = executor.run(
            feed_dict={
                X: X_val,
                y_: y_val,
                W1: W1_val,
                W2: W2_val,
                W3: W3_val,
                W4: W4_val,
                W5: W5_val,
                W6: W6_val,
                b1: b1_val,
                b2: b2_val,
                b3: b3_val,
                b4: b4_val,
                b5: b5_val,
                b6: b6_val,
                executor.Variable_node_to_mv[W1][0]: W1_val_m,
                executor.Variable_node_to_mv[W2][0]: W2_val_m,
                executor.Variable_node_to_mv[W3][0]: W3_val_m,
                executor.Variable_node_to_mv[W4][0]: W4_val_m,
                executor.Variable_node_to_mv[W5][0]: W5_val_m,
                executor.Variable_node_to_mv[W6][0]: W6_val_m,
                executor.Variable_node_to_mv[b1][0]: b1_val_m,
                executor.Variable_node_to_mv[b2][0]: b2_val_m,
                executor.Variable_node_to_mv[b3][0]: b3_val_m,
                executor.Variable_node_to_mv[b4][0]: b4_val_m,
                executor.Variable_node_to_mv[b5][0]: b5_val_m,
                executor.Variable_node_to_mv[b6][0]: b6_val_m,
                executor.Variable_node_to_mv[W1][1]: W1_val_v,
                executor.Variable_node_to_mv[W2][1]: W2_val_v,
                executor.Variable_node_to_mv[W3][1]: W3_val_v,
                executor.Variable_node_to_mv[W4][1]: W4_val_v,
                executor.Variable_node_to_mv[W5][1]: W5_val_v,
                executor.Variable_node_to_mv[W6][1]: W6_val_v,
                executor.Variable_node_to_mv[b1][1]: b1_val_v,
                executor.Variable_node_to_mv[b2][1]: b2_val_v,
                executor.Variable_node_to_mv[b3][1]: b3_val_v,
                executor.Variable_node_to_mv[b4][1]: b4_val_v,
                executor.Variable_node_to_mv[b5][1]: b5_val_v,
                executor.Variable_node_to_mv[b6][1]: b6_val_v},
            convert_to_numpy_ret_vals=True)
        valid_y_predicted = res[-1]
        correct_prediction = np.equal(
            np.argmax(valid_y_val, 1),
            np.argmax(valid_y_predicted, 1)).astype(np.float)
        correct_predictions.extend(correct_prediction)
    accuracy = np.mean(correct_predictions)
    # validation set accuracy=0.970800
    print("validation set accuracy=%f" % accuracy)


class GPURecord(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(GPU)
        self.f = open("log/VGG x2 new/scheduled/gpu_record.txt", "w+")
        # todo 临时用作释放的计数器
        self.times = 0
        self.max_gpu_memory = 0
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.base_used = meminfo.used / 1024 ** 2

    def run(self):
        while True:
            # if self.times == 30000:
            #     self.f.close()
            #     break
            self.times += 1
            # time.sleep(0.1)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            memory_used = meminfo.used / 1024 ** 2
            if memory_used>self.max_gpu_memory:
                self.max_gpu_memory = memory_used
            print("time", datetime.datetime.now(),
                  "\tmemory", memory_used,
                  "\tmax_memory_used", self.max_gpu_memory,
                  "\tretained_memory_used", memory_used-self.base_used,
                  "\tretained_max_memory_used", self.max_gpu_memory-self.base_used, file=self.f)  # 已用显存大小
            self.f.flush()

    def stop(self):
        self.f.close()


if __name__ == '__main__':
    gpu_record = GPURecord()
    global_message_queue = multiprocessing.Queue()
    global_control_queue = multiprocessing.Queue()

    top_control_queue_list = []
    top_message_queue_list = []

    executor_ctx = ndarray.gpu(0)
    num_epochs = 20
    print_loss_val_each_epoch = True

    top_control_queue = multiprocessing.Queue()
    top_control_queue_list.append(top_control_queue)
    top_message_queue = multiprocessing.Queue()
    top_message_queue_list.append(top_message_queue)
    job_number = 1

    p1 = Process(target=mnist_mlp,
                 args=(executor_ctx, num_epochs, print_loss_val_each_epoch, top_control_queue, top_message_queue))
    p1.start()

    top_control_queue2 = multiprocessing.Queue()
    top_control_queue_list.append(top_control_queue2)
    top_message_queue2 = multiprocessing.Queue()
    top_message_queue_list.append(top_message_queue2)
    job_number += 1

    p2 = Process(target=mnist_mlp,
                 args=(executor_ctx, num_epochs, print_loss_val_each_epoch, top_control_queue2, top_message_queue2))
    p2.start()

    scheduler = Process(target=mp.multiprocess_init, args=(global_message_queue, global_control_queue))
    scheduler.start()
    # scheduler.join()

    #　gpu_record.start()
    while True:
        for i in range(job_number):
            if not top_message_queue_list[i].empty():
                print("job ", i, "message")
                global_message_queue.put([i, top_message_queue_list[i].get()])
        if not global_control_queue.empty():
            global_control = global_control_queue.get()
            for i in range(job_number):
                if i in global_control:
                    print("job ", i, "control")
                    top_control_queue_list[i].put(global_control[i])






    # todo 算法传入系统的信息规则
    # 上层传入下层包括三个list: swap list, release list, recomputation list
    # 上层写入下层的每次的control message：task_id, node_id, start_time, start_node, move_to_gpu, start_node_type, recompute
    # 根据task_id选择对应的control_queue，将其余所有信息作为一个整体list放入queue中。
    # 顺序为(start_node, start_node_type, start_time, node_id, move_to_gpu, recompute)
    # 此处保证start_time按照顺序排布
    # move_to_gpu: false means cpu, true means gpu
    # start_node_type: 0 means input_time, 1 means output_time
    # recompute: false means swap, true means recompute
    # 此处全部为index

