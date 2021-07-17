from multiprocessing import Process
from pycode.tinyflow import ndarray, gpu_op
from pycode.tinyflow import autodiff as ad
from pycode.tinyflow import mainV2 as mp
import six.moves.cPickle as pickle
import gzip
import numpy as np
import os
import queue
import multiprocessing

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

    W1 = ad.Variable(name="W1")
    W2 = ad.Variable(name="W2")
    W3 = ad.Variable(name="W3")
    b1 = ad.Variable(name="b1")
    b2 = ad.Variable(name="b2")
    b3 = ad.Variable(name="b3")
    X = ad.Variable(name="X")
    y_ = ad.Variable(name="y_")

    # 下面是三层网络的激活函数，两个relu和一个softmax

    # relu(X W1+b1)
    z1 = ad.matmul_op(X, W1)
    z2 = z1 + ad.broadcastto_op(b1, z1)
    z3 = ad.relu_op(z2)

    # relu(z3 W2+b2)
    z4 = ad.matmul_op(z3, W2)
    z5 = z4 + ad.broadcastto_op(b2, z4)
    z6 = ad.relu_op(z5)

    # softmax(z5 W2+b2)
    z7 = ad.matmul_op(z6, W3)
    y = z7 + ad.broadcastto_op(b3, z7)

    loss = ad.softmaxcrossentropy_op(y, y_)

    grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3 = ad.gradients(
        loss, [W1, W2, W3, b1, b2, b3])


    # 此处向前为符号定义


    # 只声明，不操作
    executor = ad.Executor(
        [loss, grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3, y],
        ctx=executor_ctx, top_control_queue=top_control_queue, top_message_queue=top_message_queue)

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
    W1_val = rand.normal(scale=0.1, size=(784, 256))
    W2_val = rand.normal(scale=0.1, size=(256, 100))
    W3_val = rand.normal(scale=0.1, size=(100, 10))
    b1_val = rand.normal(scale=0.1, size=(256))
    b2_val = rand.normal(scale=0.1, size=(100))
    b3_val = rand.normal(scale=0.1, size=(10))
    X_val = np.empty(shape=(batch_size, 784), dtype=np.float32)
    y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)
    valid_X_val = np.empty(shape=(batch_size, 784), dtype=np.float32)
    valid_y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)

    # todo 此处修改回gpu
    W1_val = ndarray.array(W1_val, ctx=executor_ctx)
    W2_val = ndarray.array(W2_val, ctx=executor_ctx)
    W3_val = ndarray.array(W3_val, ctx=executor_ctx)
    b1_val = ndarray.array(b1_val, ctx=executor_ctx)
    b2_val = ndarray.array(b2_val, ctx=executor_ctx)
    b3_val = ndarray.array(b3_val, ctx=executor_ctx)
    X_val = ndarray.array(X_val, ctx=executor_ctx)
    y_val = ndarray.array(y_val, ctx=executor_ctx)

    # 此处以上将数据分别转化为cpu和gpu两种格式



    lr = 1.0e-3
    for i in range(num_epochs):
        print("epoch %d" % i)
        for minibatch_index in range(n_train_batches):
            minibatch_start = minibatch_index * batch_size
            minibatch_end = (minibatch_index + 1) * batch_size
            X_val[:] = train_set_x[minibatch_start:minibatch_end]
            y_val[:] = convert_to_one_hot(
                train_set_y[minibatch_start:minibatch_end])


            # 计算单步的梯度
            loss_val, grad_W1_val, grad_W2_val, grad_W3_val, \
                grad_b1_val, grad_b2_val, grad_b3_val, _ = executor.run(
                    feed_dict={
                        X: X_val,
                        y_: y_val,
                        W1: W1_val,
                        W2: W2_val,
                        W3: W3_val,
                        b1: b1_val,
                        b2: b2_val,
                        b3: b3_val})


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

            sgd_update_gpu(W1_val, grad_W1_val, lr, cuda_stream)
            sgd_update_gpu(W2_val, grad_W2_val, lr, cuda_stream)
            sgd_update_gpu(W3_val, grad_W3_val, lr, cuda_stream)
            sgd_update_gpu(b1_val, grad_b1_val, lr, cuda_stream)
            sgd_update_gpu(b2_val, grad_b2_val, lr, cuda_stream)
            sgd_update_gpu(b3_val, grad_b3_val, lr, cuda_stream)
            # print(loss_val.asnumpy())
        if print_loss_val_each_epoch:
            if isinstance(loss_val, ndarray.NDArray):
                print(loss_val.asnumpy())
            else:
                print(loss_val)

    print("success")
    return

    correct_predictions = []
    for minibatch_index in range(n_valid_batches):
        minibatch_start = minibatch_index * batch_size
        minibatch_end = (minibatch_index + 1) * batch_size
        valid_X_val[:] = valid_set_x[minibatch_start:minibatch_end]
        valid_y_val[:] = convert_to_one_hot(
            valid_set_y[minibatch_start:minibatch_end])
        _, _, _, _, _, _, _, valid_y_predicted = executor.run(
            feed_dict={
                X: valid_X_val,
                y_: valid_y_val,
                W1: W1_val,
                W2: W2_val,
                W3: W3_val,
                b1: b1_val,
                b2: b2_val,
                b3: b3_val},
            convert_to_numpy_ret_vals=True)
        correct_prediction = np.equal(
            np.argmax(valid_y_val, 1),
            np.argmax(valid_y_predicted, 1)).astype(np.float)
        correct_predictions.extend(correct_prediction)
    accuracy = np.mean(correct_predictions)
    # validation set accuracy=0.970800
    print("validation set accuracy=%f" % accuracy)


if __name__ == '__main__':


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

    p1 = Process(target=mnist_mlp, args=(executor_ctx, num_epochs, print_loss_val_each_epoch, top_control_queue, top_message_queue))
    p1.start()
    # p1.join()

    scheduler = Process(target=mp.multiprocess_init, args=(global_message_queue, global_control_queue))
    scheduler.start()
    # scheduler.join()


    while True:
        for i in range(job_number):
            if not top_message_queue_list[i].empty():
                global_message_queue.put([i, top_message_queue.get()])
        if not global_control_queue.empty():
            global_control = global_control_queue.get()
            for i in range(job_number):
                top_control_queue.put(global_control[i])


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

