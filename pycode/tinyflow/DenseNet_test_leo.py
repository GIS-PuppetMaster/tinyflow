import os

GPU = 1
os.environ['CUDA_VISIBLE_DEVICES'] = f'{GPU}'
import sys

sys.path.append('../../')
from pycode.tinyflow import autodiff as ad
from pycode.tinyflow.get_result import get_result
from util import *


class DenseNet121():
    def __init__(self, num_step, batch_size, log_path, job_id):
        self.n_filter = 32  # growth rate
        self.image_channel = 3
        self.image_size = 224
        self.dropout_rate = 0.2
        self.num_step = num_step
        self.batch_size = batch_size
        self.log_path = log_path
        self.job_id = job_id
        self.executor_ctx = None
        self.n_class = None
        self.ad = ad
        self.top_control_queue = None
        self.top_message_queue = None

    def bottleneck_layer(self, inputs, in_filter, layer_name, executor_ctx=None):

        W1 = self.ad.Variable(layer_name + "_W1")
        W2 = self.ad.Variable(layer_name + "_W2")

        # 1*1 conv
        bn1 = self.ad.bn_forward_op(inputs, "NCHW", "pre_activation")
        act1 = self.ad.activation_forward_op(bn1, "NCHW", "relu")
        conv1 = self.ad.convolution_2d_forward_op(act1, W1, "NCHW", "SAME", 1, 1)
        drop1 = self.ad.dropout_forward_op(conv1, "NCHW", self.dropout_rate)

        # 3*3 conv
        bn2 = self.ad.bn_forward_op(drop1, "NCHW", "pre_activation")
        act2 = self.ad.activation_forward_op(bn2, "NCHW", "relu")
        conv2 = self.ad.convolution_2d_forward_op(act2, W2, "NCHW", "SAME", 1, 1)
        drop2 = self.ad.dropout_forward_op(conv2, "NCHW", self.dropout_rate)
        if executor_ctx is None:
            W1_val = (4 * self.n_filter, in_filter, 1, 1)  # kernel_size=1*1
            W2_val = (self.n_filter, 4 * self.n_filter, 3, 3)  # kernel_size=3*3
        else:
            W1_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(4 * self.n_filter, in_filter, 1, 1)), executor_ctx)  # kernel_size=1*1
            W2_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(self.n_filter, 4 * self.n_filter, 3, 3)), executor_ctx)  # kernel_size=3*3

        dict = {W1: W1_val, W2: W2_val}
        return drop2, dict, self.n_filter

    def dense_block(self, inputs, in_filter, nb_layers, block_name, executor_ctx=None):
        x1 = inputs
        dict1 = {}
        for i in range(nb_layers):
            x2, dict2, out_filter = self.bottleneck_layer(x1, in_filter, block_name + "_bottleneck" + str(i), executor_ctx)
            x1 = self.ad.concat_forward_op(x1, x2)
            in_filter = in_filter + out_filter
            dict1.update(dict2)
        return x1, dict1, in_filter

    def transition_layer(self, inputs, in_filter, layer_name, executor_ctx=None):

        W1 = self.ad.Variable(layer_name + "_W1")
        bn1 = self.ad.bn_forward_op(inputs, "NCHW", "pre_activation")
        act1 = self.ad.activation_forward_op(bn1, "NCHW", "relu")
        conv1 = self.ad.convolution_2d_forward_op(act1, W1, "NCHW", "SAME", 1, 1)
        drop1 = self.ad.dropout_forward_op(conv1, "NCHW", self.dropout_rate)
        pool0 = self.ad.pooling_2d_forward_op(drop1, "NCHW", "mean", 0, 0, 2, 2, 2, 2)  # stride=2   pool_size=2*2
        if executor_ctx is None:
            W1_val = (int(0.5 * in_filter), in_filter, 1, 1)  # kernel_size=1*1
        else:
            W1_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(int(0.5 * in_filter), in_filter, 1, 1)), executor_ctx)  # kernel_size=1*1
        dict = {W1: W1_val}
        return pool0, dict, int(0.5 * in_filter)

    def get_predict_results(self, n_class, **kwargs):
        X = self.ad.Placeholder("X")
        y_ = self.ad.Placeholder("y_")
        W0 = self.ad.Variable("W0")
        W1 = self.ad.Variable("W1")
        b1 = self.ad.Variable("b1")

        conv0 = self.ad.convolution_2d_forward_op(X, W0, "NCHW", "SAME", 2, 2)  # stride=2
        pool0 = self.ad.pooling_2d_forward_op(conv0, "NCHW", "max", 1, 1, 2, 2, 3, 3)  # stride=2   pool_size=3*3

        dense_1, dict_1, out_filter1 = self.dense_block(inputs=pool0, in_filter=2 * self.n_filter, nb_layers=6, block_name="dense1")
        transition_1, dict_2, out_filter2 = self.transition_layer(inputs=dense_1, in_filter=out_filter1, layer_name="trans1")

        dense_2, dict_3, out_filter3 = self.dense_block(inputs=transition_1, in_filter=out_filter2, nb_layers=12, block_name="dense2")
        transition_2, dict_4, out_filter4 = self.transition_layer(inputs=dense_2, in_filter=out_filter3, layer_name="trans2")

        dense_3, dict_5, out_filter5 = self.dense_block(inputs=transition_2, in_filter=out_filter4, nb_layers=24, block_name="dense3")
        transition_3, dict_6, out_filter6 = self.transition_layer(inputs=dense_3, in_filter=out_filter5, layer_name="trans3")

        dense_4, dict_7, out_filter7 = self.dense_block(inputs=transition_3, in_filter=out_filter6, nb_layers=16, block_name="dense4")

        bn1 = self.ad.bn_forward_op(dense_4, "NCHW", "pre_activation")
        act1 = self.ad.activation_forward_op(bn1, "NCHW", "relu")
        pool1 = self.ad.pooling_2d_forward_op(act1, "NCHW", "mean", 0, 0, 1, 1, 7, 7)  # global_pool

        flat = self.ad.flatten_op(pool1)
        dense = self.ad.dense(flat, W1, b1)
        y = self.ad.fullyactivation_forward_op(dense, "NCHW", "softmax")

        loss = self.ad.crossEntropy_loss(y, y_)

        W0_val = (2 * self.n_filter, self.image_channel, 7, 7)  # n_filter   n_channel=3   kernel_size=7*7
        W1_val = (out_filter7, n_class)
        b1_val = (n_class,)

        executor = self.ad.Executor(loss, y, 0.001, None, None, log_path=self.log_path, **kwargs)

        feed_dict = {W0: W0_val, W1: W1_val, b1: b1_val}
        feed_dict.update(dict_1)
        feed_dict.update(dict_2)
        feed_dict.update(dict_3)
        feed_dict.update(dict_4)
        feed_dict.update(dict_5)
        feed_dict.update(dict_6)
        feed_dict.update(dict_7)

        feed_dict_mv = {}
        for key, value in feed_dict.items():
            m_key = executor.Variable_node_to_mv[key][0]
            m_val = value
            v_key = executor.Variable_node_to_mv[key][1]
            v_val = value
            feed_dict_mv.update({m_key: m_val, v_key: v_val})

        feed_dict.update(feed_dict_mv)
        X_val = (self.batch_size, self.image_channel, self.image_size, self.image_size)  # number = batch_size  channel = 3  image_size = 224*224
        y_val = (self.batch_size, 1000)  # n_class = 1000
        feed_dict[X] = X_val
        feed_dict[y_] = y_val
        executor.init_operator_latency(feed_dict_sample=feed_dict, **kwargs)
        return executor.predict_results

    def run(self, executor_ctx, top_control_queue, top_message_queue, n_class, X_val, y_val, **kwargs):
        self.n_class = n_class
        self.top_control_queue = top_control_queue
        self.top_message_queue = top_message_queue
        self.executor_ctx = executor_ctx
        X = self.ad.Placeholder("X")
        y_ = self.ad.Placeholder("y_")
        W0 = self.ad.Variable("W0")
        W1 = self.ad.Variable("W1")
        b1 = self.ad.Variable("b1")

        conv0 = self.ad.convolution_2d_forward_op(X, W0, "NCHW", "SAME", 2, 2)  # stride=2
        pool0 = self.ad.pooling_2d_forward_op(conv0, "NCHW", "max", 1, 1, 2, 2, 3, 3)  # stride=2   pool_size=3*3

        dense_1, dict_1, out_filter1 = self.dense_block(inputs=pool0, in_filter=2 * self.n_filter, nb_layers=6, block_name="dense1", executor_ctx=executor_ctx)
        transition_1, dict_2, out_filter2 = self.transition_layer(inputs=dense_1, in_filter=out_filter1, layer_name="trans1", executor_ctx=executor_ctx)

        dense_2, dict_3, out_filter3 = self.dense_block(inputs=transition_1, in_filter=out_filter2, nb_layers=12, block_name="dense2", executor_ctx=executor_ctx)
        transition_2, dict_4, out_filter4 = self.transition_layer(inputs=dense_2, in_filter=out_filter3, layer_name="trans2", executor_ctx=executor_ctx)

        dense_3, dict_5, out_filter5 = self.dense_block(inputs=transition_2, in_filter=out_filter4, nb_layers=24, block_name="dense3", executor_ctx=executor_ctx)
        transition_3, dict_6, out_filter6 = self.transition_layer(inputs=dense_3, in_filter=out_filter5, layer_name="trans3", executor_ctx=executor_ctx)

        dense_4, dict_7, out_filter7 = self.dense_block(inputs=transition_3, in_filter=out_filter6, nb_layers=16, block_name="dense4", executor_ctx=executor_ctx)

        bn1 = self.ad.bn_forward_op(dense_4, "NCHW", "pre_activation")
        act1 = self.ad.activation_forward_op(bn1, "NCHW", "relu")
        pool1 = self.ad.pooling_2d_forward_op(act1, "NCHW", "mean", 0, 0, 1, 1, 7, 7)  # global_pool

        flat = self.ad.flatten_op(pool1)
        dense = self.ad.dense(flat, W1, b1)
        y = self.ad.fullyactivation_forward_op(dense, "NCHW", "softmax")

        loss = self.ad.crossEntropy_loss(y, y_)

        W0_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(2 * self.n_filter, self.image_channel, 7, 7)), executor_ctx)  # n_filter   n_channel=3   kernel_size=7*7
        W1_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(out_filter7, n_class)), executor_ctx)
        b1_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(n_class)), executor_ctx)

        executor = self.ad.Executor(loss, y, 0.001, top_control_queue=top_control_queue,
                                    top_message_queue=top_message_queue, log_path=self.log_path)

        feed_dict = {W0: W0_val, W1: W1_val, b1: b1_val}
        feed_dict.update(dict_1)
        feed_dict.update(dict_2)
        feed_dict.update(dict_3)
        feed_dict.update(dict_4)
        feed_dict.update(dict_5)
        feed_dict.update(dict_6)
        feed_dict.update(dict_7)

        feed_dict_mv = {}
        for key, value in feed_dict.items():
            m_key = executor.Variable_node_to_mv[key][0]
            m_val = ndarray.array(np.zeros(shape=value.shape), executor_ctx)
            v_key = executor.Variable_node_to_mv[key][1]
            v_val = ndarray.array(np.zeros(shape=value.shape), executor_ctx)
            feed_dict_mv.update({m_key: m_val, v_key: v_val})

        feed_dict.update(feed_dict_mv)
        if 'predict_results' in kwargs.keys():
            executor.predict_results = kwargs['predict_results']
        else:
            X_val = np.random.normal(loc=0, scale=0.1, size=(
                self.batch_size, self.image_channel, self.image_size, self.image_size))  # number = batch_size  channel = 3  image_size = 224*224
            y_val = np.random.normal(loc=0, scale=0.1, size=(self.batch_size, 1000))  # n_class = 1000
            feed_dict[X] = ndarray.array(X_val, ctx=executor_ctx)
            feed_dict[y_] = ndarray.array(y_val, ctx=executor_ctx)
            executor.init_operator_latency(feed_dict_sample=feed_dict, **kwargs)
        gpu_record_cold_start = GPURecord(self.log_path, suffix='_cold_start')
        gpu_record = GPURecord(self.log_path)
        if self.job_id == 0:
            f1 = open(f"{self.log_path}/gpu_time.txt", "w+")
        start_record = -1
        start_cold_start_record = False
        already_start_record = False
        already_start_cold_start_record = False
        for i in range(self.num_step):
            print("step", i)
            if self.job_id == 0:
                if i == 0:
                    start_time = time.time()
                    if 'vanilla' in self.log_path:
                        gpu_record_cold_start.start()
                        print(f'vanilla start_record at: {i}')
                if not already_start_record:
                    if start_cold_start_record and not already_start_cold_start_record:
                        gpu_record_cold_start.start()
                        already_start_cold_start_record = True
                        print(f'cold_start start_record at: {i}')
                    if i == start_record:
                        gpu_record.start()
                        already_start_record = True
                        print(f'start_record at: {i}')
                    if self.ad.have_got_control_message == 2 and start_record == -1:
                        print('got control message')
                        start_record = i + 1
                    if self.ad.have_got_control_message == 1:
                        start_cold_start_record = True
            feed_dict[X] = ndarray.array(X_val, ctx=self.executor_ctx)
            feed_dict[y_] = ndarray.array(y_val, ctx=self.executor_ctx)
            res = executor.run(feed_dict=feed_dict)
            loss_val = res[0]
            feed_dict = res[1]
        if self.job_id == 0:
            gpu_record_cold_start.stop()
            gpu_record.stop()
            f1.write(f'time_cost:{time.time() - start_time}')
            f1.flush()
            f1.close()
        print(loss_val)

        print("success")
        if not self.top_message_queue.empty():
            self.top_message_queue.get()
        if not self.top_control_queue.empty():
            self.top_control_queue.get()
        self.top_message_queue.close()
        self.top_control_queue.close()
        self.top_control_queue.join_thread()
        self.top_message_queue.join_thread()
        return 0

    # def run(self, executor_ctx, top_control_queue, top_message_queue, n_class, X_val, y_val, **kwargs):
    #     self.init_model(executor_ctx, n_class, top_control_queue, top_message_queue, **kwargs)
    #     return self.run_without_init(X_val, y_val)


def run_exp(workloads, analysis_result=True, skip=None, **kwargs):
    for path, repeat, jobs_num, batch_size in workloads:
        raw_path = path
        for i in range(2):
            if i == 0 and skip != 'schedule':
                path = raw_path + 'schedule'
                print(path)
                main(path, repeat, jobs_num, batch_size, DenseNet121, **kwargs)
            elif skip != 'vanilla':
                path = raw_path + 'vanilla'
                print(path)
                main(path, repeat, jobs_num, batch_size, DenseNet121, **kwargs)
        if analysis_result:
            get_result(raw_path, repeat)


if __name__ == '__main__':
    run_exp([['./log/Inception V3 bs4/', 3, 1, 4]])
