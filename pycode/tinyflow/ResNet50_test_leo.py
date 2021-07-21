import os

GPU = 1
os.environ['CUDA_VISIBLE_DEVICES'] = f'{GPU}'
import sys

sys.path.append('../../')
from pycode.tinyflow import autodiff as ad
from pycode.tinyflow.get_result import get_result
from util import *


class ResNet50():
    def __init__(self, num_step, batch_size, log_path, job_id):
        self.log_path = log_path
        self.job_id = job_id

        self.dropout_rate = 0.5
        self.image_channel = 3
        self.image_size = 224
        self.num_step = num_step
        self.batch_size = batch_size
        self.ad = ad
        self.executor_ctx = None
        self.n_class = None
        self.top_control_queue = None
        self.top_message_queue = None

    def identity_block(self, inputs, kernel_size, in_filter, out_filters, block_name, executor_ctx=None):

        f1, f2, f3 = out_filters

        W1 = self.ad.Variable(block_name + "W1")
        W2 = self.ad.Variable(block_name + "W2")
        W3 = self.ad.Variable(block_name + "W3")
        if executor_ctx is None:
            W1_val = (f1, in_filter, 1, 1)
            W2_val = (f2, f1, kernel_size, kernel_size)
            W3_val = (f3, f2, 1, 1)
        else:
            W1_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(f1, in_filter, 1, 1)), executor_ctx)
            W2_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(f2, f1, kernel_size, kernel_size)), executor_ctx)
            W3_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(f3, f2, 1, 1)), executor_ctx)

        # conv1
        conv1 = self.ad.convolution_2d_forward_op(inputs, W1, "NCHW", "SAME", 1, 1)
        bn1 = self.ad.bn_forward_op(conv1, "NCHW", "pre_activation")
        act1 = self.ad.activation_forward_op(bn1, "NCHW", "relu")

        # conv2
        conv2 = self.ad.convolution_2d_forward_op(act1, W2, "NCHW", "SAME", 1, 1)
        bn2 = self.ad.bn_forward_op(conv2, "NCHW", "pre_activation")
        act2 = self.ad.activation_forward_op(bn2, "NCHW", "relu")

        # conv3
        conv3 = self.ad.convolution_2d_forward_op(act2, W3, "NCHW", "VALID", 1, 1)
        bn3 = self.ad.bn_forward_op(conv3, "NCHW", "pre_activation")

        # shortcut
        shortcut = inputs
        add = self.ad.add_op(bn3, shortcut)
        act4 = self.ad.activation_forward_op(add, "NCHW", "relu")

        dict = {W1: W1_val, W2: W2_val, W3: W3_val}
        return act4, dict

    def convolutional_block(self, inputs, kernel_size, in_filter, out_filters, block_name, stride, executor_ctx=None):
        f1, f2, f3 = out_filters

        W1 = self.ad.Variable(block_name + "W1")
        W2 = self.ad.Variable(block_name + "W2")
        W3 = self.ad.Variable(block_name + "W3")
        W_shortcut = self.ad.Variable(block_name + "W_shortcut")
        if executor_ctx is None:
            W1_val = (f1, in_filter, 1, 1)
            W2_val = (f2, f1, kernel_size, kernel_size)
            W3_val = (f3, f2, 1, 1)
            W_shortcut_val = (f3, in_filter, 1, 1)
        else:
            W1_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(f1, in_filter, 1, 1)), executor_ctx)
            W2_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(f2, f1, kernel_size, kernel_size)), executor_ctx)
            W3_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(f3, f2, 1, 1)), executor_ctx)
            W_shortcut_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(f3, in_filter, 1, 1)), executor_ctx)

        # conv1
        conv1 = self.ad.convolution_2d_forward_op(inputs, W1, "NCHW", "VALID", stride, stride)
        bn1 = self.ad.bn_forward_op(conv1, "NCHW", "pre_activation")
        act1 = self.ad.activation_forward_op(bn1, "NCHW", "relu")

        # conv2
        conv2 = self.ad.convolution_2d_forward_op(act1, W2, "NCHW", "SAME", 1, 1)
        bn2 = self.ad.bn_forward_op(conv2, "NCHW", "pre_activation")
        act2 = self.ad.activation_forward_op(bn2, "NCHW", "relu")

        # conv3
        conv3 = self.ad.convolution_2d_forward_op(act2, W3, "NCHW", "VALID", 1, 1)
        bn3 = self.ad.bn_forward_op(conv3, "NCHW", "pre_activation")

        # shortcut_path
        conv4 = self.ad.convolution_2d_forward_op(inputs, W_shortcut, "NCHW", "VALID", stride, stride)
        shortcut = self.ad.bn_forward_op(conv4, "NCHW", "pre_activation")

        # shortcut
        add = self.ad.add_op(bn3, shortcut)
        act4 = self.ad.activation_forward_op(add, "NCHW", "relu")

        dict = {W1: W1_val, W2: W2_val, W3: W3_val, W_shortcut: W_shortcut_val}
        return act4, dict

    def get_predict_results(self, n_class, **kwargs):
        X = self.ad.Placeholder("X")
        y_ = self.ad.Placeholder("y_")
        W1 = self.ad.Variable("W1")
        W6 = self.ad.Variable("W6")
        b6 = self.ad.Variable("b6")
        W7 = self.ad.Variable("W7")
        b7 = self.ad.Variable("b7")
        # zero pading
        # pad = 3   stride=1   pool_size=1*1
        pool0 = self.ad.pooling_2d_forward_op(X, "NCHW", "max", 3, 3, 1, 1, 1, 1)  # 3*224*224 ->  3*230*230

        # conv1
        conv1 = self.ad.convolution_2d_forward_op(pool0, W1, "NCHW", "VALID", 2, 2)  # stride = 2  3*230*230 -> 64*112*112
        bn1 = self.ad.bn_forward_op(conv1, "NCHW", "pre_activation")
        act1 = self.ad.activation_forward_op(bn1, "NCHW", "relu")
        # pad = 0   stride=2   pool_size=3*3
        pool1 = self.ad.pooling_2d_forward_op(act1, "NCHW", "max", 0, 0, 2, 2, 3, 3)  # 64*112*112 -> 64*112*112

        # conv2_x
        conv2, dict2 = self.convolutional_block(inputs=pool1, kernel_size=3, in_filter=64, out_filters=[64, 64, 256], block_name="2a_", stride=1)
        iden2_1, dict2_1 = self.identity_block(inputs=conv2, kernel_size=3, in_filter=256, out_filters=[64, 64, 256], block_name="2b_")
        iden2_2, dict2_2 = self.identity_block(iden2_1, 3, 256, [64, 64, 256], "2c_")

        # conv3_x
        conv3, dict3 = self.convolutional_block(iden2_2, 3, 256, [128, 128, 512], "3a_", 2)
        iden3_1, dict3_1 = self.identity_block(conv3, 3, 512, [128, 128, 512], "3b_")
        iden3_2, dict3_2 = self.identity_block(iden3_1, 3, 512, [128, 128, 512], "3c_")
        iden3_3, dict3_3 = self.identity_block(iden3_2, 3, 512, [128, 128, 512], "3d_")

        # conv4_x
        conv4, dict4 = self.convolutional_block(iden3_3, 3, 512, [256, 256, 1024], "4a_", 2)
        iden4_1, dict4_1 = self.identity_block(conv4, 3, 1024, [256, 256, 1024], "4b_")
        iden4_2, dict4_2 = self.identity_block(iden4_1, 3, 1024, [256, 256, 1024], "4c_")
        iden4_3, dict4_3 = self.identity_block(iden4_2, 3, 1024, [256, 256, 1024], "4d_")
        iden4_4, dict4_4 = self.identity_block(iden4_3, 3, 1024, [256, 256, 1024], "4e_")
        iden4_5, dict4_5 = self.identity_block(iden4_4, 3, 1024, [256, 256, 1024], "4f_")

        # conv5_x
        conv5, dict5 = self.convolutional_block(iden4_5, 3, 1024, [512, 512, 2048], "5a_", 2)
        iden5_1, dict5_1 = self.identity_block(conv5, 3, 2048, [512, 512, 2048], "5b_")
        iden5_2, dict5_2 = self.identity_block(iden5_1, 3, 2048, [512, 512, 2048], "5c_")
        pool5 = self.ad.pooling_2d_forward_op(iden5_2, "NCHW", "mean", 0, 0, 1, 1, 7, 7)

        pool5_flat = self.ad.flatten_op(pool5)
        dense6 = self.ad.dense(pool5_flat, W6, b6)
        act6 = self.ad.fullyactivation_forward_op(dense6, "NCHW", "relu")
        drop6 = self.ad.fullydropout_forward_op(act6, "NCHW", self.dropout_rate)

        dense7 = self.ad.dense(drop6, W7, b7)
        y = self.ad.fullyactivation_forward_op(dense7, "NCHW", "softmax")

        loss = self.ad.crossEntropy_loss(y, y_)

        W1_val = (64, self.image_channel, 7, 7)
        W6_val = (1 * 1 * 2048, 2048)
        b6_val = (2048,)
        W7_val = (2048, n_class)
        b7_val = (n_class,)

        feed_dict = {W1: W1_val, W6: W6_val, W7: W7_val, b6: b6_val, b7: b7_val}
        feed_dict.update(dict2)
        feed_dict.update(dict2_1)
        feed_dict.update(dict2_2)
        feed_dict.update(dict3)
        feed_dict.update(dict3_1)
        feed_dict.update(dict3_2)
        feed_dict.update(dict3_3)
        feed_dict.update(dict4)
        feed_dict.update(dict4_1)
        feed_dict.update(dict4_2)
        feed_dict.update(dict4_3)
        feed_dict.update(dict4_4)
        feed_dict.update(dict4_5)
        feed_dict.update(dict5)
        feed_dict.update(dict5_1)
        feed_dict.update(dict5_2)

        # 只声明，不操作
        executor = self.ad.Executor(loss, y, 0.001, None, None, log_path=self.log_path, **kwargs)
        feed_dict_mv = {}
        for key, value in feed_dict.items():
            m_key = executor.Variable_node_to_mv[key][0]
            m_val = value
            v_key = executor.Variable_node_to_mv[key][1]
            v_val = value
            feed_dict_mv.update({m_key: m_val, v_key: v_val})
        X_val = (self.batch_size, self.image_channel, self.image_size, self.image_size)  # number = batch_size  channel = 3  image_size = 224*224
        y_val = (self.batch_size, 1000)
        feed_dict.update(feed_dict_mv)
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
        W1 = self.ad.Variable("W1")
        W6 = self.ad.Variable("W6")
        b6 = self.ad.Variable("b6")
        W7 = self.ad.Variable("W7")
        b7 = self.ad.Variable("b7")
        # zero pading
        # pad = 3   stride=1   pool_size=1*1
        pool0 = self.ad.pooling_2d_forward_op(X, "NCHW", "max", 3, 3, 1, 1, 1, 1)  # 3*224*224 ->  3*230*230

        # conv1
        conv1 = self.ad.convolution_2d_forward_op(pool0, W1, "NCHW", "VALID", 2, 2)  # stride = 2  3*230*230 -> 64*112*112
        bn1 = self.ad.bn_forward_op(conv1, "NCHW", "pre_activation")
        act1 = self.ad.activation_forward_op(bn1, "NCHW", "relu")
        # pad = 0   stride=2   pool_size=3*3
        pool1 = self.ad.pooling_2d_forward_op(act1, "NCHW", "max", 0, 0, 2, 2, 3, 3)  # 64*112*112 -> 64*112*112

        # conv2_x
        conv2, dict2 = self.convolutional_block(inputs=pool1, kernel_size=3, in_filter=64, out_filters=[64, 64, 256], block_name="2a_", stride=1, executor_ctx=executor_ctx)
        iden2_1, dict2_1 = self.identity_block(inputs=conv2, kernel_size=3, in_filter=256, out_filters=[64, 64, 256], block_name="2b_", executor_ctx=executor_ctx)
        iden2_2, dict2_2 = self.identity_block(iden2_1, 3, 256, [64, 64, 256], "2c_", executor_ctx)

        # conv3_x
        conv3, dict3 = self.convolutional_block(iden2_2, 3, 256, [128, 128, 512], "3a_", 2, executor_ctx)
        iden3_1, dict3_1 = self.identity_block(conv3, 3, 512, [128, 128, 512], "3b_", executor_ctx)
        iden3_2, dict3_2 = self.identity_block(iden3_1, 3, 512, [128, 128, 512], "3c_", executor_ctx)
        iden3_3, dict3_3 = self.identity_block(iden3_2, 3, 512, [128, 128, 512], "3d_", executor_ctx)

        # conv4_x
        conv4, dict4 = self.convolutional_block(iden3_3, 3, 512, [256, 256, 1024], "4a_", 2, executor_ctx)
        iden4_1, dict4_1 = self.identity_block(conv4, 3, 1024, [256, 256, 1024], "4b_", executor_ctx)
        iden4_2, dict4_2 = self.identity_block(iden4_1, 3, 1024, [256, 256, 1024], "4c_", executor_ctx)
        iden4_3, dict4_3 = self.identity_block(iden4_2, 3, 1024, [256, 256, 1024], "4d_", executor_ctx)
        iden4_4, dict4_4 = self.identity_block(iden4_3, 3, 1024, [256, 256, 1024], "4e_", executor_ctx)
        iden4_5, dict4_5 = self.identity_block(iden4_4, 3, 1024, [256, 256, 1024], "4f_", executor_ctx)

        # conv5_x
        conv5, dict5 = self.convolutional_block(iden4_5, 3, 1024, [512, 512, 2048], "5a_", 2, executor_ctx)
        iden5_1, dict5_1 = self.identity_block(conv5, 3, 2048, [512, 512, 2048], "5b_", executor_ctx)
        iden5_2, dict5_2 = self.identity_block(iden5_1, 3, 2048, [512, 512, 2048], "5c_", executor_ctx)
        pool5 = self.ad.pooling_2d_forward_op(iden5_2, "NCHW", "mean", 0, 0, 1, 1, 7, 7)

        pool5_flat = self.ad.flatten_op(pool5)
        dense6 = self.ad.dense(pool5_flat, W6, b6)
        act6 = self.ad.fullyactivation_forward_op(dense6, "NCHW", "relu")
        drop6 = self.ad.fullydropout_forward_op(act6, "NCHW", self.dropout_rate)

        dense7 = self.ad.dense(drop6, W7, b7)
        y = self.ad.fullyactivation_forward_op(dense7, "NCHW", "softmax")

        loss = self.ad.crossEntropy_loss(y, y_)

        W1_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(64, self.image_channel, 7, 7)), executor_ctx)
        W6_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(1 * 1 * 2048, 2048)), executor_ctx)
        b6_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(2048)), executor_ctx)
        W7_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(2048, n_class)), executor_ctx)
        b7_val = ndarray.array(np.random.normal(loc=0, scale=0.1, size=(n_class)), executor_ctx)

        feed_dict = {W1: W1_val, W6: W6_val, W7: W7_val, b6: b6_val, b7: b7_val}
        feed_dict.update(dict2)
        feed_dict.update(dict2_1)
        feed_dict.update(dict2_2)
        feed_dict.update(dict3)
        feed_dict.update(dict3_1)
        feed_dict.update(dict3_2)
        feed_dict.update(dict3_3)
        feed_dict.update(dict4)
        feed_dict.update(dict4_1)
        feed_dict.update(dict4_2)
        feed_dict.update(dict4_3)
        feed_dict.update(dict4_4)
        feed_dict.update(dict4_5)
        feed_dict.update(dict5)
        feed_dict.update(dict5_1)
        feed_dict.update(dict5_2)

        # 只声明，不操作
        executor = self.ad.Executor(loss, y, 0.001, top_control_queue=top_control_queue,
                                         top_message_queue=top_message_queue, log_path=self.log_path)

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

        gpu_record_cold_start = GPURecord(self.log_path,suffix='_cold_start')
        gpu_record = GPURecord(self.log_path)
        if self.job_id == 0:
            f1 = open(f"{self.log_path}/gpu_time.txt", "w+")
        start_record = -1
        already_start_record = False
        for i in range(self.num_step):
            print("step", i)
            if self.job_id == 0:
                if i == 0:
                    gpu_record_cold_start.start()
                    start_time = time.time()
                if not already_start_record:
                    if i == start_record:
                        gpu_record.start()
                        already_start_record = True
                        print('start_record')
                    if self.ad.have_got_control_message and start_record==-1:
                        print('got control message')
                        start_record = i+5
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
                main(path, repeat, jobs_num, batch_size, ResNet50, **kwargs)
            elif skip != 'vanilla':
                path = raw_path + 'vanilla'
                print(path)
                main(path, repeat, jobs_num, batch_size, ResNet50, **kwargs)
        if analysis_result:
            get_result(raw_path, repeat)



if __name__ == '__main__':
    run_exp([['./log/ResNet x1/', 1, 1, 2]], skip='vanilla')
