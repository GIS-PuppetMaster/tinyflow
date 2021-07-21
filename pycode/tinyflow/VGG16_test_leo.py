import os

GPU = 2
os.environ['CUDA_VISIBLE_DEVICES'] = f'{GPU}'
import sys

sys.path.append('../../')
from pycode.tinyflow import autodiff as ad
from pycode.tinyflow.get_result import get_result
from util import *


class VGG16():
    def __init__(self, num_step, batch_size, log_path, job_id):
        self.job_id = job_id
        self.dropout_rate = 0.5
        self.image_channel = 3
        self.image_size = 224
        self.num_step = num_step
        self.batch_size = batch_size
        self.log_path = log_path
        self.executor_ctx = None
        self.n_class = None
        self.ad = ad
        self.top_control_queue = None
        self.top_message_queue = None

    def get_predict_results(self, n_class, **kwargs):
        X = self.ad.Placeholder("X")
        y_ = self.ad.Placeholder("y_")
        W1_1 = self.ad.Variable("W1_1")
        W1_2 = self.ad.Variable("W1_2")
        W2_1 = self.ad.Variable("W2_1")
        W2_2 = self.ad.Variable("W2_2")
        W3_1 = self.ad.Variable("W3_1")
        W3_2 = self.ad.Variable("W3_2")
        W3_3 = self.ad.Variable("W3_3")
        W4_1 = self.ad.Variable("W4_1")
        W4_2 = self.ad.Variable("W4_2")
        W4_3 = self.ad.Variable("W4_3")
        W5_1 = self.ad.Variable("W5_1")
        W5_2 = self.ad.Variable("W5_2")
        W5_3 = self.ad.Variable("W5_3")
        W6 = self.ad.Variable("W6")
        W7 = self.ad.Variable("W7")
        W8 = self.ad.Variable("W8")
        b6 = self.ad.Variable("b6")
        b7 = self.ad.Variable("b7")
        b8 = self.ad.Variable("b8")

        # conv 1
        conv1_1 = self.ad.convolution_2d_forward_op(X, W1_1, "NCHW", "SAME", 1, 1)
        act1_1 = self.ad.activation_forward_op(conv1_1, "NCHW", "relu")

        conv1_2 = self.ad.convolution_2d_forward_op(act1_1, W1_2, "NCHW", "SAME", 1, 1)
        act1_2 = self.ad.activation_forward_op(conv1_2, "NCHW", "relu")
        pool1 = self.ad.pooling_2d_forward_op(act1_2, "NCHW", "max", 0, 0, 2, 2, 2, 2)

        # conv 2
        conv2_1 = self.ad.convolution_2d_forward_op(pool1, W2_1, "NCHW", "SAME", 1, 1)
        act2_1 = self.ad.activation_forward_op(conv2_1, "NCHW", "relu")
        conv2_2 = self.ad.convolution_2d_forward_op(act2_1, W2_2, "NCHW", "SAME", 1, 1)
        act2_2 = self.ad.activation_forward_op(conv2_2, "NCHW", "relu")
        pool2 = self.ad.pooling_2d_forward_op(act2_2, "NCHW", "max", 0, 0, 2, 2, 2, 2)

        # conv 3
        conv3_1 = self.ad.convolution_2d_forward_op(pool2, W3_1, "NCHW", "SAME", 1, 1)
        act3_1 = self.ad.activation_forward_op(conv3_1, "NCHW", "relu")
        conv3_2 = self.ad.convolution_2d_forward_op(act3_1, W3_2, "NCHW", "SAME", 1, 1)
        act3_2 = self.ad.activation_forward_op(conv3_2, "NCHW", "relu")
        conv3_3 = self.ad.convolution_2d_forward_op(act3_2, W3_3, "NCHW", "SAME", 1, 1)
        act3_3 = self.ad.activation_forward_op(conv3_3, "NCHW", "relu")
        pool3 = self.ad.pooling_2d_forward_op(act3_3, "NCHW", "max", 0, 0, 2, 2, 2, 2)

        # conv 4
        conv4_1 = self.ad.convolution_2d_forward_op(pool3, W4_1, "NCHW", "SAME", 1, 1)
        act4_1 = self.ad.activation_forward_op(conv4_1, "NCHW", "relu")
        conv4_2 = self.ad.convolution_2d_forward_op(act4_1, W4_2, "NCHW", "SAME", 1, 1)
        act4_2 = self.ad.activation_forward_op(conv4_2, "NCHW", "relu")
        conv4_3 = self.ad.convolution_2d_forward_op(act4_2, W4_3, "NCHW", "SAME", 1, 1)
        act4_3 = self.ad.activation_forward_op(conv4_3, "NCHW", "relu")
        pool4 = self.ad.pooling_2d_forward_op(act4_3, "NCHW", "max", 0, 0, 2, 2, 2, 2)

        # conv 5
        conv5_1 = self.ad.convolution_2d_forward_op(pool4, W5_1, "NCHW", "SAME", 1, 1)
        act5_1 = self.ad.activation_forward_op(conv5_1, "NCHW", "relu")
        conv5_2 = self.ad.convolution_2d_forward_op(act5_1, W5_2, "NCHW", "SAME", 1, 1)
        act5_2 = self.ad.activation_forward_op(conv5_2, "NCHW", "relu")
        conv5_3 = self.ad.convolution_2d_forward_op(act5_2, W5_3, "NCHW", "SAME", 1, 1)
        act5_3 = self.ad.activation_forward_op(conv5_3, "NCHW", "relu")
        pool5 = self.ad.pooling_2d_forward_op(act5_3, "NCHW", "max", 0, 0, 2, 2, 2, 2)

        # fc6
        pool5_flat = self.ad.flatten_op(pool5)
        fc6 = self.ad.dense(pool5_flat, W6, b6)
        act6 = self.ad.fullyactivation_forward_op(fc6, "NCHW", "relu")
        drop6 = self.ad.fullydropout_forward_op(act6, "NCHW", self.dropout_rate)

        # fc7
        fc7 = self.ad.dense(drop6, W7, b7)
        act7 = self.ad.fullyactivation_forward_op(fc7, "NCHW", "relu")
        drop7 = self.ad.fullydropout_forward_op(act7, "NCHW", self.dropout_rate)

        # fc8
        fc8 = self.ad.dense(drop7, W8, b8)
        bn8 = self.ad.fullybn_forward_op(fc8, "NCHW")
        y = self.ad.fullyactivation_forward_op(bn8, "NCHW", "softmax")

        loss = self.ad.crossEntropy_loss(y, y_)
        W1_1_val = (64, self.image_channel, 3, 3)
        W1_2_val = (64, 64, 3, 3)
        W2_1_val = (128, 64, 3, 3)
        W2_2_val = (128, 128, 3, 3)
        W3_1_val = (256, 128, 3, 3)
        W3_2_val = (256, 256, 3, 3)
        W3_3_val = (256, 256, 3, 3)
        W4_1_val = (512, 256, 3, 3)
        W4_2_val = (512, 512, 3, 3)
        W4_3_val = (512, 512, 3, 3)
        W5_1_val = (512, 512, 3, 3)
        W5_2_val = (512, 512, 3, 3)
        W5_3_val = (512, 512, 3, 3)
        W6_val = (512 * int(self.image_size / 32) * int(self.image_size / 32), 4096)
        W7_val = (4096, 4096)
        W8_val = (4096, n_class)
        b6_val = (4096,)
        b7_val = (4096,)
        b8_val = (n_class,)

        # 只声明，不操作
        executor = self.ad.Executor(loss, y, 0.001, top_control_queue=None, top_message_queue=None, log_path=self.log_path, **kwargs)
        feed_dict = {
            W1_1: W1_1_val,
            W1_2: W1_2_val,
            W2_1: W2_1_val,
            W2_2: W2_2_val,
            W3_1: W3_1_val,
            W3_2: W3_2_val,
            W3_3: W3_3_val,
            W4_1: W4_1_val,
            W4_2: W4_2_val,
            W4_3: W4_3_val,
            W5_1: W5_1_val,
            W5_2: W5_2_val,
            W5_3: W5_3_val,
            W6: W6_val,
            W7: W7_val,
            W8: W8_val,
            b6: b6_val,
            b7: b7_val,
            b8: b8_val
        }
        feed_dict_mv = {}
        for key, value in feed_dict.items():
            m_key = executor.Variable_node_to_mv[key][0]
            m_val = value
            v_key = executor.Variable_node_to_mv[key][1]
            v_val = value
            feed_dict_mv.update({m_key: m_val, v_key: v_val})
        X_val = (self.batch_size, self.image_channel, self.image_size, self.image_size)  # number = batch_size  channel = 3  image_size = 224*224
        y_val = (self.batch_size, 1000)  # n_class = 1000
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
        W1_1 = self.ad.Variable("W1_1")
        W1_2 = self.ad.Variable("W1_2")
        W2_1 = self.ad.Variable("W2_1")
        W2_2 = self.ad.Variable("W2_2")
        W3_1 = self.ad.Variable("W3_1")
        W3_2 = self.ad.Variable("W3_2")
        W3_3 = self.ad.Variable("W3_3")
        W4_1 = self.ad.Variable("W4_1")
        W4_2 = self.ad.Variable("W4_2")
        W4_3 = self.ad.Variable("W4_3")
        W5_1 = self.ad.Variable("W5_1")
        W5_2 = self.ad.Variable("W5_2")
        W5_3 = self.ad.Variable("W5_3")
        W6 = self.ad.Variable("W6")
        W7 = self.ad.Variable("W7")
        W8 = self.ad.Variable("W8")
        b6 = self.ad.Variable("b6")
        b7 = self.ad.Variable("b7")
        b8 = self.ad.Variable("b8")

        # conv 1
        conv1_1 = self.ad.convolution_2d_forward_op(X, W1_1, "NCHW", "SAME", 1, 1)
        act1_1 = self.ad.activation_forward_op(conv1_1, "NCHW", "relu")

        conv1_2 = self.ad.convolution_2d_forward_op(act1_1, W1_2, "NCHW", "SAME", 1, 1)
        act1_2 = self.ad.activation_forward_op(conv1_2, "NCHW", "relu")
        pool1 = self.ad.pooling_2d_forward_op(act1_2, "NCHW", "max", 0, 0, 2, 2, 2, 2)

        # conv 2
        conv2_1 = self.ad.convolution_2d_forward_op(pool1, W2_1, "NCHW", "SAME", 1, 1)
        act2_1 = self.ad.activation_forward_op(conv2_1, "NCHW", "relu")
        conv2_2 = self.ad.convolution_2d_forward_op(act2_1, W2_2, "NCHW", "SAME", 1, 1)
        act2_2 = self.ad.activation_forward_op(conv2_2, "NCHW", "relu")
        pool2 = self.ad.pooling_2d_forward_op(act2_2, "NCHW", "max", 0, 0, 2, 2, 2, 2)

        # conv 3
        conv3_1 = self.ad.convolution_2d_forward_op(pool2, W3_1, "NCHW", "SAME", 1, 1)
        act3_1 = self.ad.activation_forward_op(conv3_1, "NCHW", "relu")
        conv3_2 = self.ad.convolution_2d_forward_op(act3_1, W3_2, "NCHW", "SAME", 1, 1)
        act3_2 = self.ad.activation_forward_op(conv3_2, "NCHW", "relu")
        conv3_3 = self.ad.convolution_2d_forward_op(act3_2, W3_3, "NCHW", "SAME", 1, 1)
        act3_3 = self.ad.activation_forward_op(conv3_3, "NCHW", "relu")
        pool3 = self.ad.pooling_2d_forward_op(act3_3, "NCHW", "max", 0, 0, 2, 2, 2, 2)

        # conv 4
        conv4_1 = self.ad.convolution_2d_forward_op(pool3, W4_1, "NCHW", "SAME", 1, 1)
        act4_1 = self.ad.activation_forward_op(conv4_1, "NCHW", "relu")
        conv4_2 = self.ad.convolution_2d_forward_op(act4_1, W4_2, "NCHW", "SAME", 1, 1)
        act4_2 = self.ad.activation_forward_op(conv4_2, "NCHW", "relu")
        conv4_3 = self.ad.convolution_2d_forward_op(act4_2, W4_3, "NCHW", "SAME", 1, 1)
        act4_3 = self.ad.activation_forward_op(conv4_3, "NCHW", "relu")
        pool4 = self.ad.pooling_2d_forward_op(act4_3, "NCHW", "max", 0, 0, 2, 2, 2, 2)

        # conv 5
        conv5_1 = self.ad.convolution_2d_forward_op(pool4, W5_1, "NCHW", "SAME", 1, 1)
        act5_1 = self.ad.activation_forward_op(conv5_1, "NCHW", "relu")
        conv5_2 = self.ad.convolution_2d_forward_op(act5_1, W5_2, "NCHW", "SAME", 1, 1)
        act5_2 = self.ad.activation_forward_op(conv5_2, "NCHW", "relu")
        conv5_3 = self.ad.convolution_2d_forward_op(act5_2, W5_3, "NCHW", "SAME", 1, 1)
        act5_3 = self.ad.activation_forward_op(conv5_3, "NCHW", "relu")
        pool5 = self.ad.pooling_2d_forward_op(act5_3, "NCHW", "max", 0, 0, 2, 2, 2, 2)

        # fc6
        pool5_flat = self.ad.flatten_op(pool5)
        fc6 = self.ad.dense(pool5_flat, W6, b6)
        act6 = self.ad.fullyactivation_forward_op(fc6, "NCHW", "relu")
        drop6 = self.ad.fullydropout_forward_op(act6, "NCHW", self.dropout_rate)

        # fc7
        fc7 = self.ad.dense(drop6, W7, b7)
        act7 = self.ad.fullyactivation_forward_op(fc7, "NCHW", "relu")
        drop7 = self.ad.fullydropout_forward_op(act7, "NCHW", self.dropout_rate)

        # fc8
        fc8 = self.ad.dense(drop7, W8, b8)
        bn8 = self.ad.fullybn_forward_op(fc8, "NCHW")
        y = self.ad.fullyactivation_forward_op(bn8, "NCHW", "softmax")

        loss = self.ad.crossEntropy_loss(y, y_)
        W1_1_val = ndarray.array(np.random.normal(0.0, 0.1, (64, self.image_channel, 3, 3)), executor_ctx)
        W1_2_val = ndarray.array(np.random.normal(0.0, 0.1, (64, 64, 3, 3)), executor_ctx)
        W2_1_val = ndarray.array(np.random.normal(0.0, 0.1, (128, 64, 3, 3)), executor_ctx)
        W2_2_val = ndarray.array(np.random.normal(0.0, 0.1, (128, 128, 3, 3)), executor_ctx)
        W3_1_val = ndarray.array(np.random.normal(0.0, 0.1, (256, 128, 3, 3)), executor_ctx)
        W3_2_val = ndarray.array(np.random.normal(0.0, 0.1, (256, 256, 3, 3)), executor_ctx)
        W3_3_val = ndarray.array(np.random.normal(0.0, 0.1, (256, 256, 3, 3)), executor_ctx)
        W4_1_val = ndarray.array(np.random.normal(0.0, 0.1, (512, 256, 3, 3)), executor_ctx)
        W4_2_val = ndarray.array(np.random.normal(0.0, 0.1, (512, 512, 3, 3)), executor_ctx)
        W4_3_val = ndarray.array(np.random.normal(0.0, 0.1, (512, 512, 3, 3)), executor_ctx)
        W5_1_val = ndarray.array(np.random.normal(0.0, 0.1, (512, 512, 3, 3)), executor_ctx)
        W5_2_val = ndarray.array(np.random.normal(0.0, 0.1, (512, 512, 3, 3)), executor_ctx)
        W5_3_val = ndarray.array(np.random.normal(0.0, 0.1, (512, 512, 3, 3)), executor_ctx)
        W6_val = ndarray.array(np.random.normal(0.0, 0.1, (512 * int(self.image_size / 32) * int(self.image_size / 32), 4096)), executor_ctx)
        W7_val = ndarray.array(np.random.normal(0.0, 0.1, (4096, 4096)) * 0.001, executor_ctx)
        W8_val = ndarray.array(np.random.normal(0.0, 0.1, (4096, n_class)) * 0.001, executor_ctx)
        b6_val = ndarray.array(np.ones(4096) * 0.1, executor_ctx)
        b7_val = ndarray.array(np.ones(4096) * 0.1, executor_ctx)
        b8_val = ndarray.array(np.ones(n_class) * 0.1, executor_ctx)

        # 只声明，不操作
        executor = self.ad.Executor(loss, y, 0.001, top_control_queue=top_control_queue, top_message_queue=top_message_queue, log_path=self.log_path)
        feed_dict = {
            W1_1: W1_1_val,
            W1_2: W1_2_val,
            W2_1: W2_1_val,
            W2_2: W2_2_val,
            W3_1: W3_1_val,
            W3_2: W3_2_val,
            W3_3: W3_3_val,
            W4_1: W4_1_val,
            W4_2: W4_2_val,
            W4_3: W4_3_val,
            W5_1: W5_1_val,
            W5_2: W5_2_val,
            W5_3: W5_3_val,
            W6: W6_val,
            W7: W7_val,
            W8: W8_val,
            b6: b6_val,
            b7: b7_val,
            b8: b8_val
        }
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
                main(path, repeat, jobs_num, batch_size, VGG16, **kwargs)
            elif skip != 'vanilla':
                path = raw_path + 'vanilla'
                print(path)
                main(path, repeat, jobs_num, batch_size, VGG16, **kwargs)
        if analysis_result:
            get_result(raw_path, repeat)


if __name__ == '__main__':
    run_exp([['./log/VGG test/', 1, 1, 2]])
    # raw_path='./log/VGG/'
    # repeat=3
    # get_result(raw_path, repeat)
