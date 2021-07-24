# from pycode.tinyflow import autodiff_vdnn as ad
# from pycode.tinyflow import gpu_op, train, ndarray, TrainExecute, TrainExecute_Adam_vDNNconv
import traceback

import numpy as np
import imp, os, datetime

from pycode.tinyflow import ndarray
from tests.Experiment import record_GPU
from multiprocessing import Process

tinyflow_path = "../../pycode/tinyflow/"


class DenseNet121(Process):
    def __init__(self, num_step, type, batch_size, gpu_num, path, file_name, need_tosave=None):
        super().__init__()
        self.need_tosave = need_tosave
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
        self.gpu_num = gpu_num
        self.n_filter = 32  # growth rate
        self.image_channel = 3
        self.dropout_rate = 0.2
        self.num_step = num_step
        self.batch_size = batch_size
        self.is_capu = False
        self.path = path
        self.file_name = file_name
        self.f1 = open(self.path + 'type' + str(type) + file_name + '_record_1.txt', 'w')
        self.f3 = open(self.path + 'type' + str(type) + file_name + '_record_3.txt', 'w')
        self.f6 = open(self.path + 'type' + str(type) + file_name + '_record_6.txt', 'w')
        self.f7 = open(self.path + 'type' + str(type) + file_name + '_record_7.txt', 'w')

        self.type = type
        if type == 0:
            self.autodiff_name = "autodiff.py"
            self.TrainExecute_name = "TrainExecuteAdam.py"
        elif type == 1:
            self.autodiff_name = "autodiff_capu.py"
            self.TrainExecute_name = "TrainExecuteAdam_Capu.py"
            self.is_capu = True
        elif type == 2:
            self.autodiff_name = "autodiff_vdnn.py"
            self.TrainExecute_name = "TrainExecuteAdam_vDNNconv.py"
        elif type == 3:
            self.autodiff_name = "autodiff.py"
            self.TrainExecute_name = "TrainExecuteAdam.py"
        self.ad = imp.load_source(self.autodiff_name, tinyflow_path + self.autodiff_name)
        self.TrainExecute = imp.load_source(self.autodiff_name, tinyflow_path + self.TrainExecute_name)

    def bottleneck_layer(self, inputs, in_filter, layer_name):

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

        W1_val = np.random.normal(loc=0, scale=0.1, size=(4 * self.n_filter, in_filter, 1, 1))  # kernel_size=1*1
        W2_val = np.random.normal(loc=0, scale=0.1, size=(self.n_filter, 4 * self.n_filter, 3, 3))  # kernel_size=3*3

        dict = {W1: W1_val, W2: W2_val}
        return drop2, dict, self.n_filter

    def dense_block(self, inputs, in_filter, nb_layers, block_name):

        x1 = inputs
        dict1 = {}
        for i in range(nb_layers):
            x2, dict2, out_filter = self.bottleneck_layer(x1, in_filter, block_name + "_bottleneck" + str(i))
            x1 = self.ad.concat_forward_op(x1, x2)
            in_filter = in_filter + out_filter
            dict1.update(dict2)
        return x1, dict1, in_filter

    def transition_layer(self, inputs, in_filter, layer_name):

        W1 = self.ad.Variable(layer_name + "_W1")
        bn1 = self.ad.bn_forward_op(inputs, "NCHW", "pre_activation")
        act1 = self.ad.activation_forward_op(bn1, "NCHW", "relu")
        conv1 = self.ad.convolution_2d_forward_op(act1, W1, "NCHW", "SAME", 1, 1)
        drop1 = self.ad.dropout_forward_op(conv1, "NCHW", self.dropout_rate)
        pool0 = self.ad.pooling_2d_forward_op(drop1, "NCHW", "mean", 0, 0, 2, 2, 2, 2)  # stride=2   pool_size=2*2

        W1_val = np.random.normal(loc=0, scale=0.1, size=(int(0.5 * in_filter), in_filter, 1, 1))  # kernel_size=1*1
        dict = {W1: W1_val}
        return pool0, dict, int(0.5 * in_filter)

    def dense_net(self, num_step, n_class, X_val, y_val):


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

        W0_val = np.random.normal(loc=0, scale=0.1, size=(2 * self.n_filter, self.image_channel, 7, 7))  # n_filter   n_channel=3   kernel_size=7*7
        W1_val = np.random.normal(loc=0, scale=0.1, size=(out_filter7, n_class))
        b1_val = np.random.normal(loc=0, scale=0.1, size=(n_class))

        feed_dict = {W0: W0_val, W1: W1_val, b1: b1_val}
        feed_dict.update(dict_1)
        feed_dict.update(dict_2)
        feed_dict.update(dict_3)
        feed_dict.update(dict_4)
        feed_dict.update(dict_5)
        feed_dict.update(dict_6)
        feed_dict.update(dict_7)

        aph = 0.001
        if self.is_capu == True and self.need_tosave != None:
            t = self.TrainExecute.TrainExecutor(loss, aph, self.need_tosave)
        else:
            t = self.TrainExecute.TrainExecutor(loss, aph)
        t.init_Variable(feed_dict)
        start_time = datetime.datetime.now()
        for i in range(num_step):
            time1 = datetime.datetime.now()

            t.run({X: X_val, y_: y_val})

            time2 = datetime.datetime.now()
            print("epoch", i + 1, "use", (time2 - time1).total_seconds()
                  , "\tstart", time1, "\tend", time2, file=self.f1)
            # print("DenseNet num_step", i)
        start_finish_time = datetime.datetime.now()
        print((start_finish_time-start_time).total_seconds(), file=self.f3)
        hit_count, swap_count = t.get_hit()
        print("hit_count ", hit_count, "\nswap_count", swap_count, file=self.f6)
        node_order = t.get_node_order()
        for i in node_order:
            print(i, file=self.f7)
        t.destroy_cudaStream()
        self.f1.close()
        self.f3.close()
        self.f6.close()
        self.f7.close()

    def run(self):
        try:
            X_val = np.random.normal(loc=0, scale=0.1, size=(self.batch_size, 3, 224, 224))  # number = batch_size  channel = 3  image_size = 224*224
            y_val = np.random.normal(loc=0, scale=0.1, size=(self.batch_size, 1000))  # n_class = 1000

            record = record_GPU.record("DenseNet", self.type, self.gpu_num, self.path, self.file_name)
            record.start()

            print("DenseNet" + " type" + str(self.type) + " start")

            self.dense_net(num_step=self.num_step, n_class=1000, X_val=X_val, y_val=y_val)

            print("DenseNet" + " type" + str(self.type) + " finish")

            record.stop()
        except Exception as e:
            traceback.print_exc()


