import numpy as np
import imp, datetime
from multiprocessing import Process

from pycode.tinyflow import ndarray
from tests.Experiment import record_GPU
import os

tinyflow_path = "../../pycode/tinyflow/"


class ResNet50(Process):
    def __init__(self, num_step, type, batch_size, gpu_num, path, file_name, need_tosave=None):
        super().__init__()
        self.need_tosave = need_tosave
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
        self.gpu_num = gpu_num
        self.dropout_rate = 0.5
        self.image_channel = 3
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

    def identity_block(self, inputs, kernel_size, in_filter, out_filters, block_name):

        f1, f2, f3 = out_filters

        W1 = self.ad.Variable(block_name + "W1")
        W2 = self.ad.Variable(block_name + "W2")
        W3 = self.ad.Variable(block_name + "W3")
        W1_val = np.random.normal(loc=0, scale=0.1, size=(f1, in_filter, 1, 1))
        W2_val = np.random.normal(loc=0, scale=0.1, size=(f2, f1, kernel_size, kernel_size))
        W3_val = np.random.normal(loc=0, scale=0.1, size=(f3, f2, 1, 1))

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

    def convolutional_block(self, inputs, kernel_size, in_filter, out_filters, block_name, stride):
        f1, f2, f3 = out_filters

        W1 = self.ad.Variable(block_name + "W1")
        W2 = self.ad.Variable(block_name + "W2")
        W3 = self.ad.Variable(block_name + "W3")
        W_shortcut = self.ad.Variable(block_name + "W_shortcut")
        W1_val = np.random.normal(loc=0, scale=0.1, size=(f1, in_filter, 1, 1))
        W2_val = np.random.normal(loc=0, scale=0.1, size=(f2, f1, kernel_size, kernel_size))
        W3_val = np.random.normal(loc=0, scale=0.1, size=(f3, f2, 1, 1))
        W_shortcut_val = np.random.normal(loc=0, scale=0.1, size=(f3, in_filter, 1, 1))

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

    def res_net(self, num_step, n_class, X_val, y_val):


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

        W1_val = np.random.normal(loc=0, scale=0.1, size=(64, self.image_channel, 7, 7))
        W6_val = np.random.normal(loc=0, scale=0.1, size=(1 * 1 * 2048, 2048))
        b6_val = np.random.normal(loc=0, scale=0.1, size=(2048))
        W7_val = np.random.normal(loc=0, scale=0.1, size=(2048, n_class))
        b7_val = np.random.normal(loc=0, scale=0.1, size=(n_class))

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
            print("ResNet num_step", i)
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
        # if self.need_tosave != 0:
        #     outspace = []
        #     arr_size = self.need_tosave * 1e6 / 4
        #     gctx = ndarray.gpu(0)
        #     while arr_size > 0:
        #         if arr_size > 10000 * 10000:
        #             outspace.append(ndarray.array(np.ones((10000, 10000)) * 0.01, ctx=gctx))
        #             arr_size -= 10000 * 10000
        #         else:
        #             need_sqrt = int(pow(arr_size, 0.5))
        #             if need_sqrt <= 0:
        #                 break
        #             outspace.append(ndarray.array(np.ones((need_sqrt, need_sqrt)) * 0.01, ctx=gctx))
        #             arr_size -= need_sqrt * need_sqrt
        X_val = np.random.normal(loc=0, scale=0.1, size=(self.batch_size, 3, 224, 224))  # number = batch_size  channel = 3  image_size = 224*224
        y_val = np.random.normal(loc=0, scale=0.1, size=(self.batch_size, 1000))  # n_class = 1000

        record = record_GPU.record("ResNet50", self.type, self.gpu_num, self.path, self.file_name)
        record.start()

        print("ResNet50" + " type" + str(self.type) + " start")
        self.res_net(num_step=self.num_step, n_class=1000, X_val=X_val, y_val=y_val)
        print("ResNet50" + " type" + str(self.type) + " finish")

        record.stop()
        # if self.need_tosave != 0:
        #     for i in range(len(outspace) - 1, -1, -1):
        #         outspace.pop(i)
# resNet = ResNet50(num_step=10, type=2, batch_size=4, gpu_num=0, file_name="")
# resNet.start()
