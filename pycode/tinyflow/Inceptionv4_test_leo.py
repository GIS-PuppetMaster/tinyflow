import os
GPU = 1
os.environ['CUDA_VISIBLE_DEVICES'] = f'{GPU}'
import sys
sys.path.append('../../')
from pycode.tinyflow import autodiff as ad
from pycode.tinyflow.get_result import get_result
from util import *


class Inceptionv4():
    def __init__(self, num_step, batch_size, log_path, job_id):
        self.dropout_rate = 0.5
        self.image_channel = 3
        self.image_size = 299
        self.num_step = num_step
        self.batch_size = batch_size
        self.job_id = job_id
        self.log_path = log_path
        self.executor_ctx = None
        self.n_class = None
        self.ad = ad
        self.top_control_queue = None
        self.top_message_queue = None

    def conv2dplusrelu(self, node, filter, model, type, stride_h, stride_w):
        node_new = self.ad.convolution_2d_forward_op(node, filter, model, type, stride_h, stride_w)
        node_after = self.ad.activation_forward_op(node_new, model, "relu")
        return node_after

    def block_inception_a(self, inputs, blockname, executor_ctx=None):
        filter1_1_0 = self.ad.Variable(blockname + "filter1_1_0")
        filter1_1_1a = self.ad.Variable(blockname + "filter1_1_1a")
        filter1_1_1b = self.ad.Variable(blockname + "filter1_1_1b")
        filter1_1_2a = self.ad.Variable(blockname + "filter1_1_2a")
        filter1_1_2b = self.ad.Variable(blockname + "filter1_1_2b")
        filter1_1_2c = self.ad.Variable(blockname + "filter1_1_2c")
        filter1_1_3 = self.ad.Variable(blockname + "filter1_1_3a")

        if executor_ctx is None:
            filter1_1_0_val = (96, 384, 1, 1)
            filter1_1_1_vala = (64, 384, 1, 1)
            filter1_1_1_valb = (96, 64, 3, 3)
            filter1_1_2_vala = (64, 384, 1, 1)
            filter1_1_2_valb = (96, 64, 3, 3)
            filter1_1_2_valc = (96, 96, 3, 3)
            filter1_1_3_val = (96, 384, 1, 1)
        else:
            filter1_1_0_val = ndarray.array(np.random.normal(scale=0.1, size=(96, 384, 1, 1)), executor_ctx)
            filter1_1_1_vala = ndarray.array(np.random.normal(scale=0.1, size=(64, 384, 1, 1)), executor_ctx)
            filter1_1_1_valb = ndarray.array(np.random.normal(scale=0.1, size=(96, 64, 3, 3)), executor_ctx)
            filter1_1_2_vala = ndarray.array(np.random.normal(scale=0.1, size=(64, 384, 1, 1)), executor_ctx)
            filter1_1_2_valb = ndarray.array(np.random.normal(scale=0.1, size=(96, 64, 3, 3)), executor_ctx)
            filter1_1_2_valc = ndarray.array(np.random.normal(scale=0.1, size=(96, 96, 3, 3)), executor_ctx)
            filter1_1_3_val = ndarray.array(np.random.normal(scale=0.1, size=(96, 384, 1, 1)), executor_ctx)

        # branch_0
        incep1_1_0 = self.conv2dplusrelu(inputs, filter1_1_0, "NCHW", "SAME", 1, 1)
        # branch 1
        incep1_1_1a = self.conv2dplusrelu(inputs, filter1_1_1a, "NCHW", "SAME", 1, 1)
        incep1_1_1 = self.conv2dplusrelu(incep1_1_1a, filter1_1_1b, "NCHW", "SAME", 1, 1)
        # branch 2
        incep1_1_2a = self.conv2dplusrelu(inputs, filter1_1_2a, "NCHW", "SAME", 1, 1)
        incep1_1_2b = self.conv2dplusrelu(incep1_1_2a, filter1_1_2b, "NCHW", "SAME", 1, 1)
        incep1_1_2 = self.conv2dplusrelu(incep1_1_2b, filter1_1_2c, "NCHW", "SAME", 1, 1)
        # branch 3
        incep1_1_3a = self.ad.pooling_2d_forward_op(inputs, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
        incep1_1_3 = self.conv2dplusrelu(incep1_1_3a, filter1_1_3, "NCHW", "SAME", 1, 1)

        concat1_1a = self.ad.concat_forward_op(incep1_1_0, incep1_1_1)
        concat1_1b = self.ad.concat_forward_op(concat1_1a, incep1_1_2)
        concat1_1 = self.ad.concat_forward_op(concat1_1b, incep1_1_3)

        dict = {filter1_1_0: filter1_1_0_val, filter1_1_1a: filter1_1_1_vala, filter1_1_1b: filter1_1_1_valb, filter1_1_2a: filter1_1_2_vala, filter1_1_2b: filter1_1_2_valb
            , filter1_1_2c: filter1_1_2_valc, filter1_1_3: filter1_1_3_val}
        return concat1_1, dict

    def block_reduction_a(self, inputs, input_size, blockname, executor_ctx=None):
        filter2_1_0 = self.ad.Variable(blockname + "filter2_1_0")
        filter2_1_1a = self.ad.Variable(blockname + "filter2_1_1a")
        filter2_1_1b = self.ad.Variable(blockname + "filter2_1_1b")
        filter2_1_1c = self.ad.Variable(blockname + "filter2_1_1c")
        if executor_ctx is None:
            filter2_1_0_val = (384, input_size, 3, 3)
            filter2_1_1_vala = (192, input_size, 1, 1)
            filter2_1_1_valb = (224, 192, 3, 3)
            filter2_1_1_valc = (256, 224, 3, 3)
        else:
            filter2_1_0_val = ndarray.array(np.random.normal(scale=0.1, size=(384, input_size, 3, 3)), executor_ctx)
            filter2_1_1_vala = ndarray.array(np.random.normal(scale=0.1, size=(192, input_size, 1, 1)), executor_ctx)
            filter2_1_1_valb = ndarray.array(np.random.normal(scale=0.1, size=(224, 192, 3, 3)), executor_ctx)
            filter2_1_1_valc = ndarray.array(np.random.normal(scale=0.1, size=(256, 224, 3, 3)), executor_ctx)

        # branch_0
        incep2_1_0 = self.conv2dplusrelu(inputs, filter2_1_0, "NCHW", "VALID", 2, 2)
        # branch 1
        incep2_1_1a = self.conv2dplusrelu(inputs, filter2_1_1a, "NCHW", "SAME", 1, 1)
        incep2_1_1b = self.conv2dplusrelu(incep2_1_1a, filter2_1_1b, "NCHW", "SAME", 1, 1)
        incep2_1_1 = self.conv2dplusrelu(incep2_1_1b, filter2_1_1c, "NCHW", "VALID", 2, 2)
        # branch 2
        incep2_1_2 = self.ad.pooling_2d_forward_op(inputs, "NCHW", "mean", 0, 0, 2, 2, 3, 3)

        concat2_1a = self.ad.concat_forward_op(incep2_1_0, incep2_1_1)
        concat2_1 = self.ad.concat_forward_op(concat2_1a, incep2_1_2)
        dict = {filter2_1_0: filter2_1_0_val, filter2_1_1a: filter2_1_1_vala, filter2_1_1b: filter2_1_1_valb, filter2_1_1c: filter2_1_1_valc}
        return concat2_1, dict

    def block_inception_b(self, inputs, input_size, blockname, executor_ctx=None):
        filter2_2_0 = self.ad.Variable(blockname + "filter2_2_0")
        filter2_2_1a = self.ad.Variable(blockname + "filter2_2_1a")
        filter2_2_1b = self.ad.Variable(blockname + "filter2_2_1b")
        filter2_2_1c = self.ad.Variable(blockname + "filter2_2_1c")
        filter2_2_2a = self.ad.Variable(blockname + "filter2_2_2a")
        filter2_2_2b = self.ad.Variable(blockname + "filter2_2_2b")
        filter2_2_2c = self.ad.Variable(blockname + "filter2_2_2c")
        filter2_2_2d = self.ad.Variable(blockname + "filter2_2_2d")
        filter2_2_2e = self.ad.Variable(blockname + "filter2_2_2e")
        filter2_2_3 = self.ad.Variable(blockname + "filter2_2_3a")
        if executor_ctx is None:
            filter2_2_0_val = (384, input_size, 1, 1)
            filter2_2_1_vala = (192, input_size, 1, 1)
            filter2_2_1_valb = (224, 192, 1, 7)
            filter2_2_1_valc = (256, 224, 7, 1)
            filter2_2_2_vala = (192, input_size, 1, 1)
            filter2_2_2_valb = (192, 192, 7, 1)
            filter2_2_2_valc = (224, 192, 1, 7)
            filter2_2_2_vald = (224, 224, 7, 1)
            filter2_2_2_vale = (256, 224, 1, 7)
            filter2_2_3_val = (128, input_size, 1, 1)
        else:
            filter2_2_0_val = ndarray.array(np.random.normal(scale=0.1, size=(384, input_size, 1, 1)), executor_ctx)
            filter2_2_1_vala = ndarray.array(np.random.normal(scale=0.1, size=(192, input_size, 1, 1)), executor_ctx)
            filter2_2_1_valb = ndarray.array(np.random.normal(scale=0.1, size=(224, 192, 1, 7)), executor_ctx)
            filter2_2_1_valc = ndarray.array(np.random.normal(scale=0.1, size=(256, 224, 7, 1)), executor_ctx)
            filter2_2_2_vala = ndarray.array(np.random.normal(scale=0.1, size=(192, input_size, 1, 1)), executor_ctx)
            filter2_2_2_valb = ndarray.array(np.random.normal(scale=0.1, size=(192, 192, 7, 1)), executor_ctx)
            filter2_2_2_valc = ndarray.array(np.random.normal(scale=0.1, size=(224, 192, 1, 7)), executor_ctx)
            filter2_2_2_vald = ndarray.array(np.random.normal(scale=0.1, size=(224, 224, 7, 1)), executor_ctx)
            filter2_2_2_vale = ndarray.array(np.random.normal(scale=0.1, size=(256, 224, 1, 7)), executor_ctx)
            filter2_2_3_val = ndarray.array(np.random.normal(scale=0.1, size=(128, input_size, 1, 1)), executor_ctx)
        # branch_0
        incep2_2_0 = self.conv2dplusrelu(inputs, filter2_2_0, "NCHW", "SAME", 1, 1)
        # branch 1
        incep2_2_1a = self.conv2dplusrelu(inputs, filter2_2_1a, "NCHW", "SAME", 1, 1)
        incep2_2_1b = self.conv2dplusrelu(incep2_2_1a, filter2_2_1b, "NCHW", "SAME", 1, 1)
        incep2_2_1 = self.conv2dplusrelu(incep2_2_1b, filter2_2_1c, "NCHW", "SAME", 1, 1)
        # branch 2
        incep2_2_2a = self.conv2dplusrelu(inputs, filter2_2_2a, "NCHW", "SAME", 1, 1)
        incep2_2_2b = self.conv2dplusrelu(incep2_2_2a, filter2_2_2b, "NCHW", "SAME", 1, 1)
        incep2_2_2c = self.conv2dplusrelu(incep2_2_2b, filter2_2_2c, "NCHW", "SAME", 1, 1)
        incep2_2_2d = self.conv2dplusrelu(incep2_2_2c, filter2_2_2d, "NCHW", "SAME", 1, 1)
        incep2_2_2 = self.conv2dplusrelu(incep2_2_2d, filter2_2_2e, "NCHW", "SAME", 1, 1)
        # branch 3
        incep2_2_3a = self.ad.pooling_2d_forward_op(inputs, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
        incep2_2_3 = self.conv2dplusrelu(incep2_2_3a, filter2_2_3, "NCHW", "SAME", 1, 1)

        concat2_2a = self.ad.concat_forward_op(incep2_2_0, incep2_2_1)
        concat2_2b = self.ad.concat_forward_op(concat2_2a, incep2_2_2)
        concat2_2 = self.ad.concat_forward_op(concat2_2b, incep2_2_3)
        dict = {filter2_2_0: filter2_2_0_val, filter2_2_1a: filter2_2_1_vala, filter2_2_1b: filter2_2_1_valb, filter2_2_1c: filter2_2_1_valc,
                filter2_2_2a: filter2_2_2_vala, filter2_2_2b: filter2_2_2_valb, filter2_2_2c: filter2_2_2_valc, filter2_2_2d: filter2_2_2_vald, filter2_2_2e: filter2_2_2_vale, filter2_2_3: filter2_2_3_val
                }
        return concat2_2, dict

    def block_reduction_b(self, inputs, input_size, blockname, executor_ctx=None):
        filter3_1_0a = self.ad.Variable(blockname + "filter3_1_0a")
        filter3_1_0b = self.ad.Variable(blockname + "filter3_1_0b")
        filter3_1_1a = self.ad.Variable(blockname + "filter3_1_1a")
        filter3_1_1b = self.ad.Variable(blockname + "filter3_1_1b")
        filter3_1_1c = self.ad.Variable(blockname + "filter3_1_1c")
        filter3_1_1d = self.ad.Variable(blockname + "filter3_1_1d")

        if executor_ctx is None:
            filter3_1_0_vala = (192, input_size, 1, 1)
            filter3_1_0_valb = (192, 192, 3, 3)
            filter3_1_1_vala = (256, input_size, 1, 1)
            filter3_1_1_valb = (256, 256, 1, 7)
            filter3_1_1_valc = (320, 256, 7, 1)
            filter3_1_1_vald = (320, 320, 3, 3)
        else:
            filter3_1_0_vala = ndarray.array(np.random.normal(scale=0.1, size=(192, input_size, 1, 1)), executor_ctx)
            filter3_1_0_valb = ndarray.array(np.random.normal(scale=0.1, size=(192, 192, 3, 3)), executor_ctx)
            filter3_1_1_vala = ndarray.array(np.random.normal(scale=0.1, size=(256, input_size, 1, 1)), executor_ctx)
            filter3_1_1_valb = ndarray.array(np.random.normal(scale=0.1, size=(256, 256, 1, 7)), executor_ctx)
            filter3_1_1_valc = ndarray.array(np.random.normal(scale=0.1, size=(320, 256, 7, 1)), executor_ctx)
            filter3_1_1_vald = ndarray.array(np.random.normal(scale=0.1, size=(320, 320, 3, 3)), executor_ctx)

        # branch_0
        incep3_1_0a = self.conv2dplusrelu(inputs, filter3_1_0a, "NCHW", "SAME", 1, 1)
        incep3_1_0 = self.conv2dplusrelu(incep3_1_0a, filter3_1_0b, "NCHW", "VALID", 2, 2)
        # branch 1
        incep3_1_1a = self.conv2dplusrelu(inputs, filter3_1_1a, "NCHW", "SAME", 1, 1)
        incep3_1_1b = self.conv2dplusrelu(incep3_1_1a, filter3_1_1b, "NCHW", "SAME", 1, 1)
        incep3_1_1c = self.conv2dplusrelu(incep3_1_1b, filter3_1_1c, "NCHW", "SAME", 1, 1)
        incep3_1_1 = self.conv2dplusrelu(incep3_1_1c, filter3_1_1d, "NCHW", "VALID", 2, 2)
        # branch 2
        incep3_1_2 = self.ad.pooling_2d_forward_op(inputs, "NCHW", "mean", 0, 0, 2, 2, 3, 3)

        concat3_1a = self.ad.concat_forward_op(incep3_1_0, incep3_1_1)
        concat3_1 = self.ad.concat_forward_op(concat3_1a, incep3_1_2)
        dict = {filter3_1_0a: filter3_1_0_vala, filter3_1_0b: filter3_1_0_valb, filter3_1_1a: filter3_1_1_vala, filter3_1_1b: filter3_1_1_valb,
                filter3_1_1c: filter3_1_1_valc, filter3_1_1d: filter3_1_1_vald}
        return concat3_1, dict

    def block_inception_c(self, inputs, input_size, blockname, executor_ctx=None):
        filter3_2_0 = self.ad.Variable(blockname + "filter3_2_0")
        filter3_2_1a = self.ad.Variable(blockname + "filter3_2_1a")
        filter3_2_1b = self.ad.Variable(blockname + "filter3_2_1b")
        filter3_2_1c = self.ad.Variable(blockname + "filter3_2_1c")
        filter3_2_2a = self.ad.Variable(blockname + "filter3_2_2a")
        filter3_2_2b = self.ad.Variable(blockname + "filter3_2_2b")
        filter3_2_2c = self.ad.Variable(blockname + "filter3_2_2c")
        filter3_2_2d = self.ad.Variable(blockname + "filter3_2_2d")
        filter3_2_2e = self.ad.Variable(blockname + "filter3_2_2e")
        filter3_2_3 = self.ad.Variable(blockname + "filter3_2_3a")
        if executor_ctx is None:
            filter3_2_0_val = (256, input_size, 1, 1)
            filter3_2_1_vala = (384, input_size, 1, 1)
            filter3_2_1_valb = (256, 384, 1, 3)
            filter3_2_1_valc = (256, 384, 3, 1)
            filter3_2_2_vala = (384, input_size, 1, 1)
            filter3_2_2_valb = (448, 384, 1, 3)
            filter3_2_2_valc = (512, 448, 3, 1)
            filter3_2_2_vald = (256, 512, 3, 1)
            filter3_2_2_vale = (256, 512, 1, 3)
            filter3_2_3_val = (256, input_size, 1, 1)
        else:
            filter3_2_0_val = ndarray.array(np.random.normal(scale=0.1, size=(256, input_size, 1, 1)), executor_ctx)
            filter3_2_1_vala = ndarray.array(np.random.normal(scale=0.1, size=(384, input_size, 1, 1)), executor_ctx)
            filter3_2_1_valb = ndarray.array(np.random.normal(scale=0.1, size=(256, 384, 1, 3)), executor_ctx)
            filter3_2_1_valc = ndarray.array(np.random.normal(scale=0.1, size=(256, 384, 3, 1)), executor_ctx)
            filter3_2_2_vala = ndarray.array(np.random.normal(scale=0.1, size=(384, input_size, 1, 1)), executor_ctx)
            filter3_2_2_valb = ndarray.array(np.random.normal(scale=0.1, size=(448, 384, 1, 3)), executor_ctx)
            filter3_2_2_valc = ndarray.array(np.random.normal(scale=0.1, size=(512, 448, 3, 1)), executor_ctx)
            filter3_2_2_vald = ndarray.array(np.random.normal(scale=0.1, size=(256, 512, 3, 1)), executor_ctx)
            filter3_2_2_vale = ndarray.array(np.random.normal(scale=0.1, size=(256, 512, 1, 3)), executor_ctx)
            filter3_2_3_val = ndarray.array(np.random.normal(scale=0.1, size=(256, input_size, 1, 1)), executor_ctx)

        # branch_0
        incep3_2_0 = self.conv2dplusrelu(inputs, filter3_2_0, "NCHW", "SAME", 1, 1)
        # branch 1
        incep3_2_1a = self.conv2dplusrelu(inputs, filter3_2_1a, "NCHW", "SAME", 1, 1)
        incep3_2_1b = self.conv2dplusrelu(incep3_2_1a, filter3_2_1b, "NCHW", "SAME", 1, 1)
        incep3_2_1c = self.conv2dplusrelu(incep3_2_1a, filter3_2_1c, "NCHW", "SAME", 1, 1)
        incep3_2_1 = self.ad.concat_forward_op(incep3_2_1b, incep3_2_1c)
        # branch 2
        incep3_2_2a = self.conv2dplusrelu(inputs, filter3_2_2a, "NCHW", "SAME", 1, 1)
        incep3_2_2b = self.conv2dplusrelu(incep3_2_2a, filter3_2_2b, "NCHW", "SAME", 1, 1)
        incep3_2_2c = self.conv2dplusrelu(incep3_2_2b, filter3_2_2c, "NCHW", "SAME", 1, 1)
        incep3_2_2d = self.conv2dplusrelu(incep3_2_2c, filter3_2_2d, "NCHW", "SAME", 1, 1)
        incep3_2_2e = self.conv2dplusrelu(incep3_2_2c, filter3_2_2e, "NCHW", "SAME", 1, 1)
        incep3_2_2 = self.ad.concat_forward_op(incep3_2_2d, incep3_2_2e)
        # branch 3
        incep3_2_3a = self.ad.pooling_2d_forward_op(inputs, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
        incep3_2_3 = self.conv2dplusrelu(incep3_2_3a, filter3_2_3, "NCHW", "SAME", 1, 1)

        concat3_2a = self.ad.concat_forward_op(incep3_2_0, incep3_2_1)
        concat3_2b = self.ad.concat_forward_op(concat3_2a, incep3_2_2)
        concat3_2 = self.ad.concat_forward_op(concat3_2b, incep3_2_3)
        dict = {filter3_2_0: filter3_2_0_val, filter3_2_1a: filter3_2_1_vala,
                filter3_2_1b: filter3_2_1_valb,
                filter3_2_1c: filter3_2_1_valc, filter3_2_2a: filter3_2_2_vala, filter3_2_2b: filter3_2_2_valb,
                filter3_2_2c: filter3_2_2_valc, filter3_2_2d: filter3_2_2_vald, filter3_2_2e: filter3_2_2_vale, filter3_2_3: filter3_2_3_val}
        return concat3_2, dict

    def get_predict_results(self, n_class, **kwargs):
        X = self.ad.Placeholder("X")
        y_ = self.ad.Placeholder("y_")
        f1 = self.ad.Variable("f1")
        f2 = self.ad.Variable("f2")
        f3 = self.ad.Variable("f3")
        f4 = self.ad.Variable("f4")
        f5_1 = self.ad.Variable("f5_1")
        f5_2 = self.ad.Variable("f5_2")
        f6_1 = self.ad.Variable("f6_1")
        f6_2 = self.ad.Variable("f6_2")
        f6_3 = self.ad.Variable("f6_3")
        f6_4 = self.ad.Variable("f6_4")
        f7 = self.ad.Variable("f7")
        W = self.ad.Variable("W")
        b = self.ad.Variable("b")

        f1val = (32, 3, 3, 3)
        f2val = (32, 32, 3, 3)
        f3val = (64, 32, 3, 3)
        f4val = (96, 64, 3, 3)
        f5_1val = (64, 160, 1, 1)
        f5_2val = (96, 64, 3, 3)
        f6_1val = (64, 160, 1, 1)
        f6_2val = (64, 64, 7, 1)
        f6_3val = (64, 64, 1, 7)
        f6_4val = (96, 64, 3, 3)
        f7val = (192, 192, 3, 3)
        W_val = (1536, n_class)
        b_val = (n_class, )
        # stem
        cov1 = self.conv2dplusrelu(X, f1, "NCHW", "VALID", 2, 2)
        cov2 = self.conv2dplusrelu(cov1, f2, "NCHW", "VALID", 1, 1)
        cov3 = self.conv2dplusrelu(cov2, f3, "NCHW", "SAME", 1, 1)
        pool4 = self.ad.pooling_2d_forward_op(cov3, "NCHW", "max", 0, 0, 2, 2, 3, 3)
        cov4 = self.conv2dplusrelu(cov3, f4, "NCHW", "VALID", 2, 2)
        concat1 = self.ad.concat_forward_op(pool4, cov4)
        cov5_1 = self.conv2dplusrelu(concat1, f5_1, "NCHW", "SAME", 1, 1)
        cov5_2 = self.conv2dplusrelu(cov5_1, f5_2, "NCHW", "VALID", 1, 1)
        cov6_1 = self.conv2dplusrelu(concat1, f6_1, "NCHW", "SAME", 1, 1)
        cov6_2 = self.conv2dplusrelu(cov6_1, f6_2, "NCHW", "SAME", 1, 1)
        cov6_3 = self.conv2dplusrelu(cov6_2, f6_3, "NCHW", "SAME", 1, 1)
        cov6_4 = self.conv2dplusrelu(cov6_3, f6_4, "NCHW", "VALID", 1, 1)
        concat2 = self.ad.concat_forward_op(cov5_2, cov6_4)
        cov7 = self.conv2dplusrelu(concat2, f7, "NCHW", "VALID", 2, 2)
        pool7 = self.ad.pooling_2d_forward_op(concat2, "NCHW", "max", 0, 0, 2, 2, 3, 3)
        concat3 = self.ad.concat_forward_op(pool7, cov7)
        a1, dicta1 = self.block_inception_a(concat3, "a1")
        a2, dicta2 = self.block_inception_a(a1, "a2")
        a3, dicta3 = self.block_inception_a(a2, "a3")
        a4, dicta4 = self.block_inception_a(a3, "a4")

        ra, dictra = self.block_reduction_a(a4, 384, "ra")
        b1, dictb1 = self.block_inception_b(ra, 1024, "b1")
        b2, dictb2 = self.block_inception_b(b1, 1024, "b2")
        b3, dictb3 = self.block_inception_b(b2, 1024, "b3")
        b4, dictb4 = self.block_inception_b(b3, 1024, "b4")
        b5, dictb5 = self.block_inception_b(b4, 1024, "b5")
        b6, dictb6 = self.block_inception_b(b5, 1024, "b6")
        b7, dictb7 = self.block_inception_b(b6, 1024, "b7")
        #
        rb, dictrb = self.block_reduction_b(b7, 1024, "rb")
        c1, dictc1 = self.block_inception_c(rb, 1536, "c1")
        c2, dictc2 = self.block_inception_c(c1, 1536, "c2")
        c3, dictc3 = self.block_inception_c(c2, 1536, "c3")
        poollast = self.ad.pooling_2d_forward_op(c3, "NCHW", "mean", 0, 0, 1, 1, 8, 8)
        squeeze = self.ad.squeeze_op(poollast)
        drop_out = self.ad.fullydropout_forward_op(squeeze, "NCHW", 0.8)
        dense = self.ad.dense(drop_out, W, b)
        y = self.ad.fullyactivation_forward_op(dense, "NCHW", "softmax")
        loss = self.ad.crossEntropy_loss(y, y_)
        executor = self.ad.Executor(loss, y, 0.001, None, None, log_path=self.log_path, **kwargs)

        feed_dict = {f1: f1val, f2: f2val, f3: f3val, f4: f4val, f5_1: f5_1val, f5_2: f5_2val, f6_1: f6_1val, f6_2: f6_2val, f6_3: f6_3val, f6_4: f6_4val, f7: f7val, W: W_val, b: b_val}
        feed_dict.update(dicta1)
        feed_dict.update(dicta2)
        feed_dict.update(dicta3)
        feed_dict.update(dicta4)
        feed_dict.update(dictra)
        feed_dict.update(dictb1)
        feed_dict.update(dictb2)
        feed_dict.update(dictb3)
        feed_dict.update(dictb4)
        feed_dict.update(dictb5)
        feed_dict.update(dictb6)
        feed_dict.update(dictb7)
        feed_dict.update(dictrb)
        feed_dict.update(dictc1)
        feed_dict.update(dictc2)
        feed_dict.update(dictc3)

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
        f1 = self.ad.Variable("f1")
        f2 = self.ad.Variable("f2")
        f3 = self.ad.Variable("f3")
        f4 = self.ad.Variable("f4")
        f5_1 = self.ad.Variable("f5_1")
        f5_2 = self.ad.Variable("f5_2")
        f6_1 = self.ad.Variable("f6_1")
        f6_2 = self.ad.Variable("f6_2")
        f6_3 = self.ad.Variable("f6_3")
        f6_4 = self.ad.Variable("f6_4")
        f7 = self.ad.Variable("f7")
        W = self.ad.Variable("W")
        b = self.ad.Variable("b")

        f1val = ndarray.array(np.random.normal(0, 0.5, (32, 3, 3, 3)), executor_ctx)
        f2val = ndarray.array(np.random.normal(0, 0.5, (32, 32, 3, 3)), executor_ctx)
        f3val = ndarray.array(np.random.normal(0, 0.5, (64, 32, 3, 3)), executor_ctx)
        f4val = ndarray.array(np.random.normal(0, 0.5, (96, 64, 3, 3)), executor_ctx)
        f5_1val = ndarray.array(np.random.normal(0, 0.5, (64, 160, 1, 1)), executor_ctx)
        f5_2val = ndarray.array(np.random.normal(0, 0.5, (96, 64, 3, 3)), executor_ctx)
        f6_1val = ndarray.array(np.random.normal(0, 0.5, (64, 160, 1, 1)), executor_ctx)
        f6_2val = ndarray.array(np.random.normal(0, 0.5, (64, 64, 7, 1)), executor_ctx)
        f6_3val = ndarray.array(np.random.normal(0, 0.5, (64, 64, 1, 7)), executor_ctx)
        f6_4val = ndarray.array(np.random.normal(0, 0.5, (96, 64, 3, 3)), executor_ctx)
        f7val = ndarray.array(np.random.normal(0, 0.5, (192, 192, 3, 3)), executor_ctx)
        W_val = ndarray.array(np.random.normal(0, 0.5, (1536, n_class)), executor_ctx)
        b_val = ndarray.array(np.random.normal(0, 0.5, (n_class)), executor_ctx)
        # stem
        cov1 = self.conv2dplusrelu(X, f1, "NCHW", "VALID", 2, 2)
        cov2 = self.conv2dplusrelu(cov1, f2, "NCHW", "VALID", 1, 1)
        cov3 = self.conv2dplusrelu(cov2, f3, "NCHW", "SAME", 1, 1)
        pool4 = self.ad.pooling_2d_forward_op(cov3, "NCHW", "max", 0, 0, 2, 2, 3, 3)
        cov4 = self.conv2dplusrelu(cov3, f4, "NCHW", "VALID", 2, 2)
        concat1 = self.ad.concat_forward_op(pool4, cov4)
        cov5_1 = self.conv2dplusrelu(concat1, f5_1, "NCHW", "SAME", 1, 1)
        cov5_2 = self.conv2dplusrelu(cov5_1, f5_2, "NCHW", "VALID", 1, 1)
        cov6_1 = self.conv2dplusrelu(concat1, f6_1, "NCHW", "SAME", 1, 1)
        cov6_2 = self.conv2dplusrelu(cov6_1, f6_2, "NCHW", "SAME", 1, 1)
        cov6_3 = self.conv2dplusrelu(cov6_2, f6_3, "NCHW", "SAME", 1, 1)
        cov6_4 = self.conv2dplusrelu(cov6_3, f6_4, "NCHW", "VALID", 1, 1)
        concat2 = self.ad.concat_forward_op(cov5_2, cov6_4)
        cov7 = self.conv2dplusrelu(concat2, f7, "NCHW", "VALID", 2, 2)
        pool7 = self.ad.pooling_2d_forward_op(concat2, "NCHW", "max", 0, 0, 2, 2, 3, 3)
        concat3 = self.ad.concat_forward_op(pool7, cov7)
        a1, dicta1 = self.block_inception_a(concat3, "a1", executor_ctx)
        a2, dicta2 = self.block_inception_a(a1, "a2", executor_ctx)
        a3, dicta3 = self.block_inception_a(a2, "a3", executor_ctx)
        a4, dicta4 = self.block_inception_a(a3, "a4", executor_ctx)

        ra, dictra = self.block_reduction_a(a4, 384, "ra", executor_ctx)
        b1, dictb1 = self.block_inception_b(ra, 1024, "b1", executor_ctx)
        b2, dictb2 = self.block_inception_b(b1, 1024, "b2", executor_ctx)
        b3, dictb3 = self.block_inception_b(b2, 1024, "b3", executor_ctx)
        b4, dictb4 = self.block_inception_b(b3, 1024, "b4", executor_ctx)
        b5, dictb5 = self.block_inception_b(b4, 1024, "b5", executor_ctx)
        b6, dictb6 = self.block_inception_b(b5, 1024, "b6", executor_ctx)
        b7, dictb7 = self.block_inception_b(b6, 1024, "b7", executor_ctx)
        #
        rb, dictrb = self.block_reduction_b(b7, 1024, "rb", executor_ctx)
        c1, dictc1 = self.block_inception_c(rb, 1536, "c1", executor_ctx)
        c2, dictc2 = self.block_inception_c(c1, 1536, "c2", executor_ctx)
        c3, dictc3 = self.block_inception_c(c2, 1536, "c3", executor_ctx)
        poollast = self.ad.pooling_2d_forward_op(c3, "NCHW", "mean", 0, 0, 1, 1, 8, 8)
        squeeze = self.ad.squeeze_op(poollast)
        drop_out = self.ad.fullydropout_forward_op(squeeze, "NCHW", 0.8)
        dense = self.ad.dense(drop_out, W, b)
        y = self.ad.fullyactivation_forward_op(dense, "NCHW", "softmax")
        loss = self.ad.crossEntropy_loss(y, y_)
        executor = self.ad.Executor(loss, y, 0.001, top_control_queue=top_control_queue,
                                         top_message_queue=top_message_queue, log_path=self.log_path, **kwargs)

        feed_dict = {f1: f1val, f2: f2val, f3: f3val, f4: f4val, f5_1: f5_1val, f5_2: f5_2val, f6_1: f6_1val, f6_2: f6_2val, f6_3: f6_3val, f6_4: f6_4val, f7: f7val, W: W_val, b: b_val}
        feed_dict.update(dicta1)
        feed_dict.update(dicta2)
        feed_dict.update(dicta3)
        feed_dict.update(dicta4)
        feed_dict.update(dictra)
        feed_dict.update(dictb1)
        feed_dict.update(dictb2)
        feed_dict.update(dictb3)
        feed_dict.update(dictb4)
        feed_dict.update(dictb5)
        feed_dict.update(dictb6)
        feed_dict.update(dictb7)
        feed_dict.update(dictrb)
        feed_dict.update(dictc1)
        feed_dict.update(dictc2)
        feed_dict.update(dictc3)

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
        start_time = datetime.datetime.now()
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
                main(path, repeat, jobs_num, batch_size, Inceptionv4, **kwargs)
            elif skip != 'vanilla':
                path = raw_path + 'vanilla'
                print(path)
                main(path, repeat, jobs_num, batch_size, Inceptionv4, **kwargs)
        if analysis_result:
            get_result(raw_path, repeat)


if __name__ == '__main__':
    run_exp([['./log/Inception V4/', 3, 1, 16], ['./log/Inception V4 x1/', 3, 1, 2], ['./log/Inception V4 x2/', 3, 2, 2], ['./log/Inception V4 x3/', 3, 3, 2]])
