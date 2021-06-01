import numpy as np
from pycode.tinyflow import mainV2 as mp
from pycode.tinyflow import autodiff as ad
from pycode.tinyflow import ndarray
import threading, pynvml, multiprocessing, os, datetime, time
from multiprocessing import Process
from util import *

with open('./log_path.txt', 'r') as f:
    log_path = f.readlines()[0]
    if not os.path.exists(log_path):
        os.makedirs(log_path)
GPU = load_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = f'{GPU}'
class Inceptionv3():

    def __init__(self, num_step, batch_size, gpu_num):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
            self.gpu_num = gpu_num

            self.dropout_rate = 0.5
            self.image_channel = 3
            self.image_size = 299
            self.num_step = num_step
            self.batch_size = batch_size

            self.ad = ad


    def conv2dplusrelu(self,node,filter,model,type,stride_h,stride_w):
        node_new=self.ad.convolution_2d_forward_op(node, filter, model, type, stride_h,stride_w)
        node_after=self.ad.activation_forward_op(node_new, model, "relu")
        return  node_after

    def inception_v3(self, executor_ctx, top_control_queue, top_message_queue, n_class, X_val, y_val):
        gpu_record = GPURecord(log_path)
        start_time = datetime.datetime.now()
        X = self.ad.Placeholder("inputs")
        y_ = self.ad.Placeholder("y_")
        filterb_1 = self.ad.Variable("filterb_1")
        filterb_2 = self.ad.Variable("filterb_2")
        filterb_3 = self.ad.Variable("filterb_3")
        filterb_4 = self.ad.Variable("filterb_4")
        filterb_5= self.ad.Variable("filterb_5")

        filtersb_val1 = ndarray.array(np.random.normal(0, 0.5, (32,3,3,3)) , executor_ctx)
        filtersb_val2 = ndarray.array(np.random.normal(0, 0.5, (32,32,3,3)), executor_ctx)
        filtersb_val3 = ndarray.array(np.random.normal(0, 0.5, (64,32,3,3)) , executor_ctx)
        filtersb_val4 = ndarray.array(np.random.normal(0, 0.5, (80, 64, 1, 1)) , executor_ctx)
        filtersb_val5 = ndarray.array(np.random.normal(0, 0.5, (192, 80, 3, 3)) , executor_ctx)

    #inception前
        covb_1=self.conv2dplusrelu(X, filterb_1, "NCHW", "VALID", 2, 2)
        covb_2=self.conv2dplusrelu(covb_1, filterb_2, "NCHW", "VALID", 1, 1)
        covb_3= self.conv2dplusrelu(covb_2, filterb_3, "NCHW", "SAME", 1, 1)
        poolb= self.ad.pooling_2d_forward_op(covb_3,"NCHW", "max", 0, 0, 2, 2, 3, 3)
        covb_4 = self.conv2dplusrelu(poolb, filterb_4, "NCHW", "VALID", 1, 1)
        covb_5 = self.conv2dplusrelu(covb_4, filterb_5, "NCHW", "VALID", 1, 1)
        covb = self.ad.pooling_2d_forward_op(covb_5,"NCHW", "max", 0, 0, 2, 2, 3, 3)



    # inception_moudle1
     #inception_moudle1_1
        filter1_1_0= self.ad.Variable("filter1_1_0")
        filter1_1_1a = self.ad.Variable("filter1_1_1a")
        filter1_1_1b = self.ad.Variable("filter1_1_1b")
        filter1_1_2a = self.ad.Variable("filter1_1_2a")
        filter1_1_2b = self.ad.Variable("filter1_1_2b")
        filter1_1_2c = self.ad.Variable("filter1_1_2c")
        filter1_1_3 = self.ad.Variable("filter1_1_3a")

        filter1_1_0_val = ndarray.array(np.random.normal(0, 0.5, (64, 192, 1, 1)) , executor_ctx)
        filter1_1_1_vala = ndarray.array(np.random.normal(0, 0.5, (48, 192, 1, 1)) , executor_ctx)
        filter1_1_1_valb = ndarray.array(np.random.normal(0, 0.5, (64, 48, 5, 5)) , executor_ctx)
        filter1_1_2_vala = ndarray.array(np.random.normal(0, 0.5, (64, 192, 1, 1)) , executor_ctx)
        filter1_1_2_valb = ndarray.array(np.random.normal(0, 0.5, (96, 64, 3, 3)) , executor_ctx)
        filter1_1_2_valc = ndarray.array(np.random.normal(0, 0.5, (96, 96, 3, 3)) , executor_ctx)
        filter1_1_3_val = ndarray.array(np.random.normal(0, 0.5, (32, 192, 1, 1)) , executor_ctx)

        # branch_0
        incep1_1_0=self.conv2dplusrelu(covb, filter1_1_0, "NCHW", "SAME", 1, 1)
        # branch 1
        incep1_1_1a=self.conv2dplusrelu(covb, filter1_1_1a, "NCHW", "SAME", 1, 1)
        incep1_1_1 = self.conv2dplusrelu(incep1_1_1a, filter1_1_1b, "NCHW", "SAME", 1, 1)
        # branch 2
        incep1_1_2a = self.conv2dplusrelu(covb, filter1_1_2a, "NCHW", "SAME", 1, 1)
        incep1_1_2b = self.conv2dplusrelu(incep1_1_2a, filter1_1_2b, "NCHW", "SAME", 1, 1)
        incep1_1_2 = self.conv2dplusrelu(incep1_1_2b, filter1_1_2c, "NCHW", "SAME", 1, 1)
        # branch 3
        incep1_1_3a = self.ad.pooling_2d_forward_op(covb,"NCHW", "mean", 1, 1, 1, 1, 3, 3)
        incep1_1_3 = self.conv2dplusrelu(incep1_1_3a, filter1_1_3, "NCHW", "SAME", 1, 1)

        concat1_1a=self.ad.concat_forward_op(incep1_1_0,incep1_1_1)
        concat1_1b = self.ad.concat_forward_op(concat1_1a, incep1_1_2)
        concat1_1 = self.ad.concat_forward_op(concat1_1b, incep1_1_3)

     #inception_moudle1_2
        filter1_2_0 = self.ad.Variable("filter1_2_0")
        filter1_2_1a = self.ad.Variable("filter1_2_1a")
        filter1_2_1b = self.ad.Variable("filter1_2_1b")
        filter1_2_2a = self.ad.Variable("filter1_2_2a")
        filter1_2_2b = self.ad.Variable("filter1_2_2b")
        filter1_2_2c = self.ad.Variable("filter1_2_2c")
        filter1_2_3 = self.ad.Variable("filter1_2_3a")

        filter1_2_0_val = ndarray.array(np.random.normal(0, 0.5, (64, 256, 1, 1)) , executor_ctx)
        filter1_2_1_vala = ndarray.array(np.random.normal(0, 0.5, (48, 256, 1, 1)) , executor_ctx)
        filter1_2_1_valb = ndarray.array(np.random.normal(0, 0.5, (64, 48, 5, 5)) , executor_ctx)
        filter1_2_2_vala = ndarray.array(np.random.normal(0, 0.5, (64, 256, 1, 1)) , executor_ctx)
        filter1_2_2_valb = ndarray.array(np.random.normal(0, 0.5, (96, 64, 3, 3)) , executor_ctx)
        filter1_2_2_valc = ndarray.array(np.random.normal(0, 0.5, (96, 96, 3, 3)) , executor_ctx)
        filter1_2_3_val = ndarray.array(np.random.normal(0, 0.5, (64, 256, 1, 1)) , executor_ctx)

        # branch_0
        incep1_2_0 = self.conv2dplusrelu(concat1_1, filter1_2_0, "NCHW", "SAME", 1, 1)
        # branch 1
        incep1_2_1a = self.conv2dplusrelu(concat1_1, filter1_2_1a, "NCHW", "SAME", 1, 1)
        incep1_2_1 = self.conv2dplusrelu(incep1_2_1a, filter1_2_1b, "NCHW", "SAME", 1, 1)
        # branch 2
        incep1_2_2a = self.conv2dplusrelu(concat1_1, filter1_2_2a, "NCHW", "SAME", 1, 1)
        incep1_2_2b = self.conv2dplusrelu(incep1_2_2a, filter1_2_2b, "NCHW", "SAME", 1, 1)
        incep1_2_2 = self.conv2dplusrelu(incep1_2_2b, filter1_2_2c, "NCHW", "SAME", 1, 1)
        # branch 3
        incep1_2_3a = self.ad.pooling_2d_forward_op(concat1_1, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
        incep1_2_3 = self.conv2dplusrelu(incep1_2_3a, filter1_2_3, "NCHW", "SAME", 1, 1)

        concat1_2a = self.ad.concat_forward_op(incep1_2_0, incep1_2_1)
        concat1_2b = self.ad.concat_forward_op(concat1_2a, incep1_2_2)
        concat1_2 = self.ad.concat_forward_op(concat1_2b, incep1_2_3)

     # inception_moudle1_3
        filter1_3_0 = self.ad.Variable("filter1_3_0")
        filter1_3_1a = self.ad.Variable("filter1_3_1a")
        filter1_3_1b = self.ad.Variable("filter1_3_1b")
        filter1_3_2a = self.ad.Variable("filter1_3_2a")
        filter1_3_2b = self.ad.Variable("filter1_3_2b")
        filter1_3_2c = self.ad.Variable("filter1_3_2c")
        filter1_3_3 = self.ad.Variable("filter1_3_3")

        filter1_3_0_val = ndarray.array(np.random.normal(0, 0.5, (64, 288, 1, 1)) , executor_ctx)
        filter1_3_1_vala = ndarray.array(np.random.normal(0, 0.5, (48, 288, 1, 1)) , executor_ctx)
        filter1_3_1_valb = ndarray.array(np.random.normal(0, 0.5, (64, 48, 5, 5)) , executor_ctx)
        filter1_3_2_vala = ndarray.array(np.random.normal(0, 0.5, (64, 288, 1, 1)) , executor_ctx)
        filter1_3_2_valb = ndarray.array(np.random.normal(0, 0.5, (96, 64, 3, 3)) , executor_ctx)
        filter1_3_2_valc = ndarray.array(np.random.normal(0, 0.5, (96, 96, 3, 3)) , executor_ctx)
        filter1_3_3_val = ndarray.array(np.random.normal(0, 0.5, (64, 288, 1, 1)) , executor_ctx)

        # branch_0
        incep1_3_0 = self.conv2dplusrelu(concat1_2, filter1_3_0, "NCHW", "SAME", 1, 1)
        # branch 1
        incep1_3_1a = self.conv2dplusrelu(concat1_2, filter1_3_1a, "NCHW", "SAME", 1, 1)
        incep1_3_1 = self.conv2dplusrelu(incep1_3_1a, filter1_3_1b, "NCHW", "SAME", 1, 1)
        # branch 2
        incep1_3_2a = self.conv2dplusrelu(concat1_2, filter1_3_2a, "NCHW", "SAME", 1, 1)
        incep1_3_2b = self.conv2dplusrelu(incep1_3_2a, filter1_3_2b, "NCHW", "SAME", 1, 1)
        incep1_3_2 = self.conv2dplusrelu(incep1_3_2b, filter1_3_2c, "NCHW", "SAME", 1, 1)
        # branch 3
        incep1_3_3a = self.ad.pooling_2d_forward_op(concat1_2, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
        incep1_3_3 = self.conv2dplusrelu(incep1_3_3a, filter1_3_3, "NCHW", "SAME", 1, 1)

        concat1_3a = self.ad.concat_forward_op(incep1_3_0, incep1_3_1)
        concat1_3b = self.ad.concat_forward_op(concat1_3a, incep1_3_2)
        concat1_3 = self.ad.concat_forward_op(concat1_3b, incep1_3_3)

    #
    #
    #
    #
    # # inception_moudle2
        # inception_moudle2_1
        filter2_1_0 = self.ad.Variable("filter2_1_0")
        filter2_1_1a = self.ad.Variable("filter2_1_1a")
        filter2_1_1b = self.ad.Variable("filter2_1_1b")
        filter2_1_1c = self.ad.Variable("filter2_1_1c")

        filter2_1_0_val = ndarray.array(np.random.normal(0, 0.5, (384, 288, 3, 3)) , executor_ctx)
        filter2_1_1_vala = ndarray.array(np.random.normal(0, 0.5, (64, 288, 1, 1)) , executor_ctx)
        filter2_1_1_valb = ndarray.array(np.random.normal(0, 0.5, (96, 64, 3, 3)) , executor_ctx)
        filter2_1_1_valc = ndarray.array(np.random.normal(0, 0.5, (96, 96, 3, 3)) , executor_ctx)

        # branch_0
        incep2_1_0 = self.conv2dplusrelu(concat1_3, filter2_1_0, "NCHW", "VALID", 2, 2)
        # branch 1
        incep2_1_1a = self.conv2dplusrelu(concat1_3, filter2_1_1a, "NCHW", "SAME", 1, 1)
        incep2_1_1b = self.conv2dplusrelu(incep2_1_1a, filter2_1_1b, "NCHW", "SAME", 1, 1)
        incep2_1_1 = self.conv2dplusrelu(incep2_1_1b, filter2_1_1c, "NCHW", "VALID", 2,2)
        # branch 2
        incep2_1_2 = self.ad.pooling_2d_forward_op(concat1_3, "NCHW", "mean", 0, 0, 2, 2, 3, 3)

        concat2_1a = self.ad.concat_forward_op(incep2_1_0, incep2_1_1)
        concat2_1 = self.ad.concat_forward_op(concat2_1a, incep2_1_2)


      # inception_moudle2_2
        filter2_2_0 = self.ad.Variable("filter2_2_0")
        filter2_2_1a = self.ad.Variable("filter2_2_1a")
        filter2_2_1b = self.ad.Variable("filter2_2_1b")
        filter2_2_1c = self.ad.Variable("filter2_2_1c")
        filter2_2_2a = self.ad.Variable("filter2_2_2a")
        filter2_2_2b = self.ad.Variable("filter2_2_2b")
        filter2_2_2c = self.ad.Variable("filter2_2_2c")
        filter2_2_2d = self.ad.Variable("filter2_2_2d")
        filter2_2_2e = self.ad.Variable("filter2_2_2e")
        filter2_2_3 = self.ad.Variable("filter2_2_3a")

        filter2_2_0_val = ndarray.array(np.random.normal(0, 0.5, (192, 768, 1, 1)) , executor_ctx)
        filter2_2_1_vala = ndarray.array(np.random.normal(0, 0.5, (128, 768, 1, 1)) , executor_ctx)
        filter2_2_1_valb = ndarray.array(np.random.normal(0, 0.5, (128, 128, 1, 7)) , executor_ctx)
        filter2_2_1_valc = ndarray.array(np.random.normal(0, 0.5, (192, 128, 7, 1)) , executor_ctx)
        filter2_2_2_vala = ndarray.array(np.random.normal(0, 0.5, (128, 768, 1, 1)) , executor_ctx)
        filter2_2_2_valb = ndarray.array(np.random.normal(0, 0.5, (128, 128, 7,1)) , executor_ctx)
        filter2_2_2_valc = ndarray.array(np.random.normal(0, 0.5, (128, 128, 1, 7)) , executor_ctx)
        filter2_2_2_vald = ndarray.array(np.random.normal(0, 0.5, (128, 128, 7, 1)) , executor_ctx)
        filter2_2_2_vale = ndarray.array(np.random.normal(0, 0.5, (192, 128, 1, 7)) , executor_ctx)
        filter2_2_3_val = ndarray.array(np.random.normal(0, 0.5, (192, 768, 1, 1)) , executor_ctx)

        # branch_0
        incep2_2_0 = self.conv2dplusrelu(concat2_1, filter2_2_0, "NCHW", "SAME", 1, 1)
        # branch 1
        incep2_2_1a = self.conv2dplusrelu(concat2_1, filter2_2_1a, "NCHW", "SAME", 1, 1)
        incep2_2_1b = self.conv2dplusrelu(incep2_2_1a, filter2_2_1b, "NCHW", "SAME", 1, 1)
        incep2_2_1 = self.conv2dplusrelu(incep2_2_1b, filter2_2_1c, "NCHW", "SAME", 1, 1)
        # branch 2
        incep2_2_2a = self.conv2dplusrelu(concat2_1, filter2_2_2a, "NCHW", "SAME", 1, 1)
        incep2_2_2b = self.conv2dplusrelu(incep2_2_2a, filter2_2_2b, "NCHW", "SAME", 1, 1)
        incep2_2_2c = self.conv2dplusrelu(incep2_2_2b, filter2_2_2c, "NCHW", "SAME", 1, 1)
        incep2_2_2d = self.conv2dplusrelu(incep2_2_2c, filter2_2_2d, "NCHW", "SAME", 1, 1)
        incep2_2_2 = self.conv2dplusrelu(incep2_2_2d, filter2_2_2e, "NCHW", "SAME", 1, 1)
        # branch 3
        incep2_2_3a = self.ad.pooling_2d_forward_op(concat2_1, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
        incep2_2_3 = self.conv2dplusrelu(incep2_2_3a, filter2_2_3, "NCHW", "SAME", 1, 1)

        concat2_2a = self.ad.concat_forward_op(incep2_2_0, incep2_2_1)
        concat2_2b = self.ad.concat_forward_op(concat2_2a, incep2_2_2)
        concat2_2 = self.ad.concat_forward_op(concat2_2b, incep2_2_3)

    # inception_moudle2_3
        filter2_3_0 = self.ad.Variable("filter2_3_0")
        filter2_3_1a = self.ad.Variable("filter2_3_1a")
        filter2_3_1b = self.ad.Variable("filter2_3_1b")
        filter2_3_1c = self.ad.Variable("filter2_3_1c")
        filter2_3_2a = self.ad.Variable("filter2_3_2a")
        filter2_3_2b = self.ad.Variable("filter2_3_2b")
        filter2_3_2c = self.ad.Variable("filter2_3_2c")
        filter2_3_2d = self.ad.Variable("filter2_3_2d")
        filter2_3_2e = self.ad.Variable("filter2_3_2e")
        filter2_3_3 = self.ad.Variable("filter2_3_3a")

        filter2_3_0_val = ndarray.array(np.random.normal(0, 0.5, (192, 768, 1, 1)) , executor_ctx)
        filter2_3_1_vala = ndarray.array(np.random.normal(0, 0.5, (160, 768, 1, 1)) , executor_ctx)
        filter2_3_1_valb = ndarray.array(np.random.normal(0, 0.5, (160, 160, 1, 7)) , executor_ctx)
        filter2_3_1_valc = ndarray.array(np.random.normal(0, 0.5, (192, 160, 7, 1)) , executor_ctx)
        filter2_3_2_vala = ndarray.array(np.random.normal(0, 0.5, (160, 768, 1, 1)) , executor_ctx)
        filter2_3_2_valb = ndarray.array(np.random.normal(0, 0.5, (160, 160, 7,1)) , executor_ctx)
        filter2_3_2_valc = ndarray.array(np.random.normal(0, 0.5, (160, 160, 1, 7)) , executor_ctx)
        filter2_3_2_vald = ndarray.array(np.random.normal(0, 0.5, (160, 160, 7, 1)) , executor_ctx)
        filter2_3_2_vale = ndarray.array(np.random.normal(0, 0.5, (192, 160, 1, 7)) , executor_ctx)
        filter2_3_3_val = ndarray.array(np.random.normal(0, 0.5, (192, 768, 1, 1)) , executor_ctx)

        # branch_0
        incep2_3_0 = self.conv2dplusrelu(concat2_2, filter2_3_0, "NCHW", "SAME", 1, 1)
        # branch 1
        incep2_3_1a = self.conv2dplusrelu(concat2_2, filter2_3_1a, "NCHW", "SAME", 1, 1)
        incep2_3_1b = self.conv2dplusrelu(incep2_3_1a, filter2_3_1b, "NCHW", "SAME", 1, 1)
        incep2_3_1 = self.conv2dplusrelu(incep2_3_1b, filter2_3_1c, "NCHW", "SAME", 1, 1)
        # branch 2
        incep2_3_2a = self.conv2dplusrelu(concat2_2, filter2_3_2a, "NCHW", "SAME", 1, 1)
        incep2_3_2b = self.conv2dplusrelu(incep2_3_2a, filter2_3_2b, "NCHW", "SAME", 1, 1)
        incep2_3_2c = self.conv2dplusrelu(incep2_3_2b, filter2_3_2c, "NCHW", "SAME", 1, 1)
        incep2_3_2d = self.conv2dplusrelu(incep2_3_2c, filter2_3_2d, "NCHW", "SAME", 1, 1)
        incep2_3_2 = self.conv2dplusrelu(incep2_3_2d, filter2_3_2e, "NCHW", "SAME", 1, 1)
        # branch 3
        incep2_3_3a = self.ad.pooling_2d_forward_op(concat2_2, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
        incep2_3_3 = self.conv2dplusrelu(incep2_3_3a, filter2_3_3, "NCHW", "SAME", 1, 1)

        concat2_3a = self.ad.concat_forward_op(incep2_3_0, incep2_3_1)
        concat2_3b = self.ad.concat_forward_op(concat2_3a, incep2_3_2)
        concat2_3 = self.ad.concat_forward_op(concat2_3b, incep2_3_3)


     # inception_moudle2_4
        filter2_4_0 = self.ad.Variable("filter2_4_0")
        filter2_4_1a = self.ad.Variable("filter2_4_1a")
        filter2_4_1b = self.ad.Variable("filter2_4_1b")
        filter2_4_1c = self.ad.Variable("filter2_4_1c")
        filter2_4_2a = self.ad.Variable("filter2_4_2a")
        filter2_4_2b = self.ad.Variable("filter2_4_2b")
        filter2_4_2c = self.ad.Variable("filter2_4_2c")
        filter2_4_2d = self.ad.Variable("filter2_4_2d")
        filter2_4_2e = self.ad.Variable("filter2_4_2e")
        filter2_4_3 = self.ad.Variable("filter2_4_3a")

        filter2_4_0_val = ndarray.array(np.random.normal(0, 0.5, (192, 768, 1, 1)) , executor_ctx)
        filter2_4_1_vala = ndarray.array(np.random.normal(0, 0.5, (160, 768, 1, 1)) , executor_ctx)
        filter2_4_1_valb = ndarray.array(np.random.normal(0, 0.5, (160, 160, 1, 7)) , executor_ctx)
        filter2_4_1_valc = ndarray.array(np.random.normal(0, 0.5, (192, 160, 7, 1)) , executor_ctx)
        filter2_4_2_vala = ndarray.array(np.random.normal(0, 0.5, (160, 768, 1, 1)) , executor_ctx)
        filter2_4_2_valb = ndarray.array(np.random.normal(0, 0.5, (160, 160, 7, 1)) , executor_ctx)
        filter2_4_2_valc = ndarray.array(np.random.normal(0, 0.5, (160, 160, 1, 7)) , executor_ctx)
        filter2_4_2_vald = ndarray.array(np.random.normal(0, 0.5, (160, 160, 7, 1)) , executor_ctx)
        filter2_4_2_vale = ndarray.array(np.random.normal(0, 0.5, (192, 160, 1, 7)) , executor_ctx)
        filter2_4_3_val = ndarray.array(np.random.normal(0, 0.5, (192, 768, 1, 1)) , executor_ctx)

        # branch_0
        incep2_4_0 = self.conv2dplusrelu(concat2_3, filter2_4_0, "NCHW", "SAME", 1, 1)
        # branch 1
        incep2_4_1a = self.conv2dplusrelu(concat2_3, filter2_4_1a, "NCHW", "SAME", 1, 1)
        incep2_4_1b = self.conv2dplusrelu(incep2_4_1a, filter2_4_1b, "NCHW", "SAME", 1, 1)
        incep2_4_1 = self.conv2dplusrelu(incep2_4_1b, filter2_4_1c, "NCHW", "SAME", 1, 1)
        # branch 2
        incep2_4_2a = self.conv2dplusrelu(concat2_3, filter2_4_2a, "NCHW", "SAME", 1, 1)
        incep2_4_2b = self.conv2dplusrelu(incep2_4_2a, filter2_4_2b, "NCHW", "SAME", 1, 1)
        incep2_4_2c = self.conv2dplusrelu(incep2_4_2b, filter2_4_2c, "NCHW", "SAME", 1, 1)
        incep2_4_2d = self.conv2dplusrelu(incep2_4_2c, filter2_4_2d, "NCHW", "SAME", 1, 1)
        incep2_4_2 = self.conv2dplusrelu(incep2_4_2d, filter2_4_2e, "NCHW", "SAME", 1, 1)
        # branch 3
        incep2_4_3a = self.ad.pooling_2d_forward_op(concat2_3, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
        incep2_4_3 = self.conv2dplusrelu(incep2_4_3a, filter2_4_3, "NCHW", "SAME", 1, 1)

        concat2_4a = self.ad.concat_forward_op(incep2_4_0, incep2_4_1)
        concat2_4b = self.ad.concat_forward_op(concat2_4a, incep2_4_2)
        concat2_4 = self.ad.concat_forward_op(concat2_4b, incep2_4_3)


    # inception_moudle2_5
        filter2_5_0 = self.ad.Variable("filter2_5_0")
        filter2_5_1a = self.ad.Variable("filter2_5_1a")
        filter2_5_1b = self.ad.Variable("filter2_5_1b")
        filter2_5_1c = self.ad.Variable("filter2_5_1c")
        filter2_5_2a = self.ad.Variable("filter2_5_2a")
        filter2_5_2b = self.ad.Variable("filter2_5_2b")
        filter2_5_2c = self.ad.Variable("filter2_5_2c")
        filter2_5_2d = self.ad.Variable("filter2_5_2d")
        filter2_5_2e = self.ad.Variable("filter2_5_2e")
        filter2_5_3 = self.ad.Variable("filter2_5_3a")

        filter2_5_0_val = ndarray.array(np.random.normal(0, 0.5, (192, 768, 1, 1)) , executor_ctx)
        filter2_5_1_vala = ndarray.array(np.random.normal(0, 0.5, (160, 768, 1, 1)) , executor_ctx)
        filter2_5_1_valb = ndarray.array(np.random.normal(0, 0.5, (160, 160, 1, 7)) , executor_ctx)
        filter2_5_1_valc = ndarray.array(np.random.normal(0, 0.5, (192, 160, 7, 1)) , executor_ctx)
        filter2_5_2_vala = ndarray.array(np.random.normal(0, 0.5, (160, 768, 1, 1)) , executor_ctx)
        filter2_5_2_valb = ndarray.array(np.random.normal(0, 0.5, (160, 160, 7, 1)) , executor_ctx)
        filter2_5_2_valc = ndarray.array(np.random.normal(0, 0.5, (160, 160, 1, 7)) , executor_ctx)
        filter2_5_2_vald = ndarray.array(np.random.normal(0, 0.5, (160, 160, 7, 1)) , executor_ctx)
        filter2_5_2_vale = ndarray.array(np.random.normal(0, 0.5, (192, 160, 1, 7)) , executor_ctx)
        filter2_5_3_val = ndarray.array(np.random.normal(0, 0.5, (192, 768, 1, 1)) , executor_ctx)

        # branch_0
        incep2_5_0 = self.conv2dplusrelu(concat2_4, filter2_5_0, "NCHW", "SAME", 1, 1)
        # branch 1
        incep2_5_1a = self.conv2dplusrelu(concat2_4, filter2_5_1a, "NCHW", "SAME", 1, 1)
        incep2_5_1b = self.conv2dplusrelu(incep2_5_1a, filter2_5_1b, "NCHW", "SAME", 1, 1)
        incep2_5_1 = self.conv2dplusrelu(incep2_5_1b, filter2_5_1c, "NCHW", "SAME", 1, 1)
        # branch 2
        incep2_5_2a = self.conv2dplusrelu(concat2_4, filter2_5_2a, "NCHW", "SAME", 1, 1)
        incep2_5_2b = self.conv2dplusrelu(incep2_5_2a, filter2_5_2b, "NCHW", "SAME", 1, 1)
        incep2_5_2c = self.conv2dplusrelu(incep2_5_2b, filter2_5_2c, "NCHW", "SAME", 1, 1)
        incep2_5_2d = self.conv2dplusrelu(incep2_5_2c, filter2_5_2d, "NCHW", "SAME", 1, 1)
        incep2_5_2 = self.conv2dplusrelu(incep2_5_2d, filter2_5_2e, "NCHW", "SAME", 1, 1)
        # branch 3
        incep2_5_3a = self.ad.pooling_2d_forward_op(concat2_4, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
        incep2_5_3 = self.conv2dplusrelu(incep2_5_3a, filter2_5_3, "NCHW", "SAME", 1, 1)

        concat2_5a = self.ad.concat_forward_op(incep2_5_0, incep2_5_1)
        concat2_5b = self.ad.concat_forward_op(concat2_5a, incep2_5_2)
        concat2_5 = self.ad.concat_forward_op(concat2_5b, incep2_5_3)





    # # inception_moudle3
        # inception_moudle3_1
        filter3_1_0a = self.ad.Variable("filter3_1_0a")
        filter3_1_0b = self.ad.Variable("filter3_1_0b")
        filter3_1_1a = self.ad.Variable("filter3_1_1a")
        filter3_1_1b = self.ad.Variable("filter3_1_1b")
        filter3_1_1c = self.ad.Variable("filter3_1_1c")
        filter3_1_1d = self.ad.Variable("filter3_1_1d")

        filter3_1_0_vala = ndarray.array(np.random.normal(0, 0.5, (192, 768, 1, 1)) , executor_ctx)
        filter3_1_0_valb = ndarray.array(np.random.normal(0, 0.5, (320, 192, 3, 3)) , executor_ctx)
        filter3_1_1_vala = ndarray.array(np.random.normal(0, 0.5, (192, 768, 1, 1)) , executor_ctx)
        filter3_1_1_valb = ndarray.array(np.random.normal(0, 0.5, (192, 192, 1,7)) , executor_ctx)
        filter3_1_1_valc = ndarray.array(np.random.normal(0, 0.5, (192, 192, 7, 1)) , executor_ctx)
        filter3_1_1_vald = ndarray.array(np.random.normal(0, 0.5, (192, 192, 3, 3)) , executor_ctx)

        # branch_0
        incep3_1_0a = self.conv2dplusrelu(concat2_5, filter3_1_0a, "NCHW", "SAME", 1, 1)
        incep3_1_0 = self.conv2dplusrelu(incep3_1_0a, filter3_1_0b, "NCHW", "VALID", 2,2)
        # branch 1
        incep3_1_1a = self.conv2dplusrelu(concat2_2, filter3_1_1a, "NCHW", "SAME", 1, 1)
        incep3_1_1b = self.conv2dplusrelu(incep3_1_1a, filter3_1_1b, "NCHW", "SAME", 1, 1)
        incep3_1_1c = self.conv2dplusrelu(incep3_1_1b, filter3_1_1c, "NCHW", "SAME", 1, 1)
        incep3_1_1 = self.conv2dplusrelu(incep3_1_1c, filter3_1_1d, "NCHW", "VALID", 2, 2)
        # branch 2
        incep3_1_2 = self.ad.pooling_2d_forward_op(concat2_2, "NCHW", "mean", 0, 0, 2, 2, 3, 3)

        concat3_1a = self.ad.concat_forward_op(incep3_1_0, incep3_1_1)
        concat3_1 = self.ad.concat_forward_op(concat3_1a, incep3_1_2)



    # inception_moudle3_2
        filter3_2_0 = self.ad.Variable("filter3_2_0")
        filter3_2_1a = self.ad.Variable("filter3_2_1a")
        filter3_2_1b = self.ad.Variable("filter3_2_1b")
        filter3_2_1c = self.ad.Variable("filter3_2_1c")
        filter3_2_2a = self.ad.Variable("filter3_2_2a")
        filter3_2_2b = self.ad.Variable("filter3_2_2b")
        filter3_2_2c = self.ad.Variable("filter3_2_2c")
        filter3_2_2d = self.ad.Variable("filter3_2_2d")
        filter3_2_3 = self.ad.Variable("filter3_2_3a")

        filter3_2_0_val = ndarray.array(np.random.normal(0, 0.5, (320, 1280, 1, 1)) , executor_ctx)
        filter3_2_1_vala = ndarray.array(np.random.normal(0, 0.5, (384, 1280, 1, 1)) , executor_ctx)
        filter3_2_1_valb = ndarray.array(np.random.normal(0, 0.5, (384, 384, 1, 3)) , executor_ctx)
        filter3_2_1_valc = ndarray.array(np.random.normal(0, 0.5, (384, 384, 3, 1)) , executor_ctx)
        filter3_2_2_vala = ndarray.array(np.random.normal(0, 0.5, (448, 1280, 1, 1)) , executor_ctx)
        filter3_2_2_valb = ndarray.array(np.random.normal(0, 0.5, (384, 448, 3, 3)) , executor_ctx)
        filter3_2_2_valc = ndarray.array(np.random.normal(0, 0.5, (384, 384, 1, 3)) , executor_ctx)
        filter3_2_2_vald = ndarray.array(np.random.normal(0, 0.5, (384, 384, 3, 1)) , executor_ctx)
        filter3_2_3_val = ndarray.array(np.random.normal(0, 0.5, (192, 1280, 1, 1)) , executor_ctx)

        # branch_0
        incep3_2_0 = self.conv2dplusrelu(concat3_1, filter3_2_0, "NCHW", "SAME", 1, 1)
        # branch 1
        incep3_2_1a = self.conv2dplusrelu(concat3_1, filter3_2_1a, "NCHW", "SAME", 1, 1)
        incep3_2_1b = self.conv2dplusrelu(incep3_2_1a, filter3_2_1b, "NCHW", "SAME", 1, 1)
        incep3_2_1c = self.conv2dplusrelu(incep3_2_1a, filter3_2_1c, "NCHW", "SAME", 1, 1)
        incep3_2_1=self.ad.concat_forward_op(incep3_2_1b, incep3_2_1c)
        # branch 2
        incep3_2_2a = self.conv2dplusrelu(concat3_1, filter3_2_2a, "NCHW", "SAME", 1, 1)
        incep3_2_2b = self.conv2dplusrelu(incep3_2_2a, filter3_2_2b, "NCHW", "SAME", 1, 1)
        incep3_2_2c = self.conv2dplusrelu(incep3_2_2b, filter3_2_2c, "NCHW", "SAME", 1, 1)
        incep3_2_2d = self.conv2dplusrelu(incep3_2_2b, filter3_2_2d, "NCHW", "SAME", 1, 1)
        incep3_2_2 = self.ad.concat_forward_op(incep3_2_2c, incep3_2_2d)
        # branch 3
        incep3_2_3a = self.ad.pooling_2d_forward_op(concat3_1, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
        incep3_2_3 = self.conv2dplusrelu(incep3_2_3a, filter3_2_3, "NCHW", "SAME", 1, 1)

        concat3_2a = self.ad.concat_forward_op(incep3_2_0, incep3_2_1)
        concat3_2b = self.ad.concat_forward_op(concat3_2a, incep3_2_2)
        concat3_2 = self.ad.concat_forward_op(concat3_2b, incep3_2_3)


    # # inception_moudle3_3
        filter3_3_0 = self.ad.Variable("filter3_3_0")
        filter3_3_1a = self.ad.Variable("filter3_3_1a")
        filter3_3_1b = self.ad.Variable("filter3_3_1b")
        filter3_3_1c = self.ad.Variable("filter3_3_1c")
        filter3_3_2a = self.ad.Variable("filter3_3_2a")
        filter3_3_2b = self.ad.Variable("filter3_3_2b")
        filter3_3_2c = self.ad.Variable("filter3_3_2c")
        filter3_3_2d = self.ad.Variable("filter3_3_2d")
        filter3_3_3 = self.ad.Variable("filter3_3_3a")

        filter3_3_0_val = ndarray.array(np.random.normal(0, 0.5, (320, 2048, 1, 1)) , executor_ctx)
        filter3_3_1_vala = ndarray.array(np.random.normal(0, 0.5, (384, 2048, 1, 1)) , executor_ctx)
        filter3_3_1_valb = ndarray.array(np.random.normal(0, 0.5, (384, 384, 1, 3)) , executor_ctx)
        filter3_3_1_valc = ndarray.array(np.random.normal(0, 0.5, (384, 384, 3, 1)) , executor_ctx)
        filter3_3_2_vala = ndarray.array(np.random.normal(0, 0.5, (448, 2048, 1, 1)) , executor_ctx)
        filter3_3_2_valb = ndarray.array(np.random.normal(0, 0.5, (384, 448, 3, 3)) , executor_ctx)
        filter3_3_2_valc = ndarray.array(np.random.normal(0, 0.5, (384, 384, 1, 3)) , executor_ctx)
        filter3_3_2_vald = ndarray.array(np.random.normal(0, 0.5, (384, 384, 3, 1)) , executor_ctx)
        filter3_3_3_val = ndarray.array(np.random.normal(0, 0.5, (192, 2048, 1, 1)) , executor_ctx)

        # branch_0
        incep3_3_0 = self.conv2dplusrelu(concat3_2, filter3_3_0, "NCHW", "SAME", 1, 1)
        # branch 1
        incep3_3_1a = self.conv2dplusrelu(concat3_2, filter3_3_1a, "NCHW", "SAME", 1, 1)
        incep3_3_1b = self.conv2dplusrelu(incep3_3_1a, filter3_3_1b, "NCHW", "SAME", 1, 1)
        incep3_3_1c = self.conv2dplusrelu(incep3_3_1a, filter3_3_1c, "NCHW", "SAME", 1, 1)
        incep3_3_1 = self.ad.concat_forward_op(incep3_3_1b, incep3_3_1c)

        # branch 2
        incep3_3_2a = self.conv2dplusrelu(concat3_2, filter3_3_2a, "NCHW", "SAME", 1, 1)
        incep3_3_2b = self.conv2dplusrelu(incep3_3_2a, filter3_3_2b, "NCHW", "SAME", 1, 1)
        incep3_3_2c = self.conv2dplusrelu(incep3_3_2b, filter3_3_2c, "NCHW", "SAME", 1, 1)
        incep3_3_2d = self.conv2dplusrelu(incep3_3_2b, filter3_3_2d, "NCHW", "SAME", 1, 1)
        incep3_3_2 = self.ad.concat_forward_op(incep3_3_2c, incep3_3_2d)
        # branch 3
        incep3_3_3a = self.ad.pooling_2d_forward_op(concat3_2, "NCHW", "mean", 1, 1, 1, 1, 3, 3)
        incep3_3_3 = self.conv2dplusrelu(incep3_3_3a, filter3_3_3, "NCHW", "SAME", 1, 1)

        concat3_3a = self.ad.concat_forward_op(incep3_3_0, incep3_3_1)
        concat3_3b = self.ad.concat_forward_op(concat3_3a, incep3_3_2)
        concat3_3 = self.ad.concat_forward_op(concat3_3b, incep3_3_3)

        filtera1 = self.ad.Variable("filtera1")
        filtera1val = ndarray.array(np.random.normal(0, 0.5, (1000, 2048, 1, 1)) , executor_ctx)

        W = self.ad.Variable("filtersmul")
        W_val = ndarray.array(np.random.normal(0, 0.5, (1000, 1000)) , executor_ctx)

        b= self.ad.Variable("biases")
        b_val = ndarray.array(np.random.normal(0, 0.5, (1000)) , executor_ctx)


        poollast=self.ad.pooling_2d_forward_op(concat3_3,"NCHW", "mean",0,0,1,1,8,8)
        dropout=self.ad.dropout_forward_op(poollast,"NCHW",0.8)
        convlast=self.conv2dplusrelu(dropout, filtera1, "NCHW", "SAME", 1, 1)
        squeeze=self.ad.squeeze_op(convlast)

        dense = self.ad.dense(squeeze, W, b)
        y = self.ad.fullyactivation_forward_op(dense, "NCHW", "softmax")
        loss = self.ad.crossEntropy_loss(y, y_)
        # fc8
        if not os.path.exists('./log/Inception V3/test'):
            os.makedirs('./log/Inception V3/test')
        executor = self.ad.Executor(loss, y, 0.001, top_control_queue=top_control_queue,
                                    top_message_queue=top_message_queue,log_path='./log/Inception V3/test')

        feed_dict={ filterb_1: filtersb_val1, filterb_2: filtersb_val2, filterb_3: filtersb_val3
                                           , filterb_4: filtersb_val4, filterb_5: filtersb_val5,
             filter1_1_0:filter1_1_0_val ,filter1_1_1a:filter1_1_1_vala,filter1_1_1b:filter1_1_1_valb,filter1_1_2a:filter1_1_2_vala,filter1_1_2b:filter1_1_2_valb
                                           ,filter1_1_2c:filter1_1_2_valc,filter1_1_3:filter1_1_3_val
             ,filter1_2_0: filter1_2_0_val, filter1_2_1a: filter1_2_1_vala,
                                           filter1_2_1b: filter1_2_1_valb, filter1_2_2a: filter1_2_2_vala,
                                           filter1_2_2b: filter1_2_2_valb, filter1_2_2c: filter1_2_2_valc, filter1_2_3: filter1_2_3_val
    
            , filter1_3_0: filter1_3_0_val, filter1_3_1a: filter1_3_1_vala,
                                           filter1_3_1b: filter1_3_1_valb, filter1_3_2a: filter1_3_2_vala,
                                           filter1_3_2b: filter1_3_2_valb, filter1_3_2c: filter1_3_2_valc,
                                           filter1_3_3: filter1_3_3_val
            ,filter2_1_0:filter2_1_0_val,filter2_1_1a:filter2_1_1_vala,filter2_1_1b:filter2_1_1_valb,filter2_1_1c:filter2_1_1_valc
    
            ,filter2_2_0:filter2_2_0_val,filter2_2_1a:filter2_2_1_vala,filter2_2_1b:filter2_2_1_valb,filter2_2_1c:filter2_2_1_valc,
                                           filter2_2_2a:filter2_2_2_vala,filter2_2_2b:filter2_2_2_valb,filter2_2_2c:filter2_2_2_valc,filter2_2_2d:filter2_2_2_vald,filter2_2_2e:filter2_2_2_vale,filter2_2_3:filter2_2_3_val
    
            , filter2_3_0: filter2_3_0_val, filter2_3_1a: filter2_3_1_vala, filter2_3_1b: filter2_3_1_valb,
                                           filter2_3_1c: filter2_3_1_valc,
                                           filter2_3_2a: filter2_3_2_vala, filter2_3_2b: filter2_3_2_valb,
                                           filter2_3_2c: filter2_3_2_valc, filter2_3_2d: filter2_3_2_vald,
                                           filter2_3_2e: filter2_3_2_vale, filter2_3_3: filter2_3_3_val
            , filter2_4_0: filter2_4_0_val, filter2_4_1a: filter2_4_1_vala, filter2_4_1b: filter2_4_1_valb,
                                           filter2_4_1c: filter2_4_1_valc,
                                           filter2_4_2a: filter2_4_2_vala, filter2_4_2b: filter2_4_2_valb,
                                           filter2_4_2c: filter2_4_2_valc, filter2_4_2d: filter2_4_2_vald,
                                           filter2_4_2e: filter2_4_2_vale, filter2_4_3: filter2_4_3_val
            , filter2_5_0: filter2_5_0_val, filter2_5_1a: filter2_5_1_vala, filter2_5_1b: filter2_5_1_valb,
                                           filter2_5_1c: filter2_5_1_valc,
                                           filter2_5_2a: filter2_5_2_vala, filter2_5_2b: filter2_5_2_valb,
                                           filter2_5_2c: filter2_5_2_valc, filter2_5_2d: filter2_5_2_vald,
                                           filter2_5_2e: filter2_5_2_vale, filter2_5_3: filter2_5_3_val
            , filter3_1_0a: filter3_1_0_vala,filter3_1_0b: filter3_1_0_valb,  filter3_1_1a: filter3_1_1_vala, filter3_1_1b: filter3_1_1_valb,
                                           filter3_1_1c: filter3_1_1_valc,filter3_1_1d: filter3_1_1_vald
            , filter3_2_0: filter3_2_0_val, filter3_2_1a: filter3_2_1_vala,
                                           filter3_2_1b: filter3_2_1_valb,
                                           filter3_2_1c: filter3_2_1_valc,filter3_2_2a: filter3_2_2_vala,filter3_2_2b: filter3_2_2_valb,
                                           filter3_2_2c: filter3_2_2_valc,filter3_2_2d: filter3_2_2_vald,filter3_2_3: filter3_2_3_val
            , filter3_3_0: filter3_3_0_val, filter3_3_1a: filter3_3_1_vala,
                                           filter3_3_1b: filter3_3_1_valb,
                                           filter3_3_1c: filter3_3_1_valc, filter3_3_2a: filter3_3_2_vala,
                                           filter3_3_2b: filter3_3_2_valb,
                                           filter3_3_2c: filter3_3_2_valc, filter3_3_2d: filter3_3_2_vald,
                                           filter3_3_3: filter3_3_3_val
            ,filtera1:filtera1val,W:W_val,b:b_val}


        feed_dict_mv = {}
        for key, value in feed_dict.items():
            print(key)
            m_key = executor.Variable_node_to_mv[key][0]
            m_val = ndarray.array(np.zeros(shape=value.shape), executor_ctx)
            v_key = executor.Variable_node_to_mv[key][1]
            v_val = ndarray.array(np.zeros(shape=value.shape), executor_ctx)
            feed_dict_mv.update({m_key: m_val, v_key: v_val})

        feed_dict.update(feed_dict_mv)
        f1 = open(f"{log_path}/gpu_time.txt", "w+")
        for i in range(self.num_step):
            print("step", i)
            if i==5:
                gpu_record.start()
                start_time = time.time()
            if i==15:
                gpu_record.stop()
                f1.write(f'time_cost:{time.time() - start_time}')
                f1.flush()
                f1.close()
            feed_dict[X] = ndarray.array(X_val, ctx=executor_ctx)
            feed_dict[y_] = ndarray.array(y_val, ctx=executor_ctx)
            res = executor.run(feed_dict=feed_dict)
            loss_val = res[0]
            feed_dict = res[1]

        print("success")
        return 0


if __name__ == '__main__':
    # gpu_record = GPURecord()
    global_message_queue = multiprocessing.Queue()
    global_control_queue = multiprocessing.Queue()

    top_control_queue_list = []
    top_message_queue_list = []
    executor_ctx = ndarray.gpu(0)
    print_loss_val_each_epoch = True

    top_control_queue = multiprocessing.Queue()
    top_control_queue_list.append(top_control_queue)
    top_message_queue = multiprocessing.Queue()
    top_message_queue_list.append(top_message_queue)
    job_number = 1

    gpu_num = GPU
    batch_size = 32
    num_step = 200
    inceptionv3 = Inceptionv3(num_step=num_step, batch_size=batch_size, gpu_num=gpu_num)
    X_val = np.random.normal(loc=0, scale=0.1, size=(
    batch_size, 3, 299, 299))  # number = batch_size  channel = 3  image_size = 224*224

    y_val = np.random.normal(loc=0, scale=0.1, size=(batch_size, 1000))  # n_class = 1000

    p1 = Process(target=inceptionv3.inception_v3,
                 args=(executor_ctx, top_control_queue, top_message_queue, 1000, X_val, y_val))
    p1.start()
    # p1.join()

    # top_control_queue2 = multiprocessing.Queue()
    # top_control_queue_list.append(top_control_queue2)
    # top_message_queue2 = multiprocessing.Queue()
    # top_message_queue_list.append(top_message_queue2)
    # job_number += 1
    #
    # gpu_num = GPU
    # batch_size = 4
    # num_step = 20
    # inceptionv3 = Inceptionv3(num_step=num_step, batch_size=batch_size, gpu_num=gpu_num)
    # X_val = np.random.normal(loc=0, scale=0.1, size=(
    #     batch_size, 3, 299, 299))  # number = batch_size  channel = 3  image_size = 224*224
    #
    # y_val = np.random.normal(loc=0, scale=0.1, size=(batch_size, 1000))  # n_class = 1000
    #
    # p2 = Process(target=inceptionv3.inception_v3,
    #              args=(executor_ctx, top_control_queue2, top_message_queue2, 1000, X_val, y_val))
    # p2.start()
    # # p1.join()

    if 'schedule' in log_path:
        scheduler = Process(target=mp.multiprocess_init, args=(global_message_queue, global_control_queue))
        scheduler.start()
    # scheduler.join()
    # gpu_record.start()

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