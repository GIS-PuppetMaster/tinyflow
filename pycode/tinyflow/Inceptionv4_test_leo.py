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
class Inceptionv4():
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

    def block_inception_a(self,inputs,blockname,executor_ctx):
        filter1_1_0 = self.ad.Variable(blockname +"filter1_1_0")
        filter1_1_1a = self.ad.Variable(blockname+"filter1_1_1a")
        filter1_1_1b = self.ad.Variable(blockname+"filter1_1_1b")
        filter1_1_2a = self.ad.Variable(blockname+"filter1_1_2a")
        filter1_1_2b = self.ad.Variable(blockname+"filter1_1_2b")
        filter1_1_2c = self.ad.Variable(blockname+"filter1_1_2c")
        filter1_1_3 = self.ad.Variable(blockname+"filter1_1_3a")

        filter1_1_0_val = ndarray.array(np.random.normal(scale=0.1, size=(96, 384, 1, 1)), executor_ctx)
        filter1_1_1_vala = ndarray.array(np.random.normal(scale=0.1, size=(64, 384, 1, 1)), executor_ctx)
        filter1_1_1_valb = ndarray.array(np.random.normal(scale=0.1, size=(96, 64, 3, 3)), executor_ctx)
        filter1_1_2_vala = ndarray.array(np.random.normal(scale=0.1, size=(64, 384, 1, 1)), executor_ctx)
        filter1_1_2_valb = ndarray.array(np.random.normal(scale=0.1, size=(96, 64, 3, 3)), executor_ctx)
        filter1_1_2_valc = ndarray.array(np.random.normal(scale=0.1, size=(96, 96, 3, 3)), executor_ctx)
        filter1_1_3_val =  ndarray.array(np.random.normal(scale=0.1, size=(96, 384, 1, 1)), executor_ctx)

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

        dict = { filter1_1_0:filter1_1_0_val ,filter1_1_1a:filter1_1_1_vala,filter1_1_1b:filter1_1_1_valb,filter1_1_2a:filter1_1_2_vala,filter1_1_2b:filter1_1_2_valb
                                           ,filter1_1_2c:filter1_1_2_valc,filter1_1_3:filter1_1_3_val}
        return concat1_1,dict
    def block_reduction_a(self,inputs,input_size,blockname, executor_ctx):
        filter2_1_0 = self.ad.Variable(blockname+"filter2_1_0")
        filter2_1_1a = self.ad.Variable(blockname+"filter2_1_1a")
        filter2_1_1b = self.ad.Variable(blockname+"filter2_1_1b")
        filter2_1_1c = self.ad.Variable(blockname+"filter2_1_1c")


        filter2_1_0_val = ndarray.array(np.random.normal(scale=0.1, size=(384, input_size, 3, 3)), executor_ctx)
        filter2_1_1_vala =ndarray.array(np.random.normal(scale=0.1, size=(192, input_size, 1, 1)), executor_ctx)
        filter2_1_1_valb =ndarray.array(np.random.normal(scale=0.1, size=(224, 192, 3, 3)), executor_ctx)
        filter2_1_1_valc =ndarray.array(np.random.normal(scale=0.1, size=(256, 224, 3, 3)), executor_ctx)

        # branch_0
        incep2_1_0 = self.conv2dplusrelu(inputs, filter2_1_0, "NCHW", "VALID", 2, 2)
        # branch 1
        incep2_1_1a = self.conv2dplusrelu(inputs, filter2_1_1a, "NCHW", "SAME", 1, 1)
        incep2_1_1b = self.conv2dplusrelu(incep2_1_1a, filter2_1_1b, "NCHW", "SAME", 1, 1)
        incep2_1_1 = self.conv2dplusrelu(incep2_1_1b, filter2_1_1c, "NCHW", "VALID", 2,2)
        # branch 2
        incep2_1_2 = self.ad.pooling_2d_forward_op(inputs, "NCHW", "mean", 0, 0, 2, 2, 3, 3)

        concat2_1a = self.ad.concat_forward_op(incep2_1_0, incep2_1_1)
        concat2_1 = self.ad.concat_forward_op(concat2_1a, incep2_1_2)
        dict = {filter2_1_0:filter2_1_0_val,filter2_1_1a:filter2_1_1_vala,filter2_1_1b:filter2_1_1_valb,filter2_1_1c:filter2_1_1_valc}
        return concat2_1,dict

    def block_inception_b(self,inputs,input_size,blockname, executor_ctx):
        filter2_2_0 = self.ad.Variable(blockname+"filter2_2_0")
        filter2_2_1a = self.ad.Variable(blockname+"filter2_2_1a")
        filter2_2_1b = self.ad.Variable(blockname+"filter2_2_1b")
        filter2_2_1c = self.ad.Variable(blockname+"filter2_2_1c")
        filter2_2_2a = self.ad.Variable(blockname+"filter2_2_2a")
        filter2_2_2b = self.ad.Variable(blockname+"filter2_2_2b")
        filter2_2_2c = self.ad.Variable(blockname+"filter2_2_2c")
        filter2_2_2d = self.ad.Variable(blockname+"filter2_2_2d")
        filter2_2_2e = self.ad.Variable(blockname+"filter2_2_2e")
        filter2_2_3 = self.ad.Variable(blockname+"filter2_2_3a")

        filter2_2_0_val =   ndarray.array(np.random.normal(scale=0.1, size=(384, input_size, 1, 1)), executor_ctx)
        filter2_2_1_vala =  ndarray.array(np.random.normal(scale=0.1, size=(192, input_size, 1, 1)), executor_ctx)
        filter2_2_1_valb =  ndarray.array(np.random.normal(scale=0.1, size=(224, 192, 1, 7)), executor_ctx)
        filter2_2_1_valc =  ndarray.array(np.random.normal(scale=0.1, size=(256, 224, 7, 1)), executor_ctx)
        filter2_2_2_vala =  ndarray.array(np.random.normal(scale=0.1, size=(192, input_size, 1, 1)), executor_ctx)
        filter2_2_2_valb =  ndarray.array(np.random.normal(scale=0.1, size=(192, 192, 7,1)), executor_ctx)
        filter2_2_2_valc =  ndarray.array(np.random.normal(scale=0.1, size=(224, 192, 1, 7)), executor_ctx)
        filter2_2_2_vald =  ndarray.array(np.random.normal(scale=0.1, size=(224, 224, 7, 1)), executor_ctx)
        filter2_2_2_vale =  ndarray.array(np.random.normal(scale=0.1, size=(256, 224, 1, 7)), executor_ctx)
        filter2_2_3_val =   ndarray.array(np.random.normal(scale=0.1, size=(128, input_size, 1, 1)), executor_ctx)
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
        dict = {filter2_2_0:filter2_2_0_val,filter2_2_1a:filter2_2_1_vala,filter2_2_1b:filter2_2_1_valb,filter2_2_1c:filter2_2_1_valc,
                                           filter2_2_2a:filter2_2_2_vala,filter2_2_2b:filter2_2_2_valb,filter2_2_2c:filter2_2_2_valc,filter2_2_2d:filter2_2_2_vald,filter2_2_2e:filter2_2_2_vale,filter2_2_3:filter2_2_3_val
    }
        return concat2_2, dict

    def block_reduction_b(self,inputs,input_size,blockname, executor_ctx):
        filter3_1_0a = self.ad.Variable(blockname+"filter3_1_0a")
        filter3_1_0b = self.ad.Variable(blockname+"filter3_1_0b")
        filter3_1_1a = self.ad.Variable(blockname+"filter3_1_1a")
        filter3_1_1b = self.ad.Variable(blockname+"filter3_1_1b")
        filter3_1_1c = self.ad.Variable(blockname+"filter3_1_1c")
        filter3_1_1d = self.ad.Variable(blockname+"filter3_1_1d")

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
        dict = {filter3_1_0a: filter3_1_0_vala,filter3_1_0b: filter3_1_0_valb,  filter3_1_1a: filter3_1_1_vala, filter3_1_1b: filter3_1_1_valb,
                                           filter3_1_1c: filter3_1_1_valc,filter3_1_1d: filter3_1_1_vald}
        return concat3_1,dict

    def block_inception_c(self,inputs,input_size,blockname, executor_ctx):
        filter3_2_0 = self.ad.Variable(blockname+"filter3_2_0")
        filter3_2_1a = self.ad.Variable(blockname+"filter3_2_1a")
        filter3_2_1b = self.ad.Variable(blockname+"filter3_2_1b")
        filter3_2_1c = self.ad.Variable(blockname+"filter3_2_1c")
        filter3_2_2a = self.ad.Variable(blockname+"filter3_2_2a")
        filter3_2_2b = self.ad.Variable(blockname+"filter3_2_2b")
        filter3_2_2c = self.ad.Variable(blockname+"filter3_2_2c")
        filter3_2_2d = self.ad.Variable(blockname+"filter3_2_2d")
        filter3_2_2e = self.ad.Variable(blockname+"filter3_2_2e")
        filter3_2_3 = self.ad.Variable(blockname+"filter3_2_3a")

        filter3_2_0_val = ndarray.array(np.random.normal(scale=0.1, size=(256, input_size, 1, 1)), executor_ctx)
        filter3_2_1_vala = ndarray.array(np.random.normal(scale=0.1, size=(384, input_size, 1, 1)), executor_ctx)
        filter3_2_1_valb = ndarray.array(np.random.normal(scale=0.1, size=(256, 384, 1, 3)), executor_ctx)
        filter3_2_1_valc = ndarray.array(np.random.normal(scale=0.1, size=(256, 384, 3, 1)), executor_ctx)
        filter3_2_2_vala = ndarray.array(np.random.normal(scale=0.1, size=(384, input_size, 1, 1)), executor_ctx)
        filter3_2_2_valb = ndarray.array(np.random.normal(scale=0.1, size=(448, 384, 1, 3)), executor_ctx)
        filter3_2_2_valc = ndarray.array(np.random.normal(scale=0.1, size=(512, 448, 3, 1)), executor_ctx)
        filter3_2_2_vald = ndarray.array(np.random.normal(scale=0.1, size=(256, 512, 3, 1)), executor_ctx)
        filter3_2_2_vale = ndarray.array(np.random.normal(scale=0.1, size=(256, 512, 1,3)), executor_ctx)
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
        dict = { filter3_2_0: filter3_2_0_val, filter3_2_1a: filter3_2_1_vala,
                                           filter3_2_1b: filter3_2_1_valb,
                                           filter3_2_1c: filter3_2_1_valc,filter3_2_2a: filter3_2_2_vala,filter3_2_2b: filter3_2_2_valb,
                                           filter3_2_2c: filter3_2_2_valc,filter3_2_2d: filter3_2_2_vald,filter3_2_2e: filter3_2_2_vale,filter3_2_3: filter3_2_3_val}
        return concat3_2, dict

    def inception_v4(self, executor_ctx, top_control_queue, top_message_queue, n_class, X_val, y_val):
        gpu_record = GPURecord(log_path)
        start_time = datetime.datetime.now()
        X = self.ad.Placeholder("X")
        Y_ = self.ad.Placeholder("y_")
        f1=self.ad.Variable("f1")
        f2=self.ad.Variable("f2")
        f3 = self.ad.Variable("f3")
        f4= self.ad.Variable("f4")
        f5_1= self.ad.Variable("f5_1")
        f5_2 = self.ad.Variable("f5_2")
        f6_1 = self.ad.Variable("f6_1")
        f6_2 = self.ad.Variable("f6_2")
        f6_3 = self.ad.Variable("f6_3")
        f6_4 = self.ad.Variable("f6_4")
        f7 = self.ad.Variable("f7")
        W = self.ad.Variable("W")
        b = self.ad.Variable("b")

        f1val = ndarray.array(np.random.normal(0, 0.5, (32, 3, 3, 3)), executor_ctx)
        f2val =ndarray.array(np.random.normal(0, 0.5, (32, 32, 3, 3)), executor_ctx)
        f3val = ndarray.array(np.random.normal(0, 0.5, (64, 32, 3, 3)), executor_ctx)
        f4val = ndarray.array(np.random.normal(0, 0.5, (96, 64, 3, 3)), executor_ctx)
        f5_1val = ndarray.array(np.random.normal(0, 0.5, (64, 160, 1, 1)), executor_ctx)
        f5_2val = ndarray.array(np.random.normal(0, 0.5, (96, 64, 3, 3)), executor_ctx)
        f6_1val = ndarray.array(np.random.normal(0, 0.5, (64, 160, 1, 1)), executor_ctx)
        f6_2val =ndarray.array(np.random.normal(0, 0.5, (64, 64, 7, 1)), executor_ctx)
        f6_3val = ndarray.array(np.random.normal(0, 0.5, (64, 64, 1, 7)), executor_ctx)
        f6_4val =ndarray.array(np.random.normal(0, 0.5, (96, 64, 3, 3)), executor_ctx)
        f7val =ndarray.array(np.random.normal(0, 0.5, (192, 192, 3, 3)), executor_ctx)
        W_val = ndarray.array(np.random.normal(0, 0.5, (1536,n_class)), executor_ctx)
        b_val = ndarray.array(np.random.normal(0, 0.5, (n_class)), executor_ctx)
    # stem
        cov1=self.conv2dplusrelu(X, f1, "NCHW", "VALID", 2, 2)
        cov2=self.conv2dplusrelu(cov1, f2, "NCHW", "VALID", 1, 1)
        cov3= self.conv2dplusrelu(cov2, f3, "NCHW", "SAME", 1, 1)
        pool4= self.ad.pooling_2d_forward_op(cov3, "NCHW", "max", 0, 0, 2, 2, 3, 3)
        cov4 = self.conv2dplusrelu(cov3, f4, "NCHW", "VALID", 2, 2)
        concat1=self.ad.concat_forward_op(pool4,cov4)
        cov5_1=self.conv2dplusrelu(concat1, f5_1, "NCHW", "SAME", 1, 1)
        cov5_2= self.conv2dplusrelu(cov5_1, f5_2, "NCHW", "VALID", 1, 1)
        cov6_1 = self.conv2dplusrelu(concat1, f6_1, "NCHW", "SAME", 1, 1)
        cov6_2= self.conv2dplusrelu(cov6_1, f6_2, "NCHW", "SAME", 1, 1)
        cov6_3 = self.conv2dplusrelu(cov6_2, f6_3, "NCHW", "SAME", 1, 1)
        cov6_4 = self.conv2dplusrelu(cov6_3, f6_4, "NCHW", "VALID", 1, 1)
        concat2=self.ad.concat_forward_op(cov5_2,cov6_4)
        cov7= self.conv2dplusrelu(concat2, f7, "NCHW", "VALID", 2, 2)
        pool7=self.ad.pooling_2d_forward_op(concat2, "NCHW", "max", 0, 0, 2, 2, 3, 3)
        concat3=self.ad.concat_forward_op(pool7,cov7)
        a1,dicta1  = self.block_inception_a(concat3,"a1",executor_ctx)
        a2, dicta2 = self.block_inception_a(a1, "a2",executor_ctx)
        a3, dicta3 = self.block_inception_a(a2, "a3",executor_ctx)
        a4, dicta4 = self.block_inception_a(a3, "a4",executor_ctx)

        ra,dictra=self.block_reduction_a(a4,384,"ra",executor_ctx)
        b1, dictb1 = self.block_inception_b(ra, 1024,"b1",executor_ctx)
        b2, dictb2 = self.block_inception_b(b1, 1024, "b2",executor_ctx)
        b3, dictb3 = self.block_inception_b(b2, 1024, "b3",executor_ctx)
        b4, dictb4 = self.block_inception_b(b3, 1024, "b4",executor_ctx)
        b5, dictb5 = self.block_inception_b(b4, 1024, "b5",executor_ctx)
        b6, dictb6 = self.block_inception_b(b5, 1024, "b6",executor_ctx)
        b7, dictb7 = self.block_inception_b(b6, 1024, "b7",executor_ctx)
        #
        rb,dictrb=self.block_reduction_b(b7,1024,"rb",executor_ctx)
        c1, dictc1 = self.block_inception_c(rb, 1536, "c1",executor_ctx)
        c2, dictc2 = self.block_inception_c(c1, 1536, "c2",executor_ctx)
        c3, dictc3 = self.block_inception_c(c2, 1536, "c3",executor_ctx)
        poollast = self.ad.pooling_2d_forward_op(c3, "NCHW", "mean", 0, 0, 1, 1, 8, 8)
        squeeze = self.ad.squeeze_op(poollast)
        drop_out = self.ad.fullydropout_forward_op(squeeze, "NCHW", 0.8)
        dense = self.ad.dense(drop_out, W, b)
        y = self.ad.fullyactivation_forward_op(dense, "NCHW", "softmax")
        loss = self.ad.crossEntropy_loss(y, Y_)
        executor = self.ad.Executor(loss, y, 0.001, top_control_queue=top_control_queue,
                                    top_message_queue=top_message_queue)

        feed_dict = {f1: f1val, f2: f2val, f3: f3val, f4: f4val,f5_1:f5_1val,f5_2:f5_2val,f6_1:f6_1val,f6_2:f6_2val,f6_3:f6_3val,f6_4:f6_4val,f7:f7val,W:W_val,b:b_val}
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
        f1 = open(f"{log_path}/gpu_time.txt", "w+")
        for i in range(self.num_step):
            print("step", i)
            if i==5:
                gpu_record.start()
                start_time = time.time()
            if i==10:
                gpu_record.stop()
                f1.write(f'time_cost:{time.time() - start_time}')
                f1.flush()
                f1.close()
            feed_dict[X] = ndarray.array(X_val, ctx=executor_ctx)
            feed_dict[Y_] = ndarray.array(y_val, ctx=executor_ctx)
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
    batch_size=4
    num_step = 20
    inceptionv4 = Inceptionv4(num_step=num_step, batch_size=batch_size, gpu_num=gpu_num)
    X_val = np.random.normal(loc=0, scale=0.1, size=(batch_size, 3, 299, 299))# number = batch_size  channel = 3  image_size = 224*224

    y_val = np.random.normal(loc=0, scale=0.1, size=(batch_size, 1000)) # n_class = 1000


    p1 = Process(target=inceptionv4.inception_v4, args=(executor_ctx, top_control_queue, top_message_queue, 1000, X_val, y_val))
    p1.start()
    # p1.join()

    top_control_queue2 = multiprocessing.Queue()
    top_control_queue_list.append(top_control_queue2)
    top_message_queue2 = multiprocessing.Queue()
    top_message_queue_list.append(top_message_queue2)
    job_number += 1

    gpu_num = GPU
    batch_size = 4
    num_step = 20
    inceptionv4 = Inceptionv4(num_step=num_step, batch_size=batch_size, gpu_num=gpu_num)
    X_val = np.random.normal(loc=0, scale=0.1,
                             size=(batch_size, 3, 299, 299))  # number = batch_size  channel = 3  image_size = 224*224

    y_val = np.random.normal(loc=0, scale=0.1, size=(batch_size, 1000))  # n_class = 1000

    p2 = Process(target=inceptionv4.inception_v4,
                 args=(executor_ctx, top_control_queue2, top_message_queue2, 1000, X_val, y_val))
    p2.start()


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