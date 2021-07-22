import numpy as np
import imp, os,datetime

from pycode.tinyflow import ndarray
from tests.Experiment import record_GPU
from multiprocessing import Process
tinyflow_path = "../../pycode/tinyflow/"

class Inceptionv3(Process):
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
        self.type = type
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


    def conv2dplusrelu(self,node,filter,model,type,stride_h,stride_w):
        node_new=self.ad.convolution_2d_forward_op(node, filter, model, type, stride_h,stride_w)
        node_after=self.ad.activation_forward_op(node_new, model, "relu")
        return  node_after
    def inception_v3(self):
        X = self.ad.Placeholder("inputs")
        y_ = self.ad.Placeholder("y_")
        filterb_1 = self.ad.Variable("filterb_1")
        filterb_2 = self.ad.Variable("filterb_2")
        filterb_3 = self.ad.Variable("filterb_3")
        filterb_4 = self.ad.Variable("filterb_4")
        filterb_5= self.ad.Variable("filterb_5")

        filtersb_val1 = np.ones((32,3,3,3)) * 0.01
        filtersb_val2 = np.ones((32,32,3,3)) * 0.01
        filtersb_val3 = np.ones((64,32,3,3)) * 0.01
        filtersb_val4 = np.ones((80, 64, 1, 1)) * 0.01
        filtersb_val5 = np.ones((192, 80, 3, 3)) * 0.01

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

        filter1_1_0_val = np.ones((64, 192, 1, 1)) * 0.01
        filter1_1_1_vala = np.ones((48, 192, 1, 1)) * 0.01
        filter1_1_1_valb = np.ones((64, 48, 5, 5)) * 0.01
        filter1_1_2_vala = np.ones((64, 192, 1, 1)) * 0.01
        filter1_1_2_valb = np.ones((96, 64, 3, 3)) * 0.01
        filter1_1_2_valc = np.ones((96, 96, 3, 3)) * 0.01
        filter1_1_3_val = np.ones((32, 192, 1, 1)) * 0.01

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

        filter1_2_0_val = np.ones((64, 256, 1, 1)) * 0.01
        filter1_2_1_vala = np.ones((48, 256, 1, 1)) * 0.01
        filter1_2_1_valb = np.ones((64, 48, 5, 5)) * 0.01
        filter1_2_2_vala = np.ones((64, 256, 1, 1)) * 0.01
        filter1_2_2_valb = np.ones((96, 64, 3, 3)) * 0.01
        filter1_2_2_valc = np.ones((96, 96, 3, 3)) * 0.01
        filter1_2_3_val = np.ones((64, 256, 1, 1)) * 0.01

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

        filter1_3_0_val = np.ones((64, 288, 1, 1)) * 0.01
        filter1_3_1_vala = np.ones((48, 288, 1, 1)) * 0.01
        filter1_3_1_valb = np.ones((64, 48, 5, 5)) * 0.01
        filter1_3_2_vala = np.ones((64, 288, 1, 1)) * 0.01
        filter1_3_2_valb = np.ones((96, 64, 3, 3)) * 0.01
        filter1_3_2_valc = np.ones((96, 96, 3, 3)) * 0.01
        filter1_3_3_val = np.ones((64, 288, 1, 1)) * 0.01

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

        filter2_1_0_val = np.ones((384, 288, 3, 3)) * 0.01
        filter2_1_1_vala = np.ones((64, 288, 1, 1)) * 0.01
        filter2_1_1_valb = np.ones((96, 64, 3, 3)) * 0.01
        filter2_1_1_valc = np.ones((96, 96, 3, 3)) * 0.01

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

        filter2_2_0_val = np.ones((192, 768, 1, 1)) * 0.01
        filter2_2_1_vala = np.ones((128, 768, 1, 1)) * 0.01
        filter2_2_1_valb = np.ones((128, 128, 1, 7)) * 0.01
        filter2_2_1_valc = np.ones((192, 128, 7, 1)) * 0.01
        filter2_2_2_vala = np.ones((128, 768, 1, 1)) * 0.01
        filter2_2_2_valb = np.ones((128, 128, 7,1)) * 0.01
        filter2_2_2_valc = np.ones((128, 128, 1, 7)) * 0.01
        filter2_2_2_vald = np.ones((128, 128, 7, 1)) * 0.01
        filter2_2_2_vale = np.ones((192, 128, 1, 7)) * 0.01
        filter2_2_3_val = np.ones((192, 768, 1, 1)) * 0.01

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

        filter2_3_0_val = np.ones((192, 768, 1, 1)) * 0.01
        filter2_3_1_vala = np.ones((160, 768, 1, 1)) * 0.01
        filter2_3_1_valb = np.ones((160, 160, 1, 7)) * 0.01
        filter2_3_1_valc = np.ones((192, 160, 7, 1)) * 0.01
        filter2_3_2_vala = np.ones((160, 768, 1, 1)) * 0.01
        filter2_3_2_valb = np.ones((160, 160, 7,1)) * 0.01
        filter2_3_2_valc = np.ones((160, 160, 1, 7)) * 0.01
        filter2_3_2_vald = np.ones((160, 160, 7, 1)) * 0.01
        filter2_3_2_vale = np.ones((192, 160, 1, 7)) * 0.01
        filter2_3_3_val = np.ones((192, 768, 1, 1)) * 0.01

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

        filter2_4_0_val = np.ones((192, 768, 1, 1)) * 0.01
        filter2_4_1_vala = np.ones((160, 768, 1, 1)) * 0.01
        filter2_4_1_valb = np.ones((160, 160, 1, 7)) * 0.01
        filter2_4_1_valc = np.ones((192, 160, 7, 1)) * 0.01
        filter2_4_2_vala = np.ones((160, 768, 1, 1)) * 0.01
        filter2_4_2_valb = np.ones((160, 160, 7, 1)) * 0.01
        filter2_4_2_valc = np.ones((160, 160, 1, 7)) * 0.01
        filter2_4_2_vald = np.ones((160, 160, 7, 1)) * 0.01
        filter2_4_2_vale = np.ones((192, 160, 1, 7)) * 0.01
        filter2_4_3_val = np.ones((192, 768, 1, 1)) * 0.01

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

        filter2_5_0_val = np.ones((192, 768, 1, 1)) * 0.01
        filter2_5_1_vala = np.ones((160, 768, 1, 1)) * 0.01
        filter2_5_1_valb = np.ones((160, 160, 1, 7)) * 0.01
        filter2_5_1_valc = np.ones((192, 160, 7, 1)) * 0.01
        filter2_5_2_vala = np.ones((160, 768, 1, 1)) * 0.01
        filter2_5_2_valb = np.ones((160, 160, 7, 1)) * 0.01
        filter2_5_2_valc = np.ones((160, 160, 1, 7)) * 0.01
        filter2_5_2_vald = np.ones((160, 160, 7, 1)) * 0.01
        filter2_5_2_vale = np.ones((192, 160, 1, 7)) * 0.01
        filter2_5_3_val = np.ones((192, 768, 1, 1)) * 0.01

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

        filter3_1_0_vala = np.ones((192, 768, 1, 1)) * 0.01
        filter3_1_0_valb = np.ones((320, 192, 3, 3)) * 0.01
        filter3_1_1_vala = np.ones((192, 768, 1, 1)) * 0.01
        filter3_1_1_valb = np.ones((192, 192, 1,7)) * 0.01
        filter3_1_1_valc = np.ones((192, 192, 7, 1)) * 0.01
        filter3_1_1_vald = np.ones((192, 192, 3, 3)) * 0.01

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

        filter3_2_0_val = np.ones((320, 1280, 1, 1)) * 0.01
        filter3_2_1_vala = np.ones((384, 1280, 1, 1)) * 0.01
        filter3_2_1_valb = np.ones((384, 384, 1, 3)) * 0.01
        filter3_2_1_valc = np.ones((384, 384, 3, 1)) * 0.01
        filter3_2_2_vala = np.ones((448, 1280, 1, 1)) * 0.01
        filter3_2_2_valb = np.ones((384, 448, 3, 3)) * 0.01
        filter3_2_2_valc = np.ones((384, 384, 1, 3)) * 0.01
        filter3_2_2_vald = np.ones((384, 384, 3, 1)) * 0.01
        filter3_2_3_val = np.ones((192, 1280, 1, 1)) * 0.01

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

        filter3_3_0_val = np.ones((320, 2048, 1, 1)) * 0.01
        filter3_3_1_vala = np.ones((384, 2048, 1, 1)) * 0.01
        filter3_3_1_valb = np.ones((384, 384, 1, 3)) * 0.01
        filter3_3_1_valc = np.ones((384, 384, 3, 1)) * 0.01
        filter3_3_2_vala = np.ones((448, 2048, 1, 1)) * 0.01
        filter3_3_2_valb = np.ones((384, 448, 3, 3)) * 0.01
        filter3_3_2_valc = np.ones((384, 384, 1, 3)) * 0.01
        filter3_3_2_vald = np.ones((384, 384, 3, 1)) * 0.01
        filter3_3_3_val = np.ones((192, 2048, 1, 1)) * 0.01

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
        filtera1val = np.ones((1000, 2048, 1, 1)) * 0.01

        W = self.ad.Variable("filtersmul")
        W_val = np.ones((1000, 1000)) * 0.01

        b= self.ad.Variable("biases")
        b_val = np.ones((1000)) * 0.01


        poollast=self.ad.pooling_2d_forward_op(concat3_3,"NCHW", "mean",0,0,1,1,8,8)
        dropout=self.ad.dropout_forward_op(poollast,"NCHW",0.8)
        convlast=self.conv2dplusrelu(dropout, filtera1, "NCHW", "SAME", 1, 1)
        squeeze=self.ad.squeeze_op(convlast)

        dense = self.ad.dense(squeeze, W, b)
        y = self.ad.fullyactivation_forward_op(dense, "NCHW", "softmax")
        loss = self.ad.crossEntropy_loss(y, y_)
        # fc8

        X_val = np.empty(shape=(self.batch_size, 3, 299, 299), dtype=np.float32)
        y_val = np.empty(shape=(self.batch_size, 1000), dtype=np.float32)
        aph = 0.001
        if self.is_capu == True and self.need_tosave != None:
            t = self.TrainExecute.TrainExecutor(loss, aph, self.need_tosave)
        else:
            t = self.TrainExecute.TrainExecutor(loss, aph)
        t.init_Variable(
            { filterb_1: filtersb_val1, filterb_2: filtersb_val2, filterb_3: filtersb_val3
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
            ,filtera1:filtera1val,W:W_val,b:b_val})
        start_time = datetime.datetime.now()
        for i in range(self.num_step):
            # time1 = datetime.datetime.now()
            t.run({X: X_val, y_: y_val})
            # time2 = datetime.datetime.now()
            # print("epoch", i + 1, "use", time2 - time1
            #       , "\tstart", time1, "\tend", time2, file=self.f1)
            print("InceptionV3 num_step", i)
        start_finish_time = t.get_start_finish_time()
        print((start_finish_time-start_time).microseconds, file=self.f3)
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

        return 0
    def run(self):
        if self.need_tosave != 0:
            outspace = []
            arr_size = self.need_tosave * 1e6 / 4
            gctx = ndarray.gpu(0)
            while arr_size > 0:
                if arr_size > 10000 * 10000:
                    outspace.append(ndarray.array(np.ones((10000, 10000)) * 0.01, ctx=gctx))
                    arr_size -= 10000 * 10000
                else:
                    need_sqrt = int(pow(arr_size, 0.5))
                    if need_sqrt <= 0:
                        break
                    outspace.append(ndarray.array(np.ones((need_sqrt, need_sqrt)) * 0.01, ctx=gctx))
                    arr_size -= need_sqrt * need_sqrt
        record = record_GPU.record("InceptionV3", self.type, self.gpu_num, self.path, self.file_name)
        record.start()
        print("InceptionV3" + " type" + str(self.type) + " start")

        self.inception_v3()
        print("InceptionV3" + " type" + str(self.type) + " finish")
        record.stop()
        if self.need_tosave != 0:
            for i in range(len(outspace) - 1, -1, -1):
                outspace.pop(i)
