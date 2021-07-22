from __future__ import absolute_import
import time
import numpy as np
from pycode.tinyflow import ndarray, gpu_op,capuchinadam
from pycode.tinyflow import autodiff_capu as ad
import os, datetime
from pycode.tinyflow import memoryManagercapu
import queue
from pynvml import *


class TrainExecutor(object):
    """Executor computes values for given set of nodes in computation graph."""


    def __init__(self, targetloss, learning_rate=0.001,need_tosave=None,ctx=ndarray.gpu(0)):
        self.need_tosave=need_tosave
        self.b1 = 0.9
        self.b2 = 0.999
        self.e = 0.00000001
        self.b1t = [0.9]
        self.b2t = [0.999]
        self.targetloss = targetloss
        self.learning_rate = learning_rate
        self.Variable_node_list = get_Variable_node_list(self.targetloss)

        self.Variable_node_list.reverse()
        self.Variable_node_grad_list = ad.gradients(self.targetloss, self.Variable_node_list)#反向node

        self.eval_node_list,self.mv,self.Variable_node_to_mv = getcomputelist(self.Variable_node_list,self.Variable_node_grad_list, self.b1, self.b2, self.b1t, self.b2t, self.e, self.learning_rate)#其内存还是Variable，但是换了个点
        self.ctx = ctx
        # 根据这个topo_order算
        self.topo_order = ad.find_topo_sort(self.eval_node_list)
        self.topo_order = swapadam(self.topo_order)


        #存node的shape
        self.node_to_shape_map = None
        #node和其对应的value，这里value自己定义,或者node本身可以判断状态
        self.node_to_arr_map = {}

        #初始化变量的np
        self.Variable_node_np_value = None

        #计算必要的资源
        self.cudaStream = gpu_op.create_cudaStream()
        self.cudnnHandle = gpu_op.create_cudnnHandle(self.cudaStream)
        self.cublasHandle = gpu_op.create_cublasHandle(self.cudaStream)


        #是否是第一次run
        self.isfirstrun = 0
        self.isc = 0

        self.capu = capuchinadam.capuchin(self.topo_order)
        self.access_index = 0
        self.will_do_queue = queue.Queue()
        self.have_done_queue = queue.Queue()
        self.memoryManager = memoryManagercapu.MemoryManager(self.will_do_queue, self.have_done_queue)
        self.memoryManager.start()
        for i in range(len(self.topo_order)):
            self.topo_order[i].index = i

        #日志记录
        self.start_finish_time = 0
        self.hit_count = 0
        self.swap_count = 0
        self.node_order = []


    def infer_shape(self, feed_shapes):
        """Given shapes of feed_dict nodes, infer shape for all nodes in graph.

        Implementation note:
        Iteratively calls node.op.infer_shape to infer shapes.
        Node shapes stored in self.node_to_shape_map.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        self.node_to_shape_map = dict(feed_shapes)

        for idx, node in enumerate(self.topo_order):
            if node in self.node_to_shape_map:
                mem = 1
                for i in range(0, len(self.node_to_shape_map[node])):
                    mem = mem * self.node_to_shape_map[node][i]
                node.memory = 4 * mem
                continue

            # print(node)
            input_shapes = [self.node_to_shape_map[i] for i in node.inputs]
            assert None not in input_shapes
            self.node_to_shape_map[node] = node.op.infer_shape(node, input_shapes, self.cudnnHandle)
            mem = 1
            for i in range(0, len(self.node_to_shape_map[node])):
                mem = mem * self.node_to_shape_map[node][i]
            node.memory = 4 * mem
            # print(self.node_to_shape_map[node])

    #放出变量的np字典
    def init_Variable(self, feed_dict):
        self.Variable_node_np_value = feed_dict
        for node in self.Variable_node_list:
            nodem = self.Variable_node_to_mv[node][0]
            nodev = self.Variable_node_to_mv[node][1]
            self.Variable_node_np_value[nodem] = np.zeros(feed_dict[node].shape)
            self.Variable_node_np_value[nodev] = np.zeros(feed_dict[node].shape)



    #feed_dict为np数组
    def run(self, feed_dict, Accuracy_node = None ,convert_to_numpy_ret_vals=False):


        if self.isfirstrun == 0:
            endtime = time.time()
            pciin, pciout = gpu_op.testPcie()
            pciin=pciin*1024
            pciout=pciout*1024

            need_tomem=0
            #第一次,把变量一起初始化了
            feed_dict.update(self.Variable_node_np_value)

            # 先确定shape

            #input的shape
            feed_shapes = {}
            for node, value in feed_dict.items():
                feed_shapes[node] = value.shape

            #把shape放进self.node_to_shape_map
            self.infer_shape(feed_shapes)



            for node in self.topo_order:
                node.srcs = list(node.inputs)
                node.swapouttime=node.memory/pciout
                node.swapintime=node.memory/pciin



            #存已经被计算过的node
            node_computed = set()

            # 日志记录
            self.start_finish_time = datetime.datetime.now()
            self.node_order.append("topo_order:")
            for i in range(len(self.topo_order)):
                self.node_order.append("index:" + str(i) + "\t" + self.topo_order[i].name)
            self.node_order.append("\nrun:")

            #开始运行
            for i in range(len(self.topo_order)):
                node = self.topo_order[i]
                self.node_order.append("index:" + str(i) + "\t" + node.name + "\ttime:" + str(datetime.datetime.now()))


                #已经被计算过了
                if node in node_computed:
                    continue
                #是inputs
                if node in feed_dict.keys():
                    #申请空间

                    ret = ndarray.array(feed_dict[node], ctx=self.ctx)
                    if isinstance(ret, int):
                        self.getpeekaccess()
                        need_tomem += ret
                        # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
                        for i in range(len(self.topo_order)):
                            dnode = self.topo_order[i]
                            if self.isevict(dnode, node):  # 不是inputs和本身，删掉
                                self.tensor_evict(dnode)
                                ret = ndarray.array(feed_dict[node], ctx=self.ctx)
                                if not isinstance(ret, int):
                                    break
                    #此时ret为ndarray
                    #value都存在self.node_to_arr_map
                    self.node_to_arr_map[node] = ret
                    node.array_status = 1
                    node_computed.add(node)
                    continue


                #不是SgdOp,申请内存
                if node.issgd == 0:
                    #给这个点申请内存
                    # 申请空间
                    t1=time.time()
                    ret = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx)
                    t2=time.time()
                    if isinstance(ret, int):
                        self.getpeekaccess()
                        need_tomem += ret
                        # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
                        for i in range(len(self.topo_order)):
                            dnode=self.topo_order[i]
                            if self.isevict(dnode, node):  # 不是inputs和本身，删掉
                                self.tensor_evict(dnode)
                                t1 = time.time()
                                ret = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx)
                                t2 = time.time()
                                if not isinstance(ret, int):
                                    break
                    # 此时ret为ndarray
                    # value都存在self.node_to_arr_map
                    self.node_to_arr_map[node] = ret
                    node.rp_time += (t2 - t1)
                    node.array_status = 1
                else:
                    # 是SgdOp,不申请内存
                    self.node_to_arr_map[node] = None

                for n in node.inputs:  # 计算前确保inputs移到GPU
                    if n.array_status == 0:  # on CPU
                        self.will_do_queue.put((n.index, self.node_to_arr_map[n]))
                        n.array_status = 2  # CPU to GPU
                        self.node_order.append("swap_in\t" + "index:" + str(n.index)
                                               + "\t" + n.name + "\ttime:" + str(datetime.datetime.now()))

                for n in node.inputs:
                    while n.array_status != 1:  # 没在GPU上的node取到GPU上
                        self.passive_mode_getusefulnode(n,node)

                #放inputs的ndarray，
                input_vals = []
                for input_node in node.inputs:
                    #此时要保证在gpu中
                    self.tensor_accsess(input_node)
                    input_vals.append(self.node_to_arr_map[input_node])




                #除了SgdOp，其他的点此时要保证在gpu中
                node_val = self.node_to_arr_map[node]

                tic=time.time()
                memorytoSaving = node.op.compute(node, input_vals, node_val, self.cudnnHandle, self.cublasHandle, self.cudaStream)
                toc=time.time()
                if memorytoSaving != 0:
                    self.getpeekaccess()
                    need_tomem += memorytoSaving
                    # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
                    for dnode in self.topo_order:
                        if self.isevict(dnode, node):  # 不是inputs和本身，删掉
                            self.tensor_evict(dnode)
                            tic = time.time()
                            memorytoSaving = node.op.compute(node, input_vals, node_val, self.cudnnHandle,self.cublasHandle, self.cudaStream)
                            toc = time.time()
                            if memorytoSaving == 0:
                                break
                #此点被计算过了
                node_computed.add(node)
                node.rp_time += (toc - tic)
                node.MSPS = node.memory / node.rp_time
                endtime = time.time()


            # 不是第一次了




            #得到ft
            for i in range (len(self.topo_order)):
                node = self.topo_order[i]
                for j in range(0,node.access_count-1):
                    out_id=node.use_access_id[j]
                    in_id=node.use_access_id[j+1]
                    outtime=self.capu.tensor_access_list[out_id][2]
                    intime=self.capu.tensor_access_list[in_id][2]
                    ft=(intime-node.swapintime)-(outtime+node.swapouttime)
                    node.FT.append(ft)
                if(node.access_count>1):
                    out_id = node.use_access_id[node.access_count-1]
                    outtime = self.capu.tensor_access_list[out_id][2]
                    node.FT.append(endtime-(outtime+node.swapouttime))
            if self.need_tosave==None:
               self.capu.hybrid_policy(need_tomem,endtime)
            else:
               self.capu.hybrid_policy(self.need_tosave,endtime,True)


            #把结果输出了： [loss,变量按网络顺序],这里只是输出value，并不保证一定在gpu中
            #但是如果这里value是None的话，他会报错
            result_output = [self.node_to_arr_map[self.targetloss]]
            re_var = []
            for node in self.Variable_node_list:
                re_var.append(self.node_to_arr_map[node])
            re_var.reverse()
            result_output = result_output + re_var
            #结果，计算正确率
            if Accuracy_node !=None:
                result_output.append(self.node_to_arr_map[Accuracy_node])
            # print(self.capu.policy)
            # print(self.need_tosave)
            # print(self.capu.prior_policy)


            #adam更新参数
            self.b1t[0] = self.b1t[0] * self.b1
            self.b2t[0] = self.b2t[0] * self.b2
            self.clear()
            self.isfirstrun = 1
            return result_output



        else:

            node_computed = set()
            # 日志记录
            self.node_order.append("\nrun:")

            # 开始运行
            for i in range(len(self.topo_order)):
                # print(i,self.topo_order[i].name)
                node = self.topo_order[i]
                self.node_order.append("index:" + str(i) + "\t" + node.name + "\ttime:" + str(datetime.datetime.now()))
                # 已经被计算过了
                if node in node_computed:
                    continue
                # 是inputs
                if node in feed_dict.keys():

                    #如果此时在gpu中,把np的值赋值过去
                    if ndarray.is_gpu_ctx(self.node_to_arr_map[node].ctx):
                        self.node_to_arr_map[node]._sync_copyfrom(feed_dict[node])
                    else:
                        # 没在GPU中,重新在GPU申请空间
                        ret = ndarray.array(feed_dict[node], ctx=self.ctx)
                        if isinstance(ret, int):
                            # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
                            for i in range(len(self.topo_order)):
                                dnode = self.topo_order[i]
                                if self.isevict(dnode, node):  # 不是inputs和本身，删掉
                                    self.tensor_evict(dnode)
                                    ret = ndarray.array(feed_dict[node], ctx=self.ctx)
                                    if not isinstance(ret, int):
                                        break
                        self.node_to_arr_map[node] = ret
                        node.array_status = 1  # on GPU

                    node_computed.add(node)
                    continue

                # 如果node是变量，不用管
                if node in self.Variable_node_list:
                    if ndarray.is_gpu_ctx(self.node_to_arr_map[node].ctx):
                        continue
                    else:
                        # 没在GPU中,重新在GPU申请空间
                        ret = ndarray.array(self.node_to_arr_map[node].asnumpy(), ctx=self.ctx)
                        if isinstance(ret, int):
                            # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
                            for i in range(len(self.topo_order)):
                                dnode = self.topo_order[i]
                                if self.isevict(dnode, node):  # 不是inputs和本身，删掉
                                    self.tensor_evict(dnode)
                                    ret = ndarray.array(self.node_to_arr_map[node].asnumpy(), ctx=self.ctx)
                                    if not isinstance(ret, int):
                                        break
                        self.node_to_arr_map[node] = ret
                        node.array_status = 1  # on GPU
                    node_computed.add(node)
                    continue


                # 如果node是adam要用的变量，不用管
                if node in self.mv:
                    if ndarray.is_gpu_ctx(self.node_to_arr_map[node].ctx):
                        continue
                    else:
                        # 没在GPU中,重新在GPU申请空间
                        ret = ndarray.array(self.node_to_arr_map[node].asnumpy(), ctx=self.ctx)
                        if isinstance(ret, int):
                            # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
                            for i in range(len(self.topo_order)):
                                dnode = self.topo_order[i]
                                if self.isevict(dnode, node):  # 不是inputs和本身，删掉
                                    self.tensor_evict(dnode)
                                    ret = ndarray.array(self.node_to_arr_map[node].asnumpy(), ctx=self.ctx)
                                    if not isinstance(ret, int):
                                        break
                        self.node_to_arr_map[node] = ret
                        node.array_status = 1  # on GPU
                    node_computed.add(node)
                    continue




                #不是sgdop的中间点
                if node.issgd == 0:
                    ret = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx)
                    if isinstance(ret, int):
                        # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
                        for i in range(len(self.topo_order)):
                            dnode = self.topo_order[i]
                            if self.isevict(dnode, node):  # 不是inputs和本身，删掉
                                self.tensor_evict(dnode)
                                ret = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx)
                                if not isinstance(ret, int):
                                    break
                    # 此时ret为ndarray
                    # value都存在self.node_to_arr_map
                    self.node_to_arr_map[node] = ret
                    node.array_status = 1



                swapout_node=[]
                recompute_node=[]
                for input_node in node.inputs:
                    prior_policy=self.prior_policy_run(input_node,node)
                    policy=self.policy_run(node.inputs)
                    if policy==1:
                        swapout_node.append(self.topo_order[self.capu.swap[self.access_index-1]])
                    if prior_policy==1:
                        swapout_node.append(input_node)
                    elif prior_policy==3:
                        recompute_node.append(input_node)

                for n in node.inputs:  # 计算前确保inputs移到GPU
                    if n.array_status == 0:  # on CPU
                        self.will_do_queue.put((n.index, self.node_to_arr_map[n]))
                        n.array_status = 2  # CPU to GPU
                        self.node_order.append("swap_in\t" + "index:" + str(n.index)
                                               + "\t" + n.name + "\ttime:" + str(datetime.datetime.now()))


                for n in node.inputs:
                    while n.array_status != 1:  # 被动模式下没在GPU上的node取到GPU上
                        self.passive_mode_getusefulnode(n,node)  #如果发生swapin失败（gpu内存不够），同样需要重新swap，算一次不命中


                input_vals = []
                for input_node in node.inputs:
                    # 此时要保证在gpu中
                    input_vals.append(self.node_to_arr_map[input_node])

                #计算前swap，隐藏开销
                for n in swapout_node:
                    if n.array_status == 1:  # on GPU
                        self.will_do_queue.put((n.index, self.node_to_arr_map[n]))
                        n.array_status = 3  # GPU to CPU
                        self.node_to_arr_map[n] = None
                        self.node_order.append("swap_out\t" + "index:" + str(n.index)
                                               + "\t" + n.name + "\ttime:" + str(datetime.datetime.now()))
                #选择重计算的节点
                for n in swapout_node:
                    if n.array_status == 1:  # on GPU
                        n.array_status = 4  # free
                        self.node_to_arr_map[n] = None
                        self.node_order.append("free\t" + "index:" + str(n.index)
                                               + "\t" + n.name + "\ttime:" + str(datetime.datetime.now()))


                node_val = self.node_to_arr_map[node]
                memorytoSaving = node.op.compute(node, input_vals, node_val, self.cudnnHandle, self.cublasHandle, self.cudaStream)
                if memorytoSaving != 0:
                    # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
                    for dnode in self.topo_order:
                        if self.isevict(dnode, node):  # 不是inputs和本身，删掉
                            self.tensor_evict(dnode)
                            memorytoSaving = node.op.compute(node, input_vals, node_val, self.cudnnHandle,self.cublasHandle, self.cudaStream)
                            if memorytoSaving == 0:
                                break
                # 此点被计算过了
                node_computed.add(node)
            # 把结果输出了： [loss,变量按网络顺序],这里只是输出value，并不保证一定在gpu中
            # 但是如果这里value是None的话，他会报错
            result_output = [self.node_to_arr_map[self.targetloss]]
            re_var = []
            for node in self.Variable_node_list:
                re_var.append(self.node_to_arr_map[node])
            re_var.reverse()
            result_output = result_output + re_var
            # 结果，计算正确率
            if Accuracy_node != None:
                result_output.append(self.node_to_arr_map[Accuracy_node])

            # adam更新参数
            self.b1t[0] = self.b1t[0] * self.b1
            self.b2t[0] = self.b2t[0] * self.b2

            self.access_index=0
            # print((self.swap_count,self.hit_count))
            self.clear()
            return result_output


    def clear(self):
        for node in self.topo_order:
            if node.array_status == 1:
                if node.isw == 0:
                    self.node_to_arr_map[node] = None
                else:
                    if self.node_to_arr_map[node] is not None:
                        self.tensor_evict(node)
            elif node.array_status == 3:
                while node.array_status != 0:
                    (node_index, node_val) = self.have_done_queue.get(block=True)
                    self.node_to_arr_map[self.topo_order[node_index]] = node_val
                    if ndarray.is_gpu_ctx(node_val.ctx):
                        self.topo_order[node_index].array_status = 1
                    else:
                        self.topo_order[node_index].array_status = 0

    def policy_run(self,inputnodes):
        policy=self.capu.policy[self.access_index]
        if policy==0 or policy==3 or policy==4:
            pass
        elif policy==1:
            pass
        else:
           swap_id = self.capu.swap[self.access_index]
           swap_node = self.topo_order[swap_id]
           if swap_node.array_status==3:
               self.arrive_to_cpu(swap_node)
           if swap_node.array_status==0:
               if swap_node in inputnodes:
                   self.hit_count+=1               #当前策略在运算前才swapin，算一次不命中
               self.swap_count += 1
               self.will_do_queue.put((swap_id, self.node_to_arr_map[swap_node]))
               self.node_order.append("swap_in\t" + "index:" + str(swap_id)
                                      + "\t" + swap_node.name + "\ttime:" + str(datetime.datetime.now()))
               swap_node.array_status = 2
        self.access_index+=1
        return  policy


    #无法在策略中掩藏开销的，进行该操作
    def prior_policy_run(self,input_node,node):
        prior_policy=self.capu.prior_policy[self.access_index]
        if prior_policy==0:
            pass
        elif prior_policy==1:
            pass
        elif prior_policy==2:
            if input_node.array_status == 3:
                self.arrive_to_cpu(input_node)
            if input_node.array_status==0:
                self.swap_count+=1
                self.hit_count+=1                    #无法掩藏开销，不命中
                self.will_do_queue.put((input_node.index, self.node_to_arr_map[input_node]))
                self.node_order.append("swap_in\t" + "index:" + str(input_node.index)
                                       + "\t" + input_node.name + "\ttime:" + str(
                    datetime.datetime.now()))
                input_node.array_status = 2
        elif prior_policy==3:
             pass
        elif prior_policy==4:
            self.recompute(input_node,node)
            self.node_order.append("finish_recompute\t"
                                   + "index:" + str(input_node.index) + "\t" + input_node.name
                                   + "\ttime:" + str(datetime.datetime.now()))

        return  prior_policy



    def get_start_finish_time(self):
        return self.start_finish_time

    def get_hit(self):
        return self.swap_count-self.hit_count, self.swap_count

    def get_node_order(self):
        return self.node_order

    def isevict(self,dnode,node):
        if (dnode not in node.inputs) and dnode != node and dnode.array_status == 1 :
            return True
        else :
            return False


    def recompute(self,rep_node,node):

        #申请重算结果地址
        ret = ndarray.empty(self.node_to_shape_map[rep_node], ctx=self.ctx)
        if isinstance(ret, int):
            for i in range(len(self.topo_order)):
                dnode = self.topo_order[i]
                if self.isevict(dnode, node) and self.isevict(dnode, rep_node):  # 不是inputs和本身，删掉
                    self.tensor_evict(dnode)
                    ret = ndarray.empty(self.node_to_shape_map[rep_node], ctx=self.ctx)
                    if not isinstance(ret, int):
                        break
        self.node_to_arr_map[rep_node] = ret
        rep_node.array_status = 1

        #得到gpu上的输入
        input_vals = []
        for n in rep_node.inputs:
            if n.array_status==1:
                pass
            elif n.array_status==0 or n.array_status==2:
                while n.array_status!=1:
                    self.passive_mode_getusefulnode(n,node)
            elif n.array_status==3:
                self.arrive_to_cpu(n)

                while n.array_status!=1:
                    self.passive_mode_getusefulnode(n,node)
                    self.node_order.append("finish_recompute\t"
                                           + "index:" + str(n.index) + "\t" + n.name
                                           + "\ttime:" + str(datetime.datetime.now()))
            elif n.array_status==4:
                self.recompute(n,rep_node)

        for n in rep_node.inputs:     #全在gpu上
            input_vals.append(n)

        node_val = self.node_to_arr_map[rep_node]
        memorytoSaving = node.op.compute(node, input_vals, node_val, self.cudnnHandle, self.cublasHandle, self.cudaStream)
        if memorytoSaving != 0:
            # 这里被动模式
            for dnode in self.topo_order:
                if self.isevict(dnode,node):  # 不是inputs和本身，删掉
                    self.tensor_evict(dnode)
                    memorytoSaving = node.op.compute(node, input_vals, node_val, self.cudnnHandle, self.cublasHandle,self.cudaStream)
                    if memorytoSaving == 0:
                        break


    def arrive_to_cpu(self,n):
        while n.array_status != 0:
            (node_index, node_val) = self.have_done_queue.get(block=True)
            if isinstance(node_val, int):
                self.topo_order[node_index].array_status = 0
                continue
            self.node_to_arr_map[self.topo_order[node_index]] = node_val
            if ndarray.is_gpu_ctx(node_val.ctx):
                self.topo_order[node_index].array_status = 1
                self.node_order.append("finish_swap_in\t"
                                       + "index:" + str(node_index) + "\t" + self.topo_order[
                                           node_index].name
                                       + "\ttime:" + str(datetime.datetime.now()))
            else:
                self.topo_order[node_index].array_status = 0
                self.node_order.append("finish_swap_out\t"
                                       + "index:" + str(node_index) + "\t" + self.topo_order[node_index].name
                                       + "\ttime:" + str(datetime.datetime.now()))


    def passive_mode_getusefulnode(self,n,node):
        if n.array_status == 0:        #可能因为超内存swapin失败,重新swap in
            self.hit_count += 1
            self.will_do_queue.put((n.index, self.node_to_arr_map[n]))
            n.array_status=2
            self.node_order.append("swap_in\t" + "index:" + str(n.index)
                                   + "\t" + n.name + "\ttime:" + str(datetime.datetime.now()))
        (node_index, node_val) = self.have_done_queue.get(block=True)  # get the node from GPU
        if isinstance(node_val, int):
            if self.isfirstrun==0:
                self.getpeekaccess()
            for dnode in self.topo_order:
                if self.isevict(dnode, node):
                    self.tensor_evict(dnode)
                    node_val -= dnode.memory
                    if node_val < 0:
                        break
            self.will_do_queue.put((node_index, self.node_to_arr_map[self.topo_order[node_index]]))
            self.node_order.append("swap_in\t" + "index:" + str(node_index)
                                   + "\t" + self.topo_order[node_index].name + "\ttime:" + str(datetime.datetime.now()))
            self.topo_order[node_index].array_status = 2
            return
        self.node_to_arr_map[self.topo_order[node_index]] = node_val
        if ndarray.is_gpu_ctx(node_val.ctx):
            self.topo_order[node_index].array_status = 1
            self.node_order.append("finish_swap_in\t"
                                   + "index:" + str(node_index) + "\t" + self.topo_order[
                                       node_index].name
                                   + "\ttime:" + str(datetime.datetime.now()))
        else:
            self.topo_order[node_index].array_status = 0
            self.node_order.append("finish_swap_out\t"
                                   + "index:" + str(node_index) + "\t" + self.topo_order[node_index].name
                                   + "\ttime:" + str(datetime.datetime.now()))


    def tensor_evict(self, access_node):
        self.will_do_queue.put((access_node.index, self.node_to_arr_map[access_node]))
        self.node_to_arr_map[access_node] = None
        access_node.array_status = 3
        self.arrive_to_cpu(access_node)



    def tensor_accsess(self,access_node):
        access_node.access_count += 1
        self.capu.add_tensor_access_info(access_node.index,access_node.access_count,time.time())
        access_node.use_access_id.append(len(self.capu.tensor_access_list)-1)

    def getpeekaccess(self):
        for i in range(len(self.topo_order)):
            if(self.topo_order[i].array_status==1):
                self.topo_order[i].peekaccess.append(self.topo_order[i].access_count)



def get_Variable_node_list(node):

    visited = set()
    Variable_order = []
    Variable_sort_dfs(node, visited, Variable_order)
    return Variable_order



def Variable_sort_dfs(node, visited, Variable_order):
    """Post-order DFS"""

    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        Variable_sort_dfs(n, visited, Variable_order)

    if node.isw == 1:
        Variable_order.append(node)



def getcomputelist(Variable_node_list, Variable_node_grad_list, b1, b2, b1t, b2t, e,learning_rate):

    computelist = []
    mv = []
    Variable_node_to_mv = {}
    for i in range(len(Variable_node_list)):
        m = ad.Variable(Variable_node_list[i].name+'m')
        v = ad.Variable(Variable_node_list[i].name+'v')
        mv.append(m)
        mv.append(v)
        Variable_node_to_mv[Variable_node_list[i]] = (m,v)
        adamnode = ad.adam_op(Variable_node_list[i],m,v,Variable_node_grad_list[i], b1, b2, b1t, b2t, e, learning_rate)
        adamnode.issgd = 1#代表不用为这个点加内存
        computelist.append(adamnode)

    return computelist,mv,Variable_node_to_mv




def swapadam(topoorder):
    for i in range(len(topoorder)):
        if topoorder[i].issgd == 1 and topoorder[i].isw == 0:
            topoorder[i].isw = 3
            filter = topoorder[i].inputs[0]
            j = len(topoorder) - 1
            while j > i:
                if topoorder[j].issgd == 1:
                    j = j - 1
                    continue
                if filter in topoorder[j].inputs:

                    break
                j = j - 1

            tmp = topoorder[i]
            topoorder.remove(tmp)
            topoorder.insert(j,tmp)
    # for i in range(len(topoorder)):
    #     print(i,topoorder[i])
    return topoorder







