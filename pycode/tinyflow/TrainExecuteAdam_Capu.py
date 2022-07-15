from __future__ import absolute_import
import time
from typing import List

from pynvml import nvmlDeviceGetPcieThroughput, nvmlInit, nvmlDeviceGetHandleByIndex, NVML_PCIE_UTIL_COUNT
import numpy as np
from pycode.tinyflow import ndarray, gpu_op, capuchinadam
from pycode.tinyflow import autodiff_capu as ad
import datetime
import queue
import threading
import ctypes
from pycode.tinyflow.autodiff_capu import Node
import gc

index_to_cpu_map = {}
index_to_cpu_flag = {}
index_to_gpu_map = {}
topo_order = []
swap_in_id = 0
swap_in_flag = True
maxmem = 0
# abnormal_passive_cost0 = 0
# abnormal_passive_cost1 = 0
# abnormal_passive_cost2 = 0
# abnormal_passive_cost3 = 0
# abnormal_passive_cost4 = 0
# abnormal_passive_cost_compute = 0
# recompute_cost = 0
# wait_for_arriving_to_cpu = 0
# first_passive_cost = 0
# run_policy_cost = 0
# after_input_node_cost = 0


class MemoryManager(threading.Thread):
    def __init__(self, will_do_queue, have_done_queue):
        threading.Thread.__init__(self)
        self.will_do_queue = will_do_queue
        self.have_done_queue = have_done_queue
        # todo hard code with device id again, may need to change
        self.cpu_ctx = ndarray.cpu(0)
        self.gpu_ctx = ndarray.gpu(0)
        self.cudaSwapStream = gpu_op.create_cudaStream()

    def run(self):
        while (True):
            node = self.will_do_queue.get(block=True)
            node_index = node[0]
            move_to_gpu = node[1]
            node_ndarray_new = None

            global index_to_cpu_map
            global index_to_gpu_map
            global swap_in_flag
            global swap_in_id
            global maxmem
            global topo_order

            if move_to_gpu == 0:
                if topo_order[node_index].array_status!=0:
                    node_ndarray = index_to_gpu_map[node_index]
                    if node_ndarray is None:
                        continue
                    ndarray_cpu = index_to_cpu_map[node_index]
                    if ndarray_cpu.data_id != node_ndarray.data_id:
                        node_ndarray.copyto(index_to_cpu_map[node_index], self.cudaSwapStream)
                    index_to_cpu_flag[node_index] = True
                    # index_to_gpu_map[node_index].free_gpu()
                    index_to_gpu_map[node_index] = None
                    topo_order[node_index].array_status = 0

            else:
                if topo_order[node_index].array_status!=1:
                    swap_in_flag = True
                    swap_in_id = node_index
                    node_ndarray = index_to_cpu_map[node_index]
                    # time1 = datetime.datetime.now()
                    node_ndarray_new = ndarray.empty(node_ndarray.shape, self.gpu_ctx, maxmem=maxmem,
                                                     nowmem=topo_order[node_index].memory)
                    # time2 = datetime.datetime.now()
                    if isinstance(node_ndarray_new, int):
                        swap_in_flag = False
                    else:
                        if index_to_gpu_map[node_index] is None:
                            node_ndarray.copyto(node_ndarray_new, self.cudaSwapStream)
                            index_to_gpu_map[node_index] = node_ndarray_new
                            index_to_cpu_flag[node_index] = False
                            topo_order[node_index].array_status = 1
                    # else:
                    #     print("swap in 和 passive import 重合")


class TrainExecutor(object):
    """Executor computes values for given set of nodes in computation graph."""

    def __init__(self, targetloss, learning_rate=0.001, maxmem=None, ctx=ndarray.gpu(0), schedule=True):
        self.schedule = schedule
        if not schedule:
            self.maxmem = -1
        else:
            self.maxmem = maxmem * 1024 * 1024
        self.b1 = 0.9
        self.b2 = 0.999
        self.e = 0.00000001
        self.b1t = [0.9]
        self.b2t = [0.999]
        self.targetloss = targetloss
        self.learning_rate = learning_rate
        self.Variable_node_list = get_Variable_node_list(self.targetloss)

        self.Variable_node_list.reverse()
        self.Variable_node_grad_list = ad.gradients(self.targetloss, self.Variable_node_list)  # 反向node

        self.eval_node_list, self.mv, self.Variable_node_to_mv = getcomputelist(self.Variable_node_list,
                                                                                self.Variable_node_grad_list, self.b1,
                                                                                self.b2, self.b1t, self.b2t, self.e,
                                                                                self.learning_rate)  # 其内存还是Variable，但是换了个点
        self.ctx = ctx
        # 根据这个topo_order算
        self.topo_order = ad.find_topo_sort(self.eval_node_list)
        self.topo_order = swap_adam(self.topo_order)
        # self.size_order = sorted(self.topo_order, key=lambda x :x.memory)

        # 存node的shape
        self.node_to_shape_map = None
        # node和其对应的value，这里value自己定义,或者node本身可以判断状态

        # 初始化变量的np
        self.Variable_node_np_value = None

        # 计算必要的资源
        self.cudaStream = gpu_op.create_cudaStream()
        self.cudnnHandle = gpu_op.create_cudnnHandle(self.cudaStream)
        self.cublasHandle = gpu_op.create_cublasHandle(self.cudaStream)

        # 是否是第一次run
        self.isfirstrun = 0
        self.isc = 0

        self.access_index = 0
        self.will_do_queue = queue.Queue()
        self.have_done_queue = queue.Queue()
        self.memoryManager = MemoryManager(self.will_do_queue, self.have_done_queue)
        self.memoryManager.setDaemon(True)
        self.memoryManager.start()
        for i in range(len(self.topo_order)):
            self.topo_order[i].index = i
            node = self.topo_order[i]
            if node.name == "FullyDropoutBackward" or node.name == "DropoutBackward" or node.name == "BNBackward" or node.name == "FullyBNBackward":
                node.isdrop = 1
                # 日志记录
        # for i in range(len(self.topo_order)):
        #     self.topo_order[i].index = i
        #     node=self.topo_order[i]
        #     if node.name == "FullyDropoutBackward" or node.name == "DropoutBackward":
        #         node.isdrop=1
        self.capu = capuchinadam.capuchin(self.topo_order)
        self.start_finish_time = 0
        self.hit_count = 0
        self.swap_count = 0
        self.node_order = []
        self.outspace = []
        self.ctx_cpu = ndarray.cpu(0)
        self.reflush_access = []
        self.peakaccess_idx = set()

        self.reflush_cost = 0
        self.abnormal_passive_cost0 = 0
        self.abnormal_passive_cost1 = 0
        self.abnormal_passive_cost2 = 0
        self.abnormal_passive_cost3 = 0
        self.abnormal_passive_cost4 = 0
        self.recompute_cost = 0
        self.wait_for_arriving_to_cpu = 0
        self.wait_for_cpu2gpu_just_after_gpu2cpu = 0
        self.first_passive_cost = 0
        self.abnormal_passive_cost_compute = 0
        self.run_policy_cost = 0
        self.after_input_node_cost = 0

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

            input_shapes = [self.node_to_shape_map[i] for i in node.inputs]
            assert None not in input_shapes
            self.node_to_shape_map[node] = node.op.infer_shape(node, input_shapes, self.cudnnHandle)
            mem = 1
            for i in range(0, len(self.node_to_shape_map[node])):
                mem = mem * self.node_to_shape_map[node][i]
            node.memory = 4 * mem

    # 放出变量的np字典
    def init_Variable(self, feed_dict):
        self.Variable_node_np_value = feed_dict
        for node in self.Variable_node_list:
            nodem = self.Variable_node_to_mv[node][0]
            nodev = self.Variable_node_to_mv[node][1]
            self.Variable_node_np_value[nodem] = np.zeros(feed_dict[node].shape)
            self.Variable_node_np_value[nodev] = np.zeros(feed_dict[node].shape)

    # feed_dict为np数组
    def run(self, feed_dict, Accuracy_node=None, convert_to_numpy_ret_vals=False):
        global index_to_gpu_map
        global index_to_cpu_map
        global index_to_cpu_flag
        global swap_in_flag
        global swap_in_id
        global topo_order
        global maxmem

        schedule = self.schedule
        if self.isfirstrun == 0:
            stime = time.time()
            # Bytes/second
            if schedule:
                pciin, pciout = gpu_op.testPcie()
                pciin *= 1024
                pciout *= 1024
                print(f'pciin:{pciin}, pciout:{pciout}')

                need_tomem = 0
            # 第一次,把变量一起初始化了
            feed_dict.update(self.Variable_node_np_value)

            # 先确定shape

            # input的shape
            feed_shapes = {}
            for node, value in feed_dict.items():
                feed_shapes[node] = value.shape

            # 把shape放进self.node_to_shape_map
            self.infer_shape(feed_shapes)

            for node in self.node_to_shape_map:
                index_to_cpu_map[node.index] = ndarray.empty(self.node_to_shape_map[node])

            if schedule:
                for node in self.topo_order:
                    node.srcs = list(node.inputs)
                    node.swapouttime = node.memory / pciout
                    node.swapintime = node.memory / pciin
                    index_to_cpu_flag[node.index] = False
                    index_to_gpu_map[node.index] = None

            # 存已经被计算过的node
            # node_computed = set()
            topo_order = self.topo_order
            # 日志记录
            self.start_finish_time = datetime.datetime.now()
            self.node_order.append("topo_order:")
            self.node_order.append("\nrun:")
            # 开始运行
            time_stamp = 0
            for idx in range(len(self.topo_order)):
                st1 = time.time()
                st2 = 0
                node = self.topo_order[idx]

                # 已经被计算过了
                # if node in node_computed:
                #     continue
                # 是inputs

                if node in feed_dict.keys():
                    # 申请空间
                    ret = ndarray.array(feed_dict[node], ctx=self.ctx, maxmem=self.maxmem, nowmem=node.memory)
                    if isinstance(ret, int) and schedule:
                        st2 = time.time()
                        self.getpeakaccess(node)
                        ret, saved_memory = self.tensors_evict(ret, node, node, idx, feed_dict[node], flag=False)
                        need_tomem += saved_memory
                        st2 = time.time() - st2
                    # 此时ret为ndarray
                    # value都存在self.node_to_arr_map
                    index_to_gpu_map[idx] = ret
                    node.array_status = 1
                    # node_computed.add(node)
                    time_stamp += time.time() - st1 - st2
                    continue

                # 不是SgdOp,申请内存

                input_vals = []
                for n in node.inputs:
                    st1 = time.time()
                    st2 = 0
                    if schedule:
                        self.tensor_accsess(n, idx, time_stamp)
                        while n.array_status == 0:
                            if index_to_cpu_flag[n.index] == False:
                                continue
                            ret = ndarray.empty(self.node_to_shape_map[n], self.ctx, self.maxmem, nowmem=n.memory)
                            if isinstance(ret, int):
                                st2_ = time.time()
                                self.getpeakaccess(node)
                                ret, saved_memory = self.tensors_evict(ret, n, node, idx)
                                need_tomem += saved_memory
                                st2 += time.time() - st2_
                            index_to_cpu_map[n.index].copyto(ret, self.cudaStream)
                            index_to_gpu_map[n.index] = ret
                            n.array_status = 1
                            index_to_cpu_flag[n.index] = False
                    input_vals.append(index_to_gpu_map[n.index])
                    time_stamp += time.time() - st1 - st2

                # 除了SgdOp，其他的点此时要保证在gpu中
                st1 = time.time()
                st2=0
                if node.issgd == 0:
                    # 给这个点申请内存
                    # 申请空间
                    t1 = time.time()
                    ret = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx, maxmem=self.maxmem,
                                        nowmem=node.memory)
                    t2 = time.time()
                    if isinstance(ret, int) and schedule:
                        st2 = time.time()
                        self.getpeakaccess(node)
                        # need_tomem += ret
                        count = 0
                        for i in range(len(self.topo_order)):
                            dnode = self.topo_order[i]
                            if self.isevict(dnode, node):
                                self.tensor_evict(dnode)
                                need_tomem += dnode.memory
                                ret -= dnode.memory
                                if (ret < 0):
                                    # self.arrive_to_cpu(dnode)
                                    count = i
                                    break
                        # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
                        while True:
                            t1 = time.time()
                            ret = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx, maxmem=self.maxmem,
                                                nowmem=node.memory)
                            t2 = time.time()
                            if not isinstance(ret, int):
                                break
                            if count < len(self.topo_order) - 1:
                                count += 1
                            dnode = self.topo_order[count]
                            if self.isevict(dnode, node):
                                self.tensor_evict(dnode)
                                need_tomem += dnode.memory
                                # self.arrive_to_cpu(dnode)
                        st2 = time.time()-st2
                    # # 此时ret为ndarray
                    # # value都存在self.node_to_arr_map
                    index_to_gpu_map[idx] = ret
                    if schedule:
                        node.rp_time += (t2 - t1)
                        node.array_status = 1
                else:
                    # 是SgdOp,不申请内存
                    index_to_gpu_map[idx] = None
                node_val = index_to_gpu_map[idx]
                tic = time.time()
                memorytoSaving = node.op.compute(node, input_vals, node_val, self.cudnnHandle, self.cublasHandle,
                                                 self.cudaStream)
                toc = time.time()
                time_stamp += time.time()-st1-st2
                if memorytoSaving != 0:
                    if not schedule:
                        assert 0
                    else:
                        self.getpeakaccess(node)
                        # need_tomem += memorytoSaving
                        count = 0
                        for i in range(len(self.topo_order)):
                            dnode = self.topo_order[i]
                            if self.isevict(dnode, node):
                                self.tensor_evict(dnode)
                                memorytoSaving -= dnode.memory
                                need_tomem += dnode.memory
                                if (memorytoSaving < 0):
                                    # self.arrive_to_cpu(dnode)
                                    count = i
                                    break
                        # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
                        while True:
                            tic = time.time()
                            memorytoSaving = node.op.compute(node, input_vals, node_val, self.cudnnHandle,
                                                             self.cublasHandle, self.cudaStream)
                            toc = time.time()
                            if memorytoSaving == 0:
                                break
                            if count < len(self.topo_order) - 1:
                                count += 1
                            dnode = self.topo_order[count]
                            if self.isevict(dnode, node):
                                need_tomem += dnode.memory
                                self.tensor_evict(dnode)
                                # self.arrive_to_cpu(dnode)
                # 此点被计算过了
                # node_computed.add(node)
                if schedule:
                    node.rp_time += (toc - tic)
                    node.MSPS = node.memory / node.rp_time
            endtime = time_stamp
            # 得到ft
            if schedule:
                for i in range(len(self.topo_order)):
                    node = self.topo_order[i]
                    for j in range(0, node.access_count - 1):
                        out_id = node.use_access_id[j]
                        in_id = node.use_access_id[j + 1]
                        outtime = self.capu.tensor_access_list[out_id][2]
                        intime = self.capu.tensor_access_list[in_id][2]
                        ft = (intime - node.swapintime) - (outtime + node.swapouttime)
                        node.FT.append(ft)
                    if (node.access_count > 1):
                        out_id = node.use_access_id[node.access_count - 1]
                        outtime = self.capu.tensor_access_list[out_id][2]
                        node.FT.append(endtime - (outtime + node.swapouttime))
                self.capu.hybrid_policy(need_tomem, endtime, self.peakaccess_idx)
            # print(self.capu.policy)
            # print(self.capu.swap)
            # print(self.capu.prior_policy)

            # adam更新参数
            self.b1t[0] = self.b1t[0] * self.b1
            self.b2t[0] = self.b2t[0] * self.b2
            self.clear()
            # 不是第一次了
            self.isfirstrun = 1
            self.first_passive_cost += time.time() - stime
            return []



        else:
            # node_computed = set()
            # print('new iter')
            for idx in range(len(self.topo_order)):
                node = self.topo_order[idx]
                # 已经被计算过了
                # if node in node_computed:
                #     continue
                # 是inputs
                if node in feed_dict.keys():

                    ret = ndarray.array(feed_dict[node], ctx=self.ctx, maxmem=self.maxmem, nowmem=node.memory)
                    st = time.time()
                    if isinstance(ret, int) and schedule:
                        ret, _ = self.tensors_evict(ret, node, node, idx, feed_dict[node], flag=False, passive=True)
                    self.abnormal_passive_cost0 += time.time() - st
                    index_to_gpu_map[idx] = ret
                    if schedule:
                        node.array_status = 1
                        index_to_cpu_flag[idx] = False

                    continue

                # 如果node是变量，不用管
                if node in self.Variable_node_list:
                    continue

                # 如果node是adam要用的变量，不用管
                if node in self.mv:
                    continue

                # 不是sgdop的中间点

                input_vals = []
                inputs_evict_after_compute = []
                inputs_free_after_compute = []
                for input_node in node.inputs:
                    if schedule:
                        # st = time.time()
                        policy = self.policy_run(input_node)
                        prior_policy = self.prior_policy_run(input_node, node)
                        # self.run_policy_cost += time.time() - st
                        if input_node.array_status == 0:
                            self.arrive_to_cpu(input_node)
                            ret = ndarray.empty(self.node_to_shape_map[input_node], ctx=self.ctx, maxmem=self.maxmem,
                                                nowmem=input_node.memory)
                            st = time.time()
                            if isinstance(ret, int):
                                ret, _ = self.tensors_evict(ret, input_node, node, idx, passive=True)
                                # print(f'abnormal passive, idx:{idx}, num:3')
                                self.abnormal_passive_cost3 += time.time() - st
                            # 此时ret为ndarray
                            # value都存在self.node_to_arr_map
                            # print(f'passively swap in tensor {input_node.index} on iter {idx}, num:3')
                            st = time.time()
                            index_to_cpu_map[input_node.index].copyto(ret, self.cudaStream)
                            index_to_gpu_map[input_node.index] = ret
                            input_node.array_status = 1
                            index_to_cpu_flag[input_node.index] = False
                            self.wait_for_cpu2gpu_just_after_gpu2cpu += time.time() - st
                    input_vals.append(index_to_gpu_map[input_node.index])
                    if schedule:
                        st = time.time()
                        if policy == 1 or prior_policy == 1:
                            # self.tensor_evict(input_node)
                            inputs_evict_after_compute.append(input_node)
                        elif prior_policy == 3:
                            # self.tensor_free(input_node)
                            inputs_free_after_compute.append(input_node)
                        self.after_input_node_cost += time.time() - st

                if node.issgd == 0:
                    if index_to_gpu_map[idx] is not None:
                        pass
                    else:
                        ret = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx, maxmem=self.maxmem,
                                            nowmem=node.memory)
                        if schedule:
                            st = time.time()
                            if isinstance(ret, int) and schedule:
                                ret, _ = self.tensors_evict(ret, node, node, idx, passive=True)
                                # print(f'abnormal passive, idx:{idx}, num:4')
                                self.abnormal_passive_cost4 += time.time() - st
                            # 此时ret为ndarray
                            # value都存在self.node_to_arr_map
                            index_to_gpu_map[idx] = ret
                            node.array_status = 1
                        else:
                            while isinstance(ret, int):
                                print("显存超限")
                                assert 0
                        index_to_gpu_map[idx] = ret
                node_val = index_to_gpu_map[idx]
                memorytoSaving = node.op.compute(node, input_vals, node_val, self.cudnnHandle, self.cublasHandle,
                                                 self.cudaStream)
                if memorytoSaving != 0:
                    if not schedule:
                        assert 0
                    else:
                        st = time.time()
                        # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
                        count = 0
                        for i in range(len(self.topo_order)):
                            dnode = self.topo_order[i]
                            if self.isevict(dnode, node):
                                self.tensor_evict(dnode)
                                memorytoSaving -= dnode.memory
                                if (memorytoSaving < 0):
                                    # self.arrive_to_cpu(dnode)
                                    count = i
                                    break
                        while True:
                            memorytoSaving = node.op.compute(node, input_vals, node_val, self.cudnnHandle,
                                                             self.cublasHandle, self.cudaStream)
                            if memorytoSaving == 0:
                                break
                            if count < len(self.topo_order) - 1:
                                count += 1
                            dnode = self.topo_order[count]
                            if self.isevict(dnode, node):
                                self.tensor_evict(dnode)
                                # self.arrive_to_cpu(dnode)
                        self.abnormal_passive_cost_compute += time.time() - st
                if schedule:
                    for input_node in inputs_evict_after_compute:
                        self.tensor_evict(input_node, parallel=True)
                    for input_node in inputs_free_after_compute:
                        self.tensor_free(input_node)
                # gc.collect()
                # 此点被计算过了
                # node_computed.add(node)
            self.b1t[0] = self.b1t[0] * self.b1
            self.b2t[0] = self.b2t[0] * self.b2


            st = time.time()
            self.capu.reflush(self.reflush_access)
            self.clear()
            self.reflush_cost += time.time() - st
            return []

    def clear(self):
        self.access_index = 0
        self.reflush_access = []
        for node in self.topo_order:
            if node.isw == 0 or node.isw == 2:
                if node.array_status == 1:
                    index_to_gpu_map[node.index] = None
                index_to_cpu_flag[node.index] = False
                node.array_status = -1
            elif node.array_status == 0 and index_to_cpu_flag[node.index] == False:
                self.arrive_to_cpu(node)

    def policy_run(self, input_node):
        global swap_in_id
        global swap_in_flag
        policy = self.capu.policy[self.access_index]
        policy_in = self.capu.policy_in[self.access_index]
        # print(input_node,input_node.array_status,policy,index_to_cpu_flag[input_node.index])\
        # 计划中swap in应该完成的时刻
        if policy_in == 5:
            # print(f'swap in input node end on policy, tensor:{input_node.index}')
            st = time.time()
            if index_to_cpu_flag[input_node.index] != False:
                while index_to_cpu_flag[input_node.index] != False or swap_in_id != input_node.index:

                    if input_node.array_status == 0:
                        if len(self.reflush_access) > 0:
                            self.reflush_access.pop()
                        break
                    if swap_in_id == input_node.index and swap_in_flag == False:
                        # print("有问题")
                        if len(self.reflush_access) > 0:
                            self.reflush_access.pop()
                        input_node.array_status = 0
                        break
            else:
                if len(self.reflush_access) > 0:
                    self.reflush_access.pop()
            self.run_policy_cost+=time.time()-st
        # 对inputs计划swap in
        if policy == 2:
            swap_id = self.capu.swap[self.access_index]
            swap_node = self.topo_order[swap_id]
            # 目前在cpu上
            if swap_node.array_status == 0:
                # print(f'swap in on policy, tensor:{swap_id}')
                self.arrive_to_cpu(swap_node)
                # 执行并行swap in
                self.will_do_queue.put((swap_id, 1))
                # swap_node.array_status = 1
            else:
                swap_in_id = swap_id
                swap_in_flag = True
            self.reflush_access.append(self.access_index)

        return policy

    # 无法在策略中掩藏开销的，进行该操作
    def prior_policy_run(self, input_node, node):
        prior_policy = self.capu.prior_policy[self.access_index]
        prior_policy_in = self.capu.prior_policy_in[self.access_index]
        # 对inputs计划进行swap in
        if prior_policy_in == 2:
            # 如果当前在cpu上
            if input_node.array_status == 0:
                # print(f'swap in passively prior_policy, tensor:{input_node.index} ')
                self.arrive_to_cpu(input_node)
                ret = ndarray.empty(self.node_to_shape_map[input_node], ctx=self.ctx, maxmem=self.maxmem,
                                    nowmem=input_node.memory)
                st = time.time()
                if isinstance(ret, int):
                    ret, _ = self.tensors_evict(ret, input_node, node)
                self.abnormal_passive_cost3 += time.time()-st
                # 此时ret为ndarray
                # value都存在self.node_to_arr_map
                index_to_cpu_map[input_node.index].copyto(ret, self.cudaStream)
                index_to_gpu_map[input_node.index] = ret
                input_node.array_status = 1
                index_to_cpu_flag[input_node.index] = False
        # 对inputs计划进行重计算
        elif prior_policy_in == 4:
            # inputs已经被释放
            if input_node.array_status == 2:
                # print(f'recompute, tensor:{input_node.index} ')
                t = time.time()
                self.recompute(input_node, node, queue.Queue())
                self.recompute_cost += time.time() - t
        self.access_index += 1
        return prior_policy

    def get_start_finish_time(self):
        return self.start_finish_time

    def get_hit(self):
        return self.swap_count - self.hit_count, self.swap_count

    def get_node_order(self):
        return self.node_order

    def isevict(self, dnode, node):
        if (dnode not in node.inputs) and dnode != node and dnode.array_status == 1:
            return True
        else:
            return False

    def recompute(self, rep_node, node, stored_rep_node):
        def free_collective_rep_tensors(stored_rep_node, memorytoSaving):
            saved_memory = 0
            while saved_memory <= memorytoSaving and not stored_rep_node.empty():
                node_to_free = stored_rep_node.get()
                if node_to_free not in rep_node.inputs:
                    self.tensor_free(node_to_free)
                    saved_memory += node_to_free.memory
                else:
                    break

        # 得到gpu上的输入
        input_vals = []
        for n in rep_node.inputs:
            if n.array_status == 1:
                if index_to_cpu_flag[n.index] == True:
                    while True:
                        if index_to_cpu_flag[n.index] == False and swap_in_id == n.index:
                            break
                        if swap_in_id == n.index and swap_in_flag == False:
                            n.array_status = 0
                            break

            if n.array_status == 0:
                self.arrive_to_cpu(n)
                ret = ndarray.empty(self.node_to_shape_map[n], ctx=self.ctx, maxmem=self.maxmem, nowmem=n.memory)
                if isinstance(ret, int):
                    free_collective_rep_tensors(stored_rep_node, ret)
                    ret = ndarray.empty(self.node_to_shape_map[n], ctx=self.ctx, maxmem=self.maxmem, nowmem=n.memory)
                    if isinstance(ret, int):
                        ret = self.tensors_evict_rep(ret, n, node, rep_node)
                # 此时ret为ndarray
                # value都存在self.node_to_arr_map
                index_to_cpu_map[n.index].copyto(ret, self.cudaStream)
                index_to_gpu_map[n.index] = ret
                n.array_status = 1
                index_to_cpu_flag[n.index] = False
            elif n.array_status == 2:
                self.recompute(n, rep_node, stored_rep_node)
            input_vals.append(index_to_gpu_map[n.index])

        # 申请重算结果地址
        ret = ndarray.empty(self.node_to_shape_map[rep_node], ctx=self.ctx, maxmem=self.maxmem, nowmem=rep_node.memory)
        if isinstance(ret, int):
            free_collective_rep_tensors(stored_rep_node, ret)
            ret = ndarray.empty(self.node_to_shape_map[rep_node], ctx=self.ctx, maxmem=self.maxmem,
                                nowmem=rep_node.memory)
            if isinstance(ret, int):
                ret = self.tensors_evict_rep(ret, rep_node, node, rep_node)
        index_to_gpu_map[rep_node.index] = ret
        rep_node.array_status = 1
        index_to_cpu_flag[rep_node.index] = False

        node_val = index_to_gpu_map[rep_node.index]
        # print(rep_node)
        memorytoSaving = rep_node.op.compute(rep_node, input_vals, node_val, self.cudnnHandle, self.cublasHandle,
                                             self.cudaStream)
        if memorytoSaving != 0:
            # 首先试图清除刚刚重计算的中间变量
            free_collective_rep_tensors(stored_rep_node, memorytoSaving)

            while True:
                memorytoSaving = rep_node.op.compute(rep_node, input_vals, node_val, self.cudnnHandle,
                                                     self.cublasHandle, self.cudaStream)

                if memorytoSaving == 0:
                    break
                elif not stored_rep_node.empty():
                    free_collective_rep_tensors(stored_rep_node, memorytoSaving)
                else:
                    raise Exception('no enough memory for collective recomputation')
        stored_rep_node.put(rep_node)

    def arrive_to_cpu(self, n):
        t = time.time()
        while index_to_cpu_flag[n.index] != True:
            continue
        self.wait_for_arriving_to_cpu = + time.time() - t
        # print('arrive_to_cpu')

    def tensor_evict(self, access_node, parallel=False):
        if parallel:
            self.will_do_queue.put((access_node.index, 0))
        else:
            node_index = access_node.index
            node_ndarray = index_to_gpu_map[node_index]
            if node_ndarray is not None:
                if index_to_cpu_map[node_index].data_id !=node_ndarray.data_id:
                    node_ndarray.copyto(index_to_cpu_map[node_index], self.cudaStream)
                index_to_cpu_flag[node_index] = True
                # index_to_gpu_map[node_index].free_gpu()
                index_to_gpu_map[node_index] = None
                topo_order[node_index].array_status = 0

    def tensor_free(self, access_node):
        # index_to_gpu_map[access_node.index].free_gpu()
        index_to_gpu_map[access_node.index] = None
        access_node.array_status = 2

    def tensors_evict(self, ret, n, node, idx=-1, feed=None, flag=True, passive=False):
        count = 0
        saved_memory = 0
        for i in range(len(self.topo_order)):
            dnode = self.topo_order[i]
            if self.isevict(dnode, node):
                self.tensor_evict(dnode)
                # if passive:
                #     print(f'passively evict tensor:{i} on iter {idx}')
                ret -= dnode.memory
                saved_memory += dnode.memory
                if (ret < 0):
                    # self.arrive_to_cpu(dnode)
                    count = i
                    break
        # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
        while True:
            if flag:
                ret = ndarray.empty(self.node_to_shape_map[n], ctx=self.ctx, maxmem=self.maxmem, nowmem=n.memory)
            else:
                ret = ndarray.array(feed, ctx=self.ctx, maxmem=self.maxmem, nowmem=n.memory)

            if not isinstance(ret, int):
                break
            if count < len(self.topo_order) - 1:
                count += 1
            dnode = self.topo_order[count]
            if self.isevict(dnode, node):
                self.tensor_evict(dnode)
                saved_memory += dnode.memory
                # if passive:
                #     print(f'passively evict tensor:{count} on iter {idx}')
                # self.arrive_to_cpu(dnode)
        return ret, saved_memory

    def tensors_evict_rep(self, ret, n, node1, node2, feed=None):
        count = 0
        for i in range(len(self.topo_order)):
            dnode = self.topo_order[i]
            if self.isevict(dnode, node1) and self.isevict(dnode, node2):
                self.tensor_evict(dnode)
                # print(f'rep evict tensor:{i}')
                ret -= dnode.memory
                if (ret < 0):
                    self.arrive_to_cpu(dnode)
                    count = i
                    break
        # 返回的int意味着内存不够，此时ret是申请失败的cudamalloc（，size）的size，同理见ndarray的初始化函数，这里被动模式
        while True:
            if feed == None:
                ret = ndarray.empty(self.node_to_shape_map[n], ctx=self.ctx, maxmem=self.maxmem, nowmem=n.memory)
            else:
                ret = ndarray.array(feed, ctx=self.ctx, maxmem=self.maxmem, nowmem=n.memory)

            if not isinstance(ret, int):
                break
            if count < len(self.topo_order) - 1:
                count += 1
            dnode = self.topo_order[count]
            if self.isevict(dnode, node1) and self.isevict(dnode, node2):
                self.tensor_evict(dnode)
                # print(f'rep evict tensor:{i}')
                self.arrive_to_cpu(dnode)
        return ret

    def tensor_accsess(self, access_node, idx, time_stamp):
        access_node.access_count += 1
        # t = time.time()
        self.capu.add_tensor_access_info(access_node.index, access_node.access_count, time_stamp, idx)
        # access_node.accesses.append((access_node.access_count, t, idx))
        # access_node.access_count - 1 = access_node.use_access_id
        access_node.use_access_id.append(len(self.capu.tensor_access_list) - 1)

    def getpeakaccess(self, node):
        self.peakaccess_idx.add(node.index)
        for i in range(len(self.topo_order)):
            if self.topo_order[i].array_status == 1 or self.topo_order[i] == node:
                self.topo_order[i].peakaccess.add(self.topo_order[i].access_count)

    def destroy_cudaStream(self):
        for node in self.topo_order:
            if index_to_gpu_map[node.index] != None:
                index_to_gpu_map[node.index].free_gpu()
            index_to_gpu_map[node.index] = None
        gpu_op.destroy_cublasHandle(self.cublasHandle)
        gpu_op.destroy_cudnnHandle(self.cudnnHandle)
        gpu_op.destroy_cudaStream(self.cudaStream)


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


def getcomputelist(Variable_node_list, Variable_node_grad_list, b1, b2, b1t, b2t, e, learning_rate):
    computelist = []
    mv = []
    Variable_node_to_mv = {}
    for i in range(len(Variable_node_list)):
        m = ad.Variable(Variable_node_list[i].name + 'm')
        v = ad.Variable(Variable_node_list[i].name + 'v')
        mv.append(m)
        mv.append(v)
        Variable_node_to_mv[Variable_node_list[i]] = (m, v)
        adamnode = ad.adam_op(Variable_node_list[i], m, v, Variable_node_grad_list[i], b1, b2, b1t, b2t, e,
                              learning_rate)
        adamnode.issgd = 1  # 代表不用为这个点加内存
        computelist.append(adamnode)

    return computelist, mv, Variable_node_to_mv


def swap_adam(t_order):
    for i in range(len(t_order)):
        if t_order[i].issgd == 1 and t_order[i].isw == 0:
            t_order[i].isw = 3
            filter = t_order[i].inputs[0]
            j = len(t_order) - 1
            while j > i:
                if t_order[j].issgd == 1:
                    j = j - 1
                    continue
                if filter in t_order[j].inputs:
                    break
                j = j - 1

            tmp = t_order[i]
            t_order.remove(tmp)
            t_order.insert(j, tmp)
    # for i in range(len(topoorder)):
    #     print(i,topoorder[i])
    return t_order
