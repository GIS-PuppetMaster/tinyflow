
from __future__ import absolute_import

import threading
import time
import numpy as np
from . import ndarray, gpu_op, memoryManager
import random
import queue
from . import autodiff as ad



index_to_cpu_map = {}
index_to_cpu_flag = {}
index_to_gpu_map = {}

# test use
def nodelist_to_name(nodelist):
    nodename = []
    for node in nodelist:
        nodename.append(node.name)
    return nodename

class MemoryManagerController(threading.Thread):
    def __init__(self, control_queue: queue.Queue, have_done_queue: queue.Queue):
        threading.Thread.__init__(self)
        self.will_do_queue = queue.Queue()
        self.have_done_queue = have_done_queue
        self.control_queue = control_queue
        # todo hard code with device id again, may need to change
        self.cpu_ctx = ndarray.cpu(0)
        self.gpu_ctx = ndarray.gpu(0)
        self.memoryManager = MemoryManager(self.will_do_queue, self.have_done_queue)
        self.memoryManager.start()

    def run(self):
        while True:
            # todo 接口内容：wait_time: 距离上一次swap的间隔时间，node_index和node_ndarray同Manager中的定义
            # todo 在此处检查当前移动是否需要，即检查是否已经在对应的ctx中，加入变量move_to_gpu
            # (wait_time, node_index, node_ndarray, move_to_gpu)
            control_message = self.control_queue.get(block=True)
            wait_time = control_message[0]
            node_index = control_message[1]
            move_to_gpu = control_message[2]
            # print(node_index, move_to_gpu)
            time.sleep(wait_time / 1000.0)
            self.will_do_queue.put((node_index, move_to_gpu))


class MemoryManager(threading.Thread):
    def __init__(self, will_do_queue: queue.Queue, have_done_queue: queue.Queue):
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


            if move_to_gpu == 0:
                node_ndarray = index_to_gpu_map[node_index]
                node_ndarray.copyto(index_to_cpu_map[node_index], self.cudaSwapStream)
                index_to_cpu_flag[node_index] = True
                index_to_gpu_map[node_index] = None
                # print("swap finish: node " + str(node_index) + " to " + str(move_to_gpu))

            else:
                node_ndarray = index_to_cpu_map[node_index]
                # time1 = datetime.datetime.now()

                node_ndarray_new = ndarray.empty(node_ndarray.shape, self.gpu_ctx)
                # time2 = datetime.datetime.now()

                node_ndarray.copyto(node_ndarray_new, self.cudaSwapStream)
                if index_to_gpu_map[node_index] is None:
                    index_to_gpu_map[node_index] = node_ndarray_new
                else:
                    print("swap in 和 passive import 重合")
                # print("swap finish: node " + str(node_index) + " to " + str(move_to_gpu))
                # print((time2 - time1).microseconds)



class Executor(object):
    """Executor computes values for given set of nodes in computation graph."""

    def __init__(self, targetloss, learning_rate, top_control_queue, top_message_queue):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        ctx: runtime DLContext, default is None which means np.ndarray on cpu
        topo_order: list of nodes in topological order
        node_to_shape_map: dict from node to shape of the node
        node_to_arr_map: dict from node to ndarray.NDArray allocated for node
        feed_shapes: shapes of feed_dict from last run(...)
        """
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
        #这个eval_node_list全是adamop不是变量
        self.eval_node_list, self.mv, self.Variable_node_to_mv = getcomputelist(self.Variable_node_list,
                                                                                self.Variable_node_grad_list, self.b1,
                                                                                self.b2, self.b1t, self.b2t, self.e,
                                                                                self.learning_rate)  # 其内存还是Variable，但是换了个点

        # 根据这个topo_order算
        self.topo_order = ad.find_topo_sort(self.eval_node_list)
        self.topo_order = swapadam(self.topo_order)
        self.node_to_shape_map = None
        self.feed_shapes = None
        self.top_control_queue = top_control_queue
        self.top_message_queue = top_message_queue
        self.control_queue = queue.Queue()
        self.have_done_queue = queue.Queue()
        self.memoryManagerController = MemoryManagerController(self.control_queue,
                                                               self.have_done_queue)
        self.memoryManagerController.start()

        self.cudaStream = gpu_op.create_cudaStream()
        self.cudnnHandle = gpu_op.create_cudnnHandle(self.cudaStream)
        self.cublasHandle = gpu_op.create_cublasHandle(self.cudaStream)

        # 按照拓扑排序设定index
        for i in range(len(self.topo_order)):
            self.topo_order[i].index = i

        print("最后输出index：")
        order = [self.targetloss.index]
        order_var = []
        order_m = []
        order_v = []
        for node in self.Variable_node_list:
            order_var.append(node.index)
            order_m.append((self.Variable_node_to_mv[node][0]).index)
            order_v.append((self.Variable_node_to_mv[node][1]).index)
        order_var.reverse()
        order_m.reverse()
        order_v.reverse()
        order = order + order_var + order_m + order_v
        print(order)
        # todo 此处hard code，后续需要修改
        self.ctx_cpu = ndarray.cpu(0)
        self.ctx_gpu = ndarray.gpu(0)

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
                continue
            input_shapes = [self.node_to_shape_map[i] for i in node.inputs]
            assert None not in input_shapes
            self.node_to_shape_map[node] = node.op.infer_shape(node, input_shapes, self.cudnnHandle)

    def memory_plan(self, feed_shapes):
        """Allocates ndarray.NDArray for every node except feed_dict nodes.

        Implementation note:
        Option 1: Alloc a ndarray.NDArray per node that persists across run()
        Option 2: Implement a memory pool to reuse memory for nodes of same
                shapes. More details see Lecture 7.

        For both options, self.node_to_arr_map stores node->NDArray mapping to
        allow mapping to persist across multiple executor.run().

        Hint: use ndarray.empty(shape, ctx=self.ctx) to allocate NDArray.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        # self.node_to_arr_map = {}
        # for node in self.topo_order:
        #     self.node_to_arr_map[node] = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx_cpu)

        assert False
    #这个多余的node对应softmax的node，用来算正确率
    def run(self, feed_dict, Accuracy_node = None, convert_to_numpy_ret_vals=False):
        """
        Parameters
        ----------
        feed_dict: a dictionary of node->np.ndarray supplied by user.
        convert_to_numpy_ret_vals: whether to convert ret vals to np.array

        Returns
        -------
        A list of values for nodes in eval_node_list. NDArray or np.ndarray.
        """

        def are_feed_shapes_equal(sa, sb):
            if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
                return False
            unmatched_item = set(sa.items()) ^ set(sb.items())
            return len(unmatched_item) == 0

        # Assume self.ctx is None implies numpy array and numpy ops.
        global index_to_gpu_map
        global index_to_cpu_map
        global index_to_cpu_flag
        index_to_gpu_map = {}
        index_to_cpu_flag = {}
        for node, value in feed_dict.items():
            # convert values to ndarray.NDArray if necessary
            # 源代码会在此处将所有CPU的内容引入GPU，为了自定义，禁用自动引入的功能，改为手动引入
            if isinstance(value, np.ndarray):
                index_to_gpu_map[node.index] = ndarray.array(value, ctx=self.ctx_cpu)
            elif isinstance(value, ndarray.NDArray):
                index_to_gpu_map[node.index] = value
            else:
                assert False, "feed_dict value type not supported"

        # collect shapes for all placeholders
        feed_shapes = {}
        for i in index_to_gpu_map:
            feed_shapes[self.topo_order[i]] = index_to_gpu_map[self.topo_order[i].index].shape

        if self.feed_shapes is None:
            # todo 向上层返回需要的信息
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            for node in self.node_to_shape_map:
                index_to_cpu_map[node.index] = ndarray.empty(self.node_to_shape_map[node], self.ctx_cpu)
            return_list = []
            for node in self.topo_order:
                node_inputs = []
                for node_input in node.inputs:
                    node_inputs.append(node_input.index)
                node_size = np.prod(self.node_to_shape_map[node]) * 4
                print("node" + str(node.index) + " size: " + str(node_size))
                # if len(self.node_to_shape_map[node]) == 1:
                #     node_size = self.node_to_shape_map[node][0] * 4
                # else:
                #     node_size = self.node_to_shape_map[node][0] * self.node_to_shape_map[node][1] * 4
                operation_name = node.op
                return_element = [node.index, node_inputs, node_size, operation_name]
                return_list.append(return_element)
            self.top_message_queue.put([0, return_list])
        else:
            return_list = []
            for i in range(len(self.topo_order)):
                return_element = (i, self.topo_order[i].runtime)
                return_list.append(return_element)
            self.top_message_queue.put([1, return_list])

        # infer shape if feed_shapes changed since last run
        # e.g. call run() on test data after trainng
        if not are_feed_shapes_equal(feed_shapes, self.feed_shapes):
            # todo not allowed to change when running
            assert False
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes

        # calculate started

        for node in self.topo_order:
            node.array_status = 0

        # todo 测试用
        have_got_global_message = False

        if not self.top_control_queue.empty():
            have_got_global_message = True
            print("get control message")
            # todo 解析从上游传入的控制信息。

            top_swap_list, top_release_list, top_recomputation_list = self.top_control_queue.get()

            # 顺序为(start_node, start_node_type, start_time, node_id, move_to_gpu)
            # 此处保证start_time按照顺序排布

            for control_node in self.topo_order:
                control_node.control_message_in = []
                control_node.control_message_in_time = 0
                control_node.control_message_out = []
                control_node.control_message_out_time = 0
                # wait_time, node_id, move_to_gpu
                control_node.recompute_list = []
                control_node.release_list = []

            for swap_message in top_swap_list:
                node_index = swap_message[0]
                start_time = swap_message[1]
                start_node_id = swap_message[2]
                move_to_gpu = swap_message[3]

                start_node = self.topo_order[start_node_id]
                if start_node.control_message_out_time == 0:
                    start_node.control_message_out_time = start_time
                    start_node.control_message_out.append((start_time, node_index, move_to_gpu))
                else:
                    start_node.control_message_out.append(
                        (start_time - start_node.control_message_out_time, node_index, move_to_gpu))
                    start_node.control_message_out_time = start_time

            for release_message in top_release_list:
                start_node_id = release_message[0]
                node_id = release_message[1]

                start_node = self.topo_order[start_node_id]
                start_node.release_list.append(node_id)

            # print(top_swap_list)
            # print(top_release_list)
            # print(top_recomputation_list)
            print("swap list")
            for node in self.topo_order:
                print(node.control_message_out)
            print("recompute list")
            for node in self.topo_order:
                print(node.recompute_list)
            print("release list")
            for node in self.topo_order:
                print(node.release_list)
            print("update control message")

        # Traverse graph in topo order and compute values for all nodes.
        for node in self.topo_order:
            if have_got_global_message:
                print(node.index)

            if node.index in index_to_gpu_map:
                # Skip placeholder nodes. Values already provided by feed_dict.
                # 找出feed_dict中已经包含的ndarray
                node.array_status = 1
                continue

            input_vals = []

            for recompute_index in node.recompute_list:
                # todo  加入重计算的过程,重计算在被动swap in之前
                recompute_node = self.topo_order[recompute_index]
                recompute_inputs = []
                for input_node in recompute_node.inputs:
                    recompute_inputs.append(node_to_gpu_map[input_node])
                recompute_ndarray = ndarray.empty(self.node_to_shape_map[recompute_node], self.ctx_gpu)
                recompute_node.array_status = 1
                recompute_node.op.compute(recompute_node, recompute_inputs, recompute_ndarray, False)
                index_to_gpu_map[recompute_node.index] = recompute_ndarray

            for n in node.inputs:
                if index_to_gpu_map[n.index] is None:
                    print("when computing " + str(node.index) + " passive import " + str(n.index))
                    # todo 考虑如何被动进行swap in
                    assert index_to_cpu_flag[n.index], "输入tensor不在cpu上"
                    node_ndarray_new = ndarray.empty(self.node_to_shape_map[n], self.ctx_gpu)
                    index_to_cpu_map[n.index].copyto(node_ndarray_new, self.cudaStream)
                    index_to_gpu_map[n.index] = node_ndarray_new
                    n.array_status = 1
                input_vals.append(index_to_gpu_map[n.index])

            # input_vals = [node_to_gpu_map[n] for n in node.inputs]
            node_val = ndarray.empty(self.node_to_shape_map[node], self.ctx_gpu)
            node.array_status = 1

            # node_val is modified in-place whether np.ndarray or NDArray
            # node_val是开辟出来用来保存每一个的节点的计算的结果的，计算成功后会放入node_to_val中

            for control_message in node.control_message_in:
                wait_time = control_message[0]
                node_id = control_message[1]
                move_to_gpu = control_message[2]
                if move_to_gpu:
                    self.control_queue.put((wait_time, node_id, move_to_gpu))
                else:
                    self.control_queue.put((wait_time, node_id, move_to_gpu))

            t1 = datetime.datetime.now()
            node.op.compute(node, input_vals, node_val, self.cudnnHandle, self.cublasHandle, self.cudaStream, False)
            t2 = datetime.datetime.now()
            node.runtime = (t2 - t1).microseconds / 1000
            # print(node.index)

            # print(node.index)
            index_to_gpu_map[node.index] = node_val

            for control_message in node.control_message_out:
                wait_time = control_message[0]
                node_id = control_message[1]
                move_to_gpu = control_message[2]
                if move_to_gpu:
                    self.control_queue.put((wait_time, node_id, move_to_gpu))
                else:
                    self.control_queue.put((wait_time, node_id, move_to_gpu))

                # # todo 仅用于测试
                # self.have_done_queue.get(block=True)
                # print("swap end")

            for release_message in node.release_list:
                index_to_gpu_map[release_message] = None
                self.topo_order[release_message].array_status = 0

        # Collect node values.
        # print("success one batch")
        # 把结果输出了： [loss,变量按网络顺序,变量对应的m，变量对应的v],这里只是输出value，并不保证一定在gpu中
        # 但是如果这里value是None的话，他会报错
        result_output = [self.index_to_gpu_map[self.targetloss.index]]
        re_var = []
        re_m = []
        re_v = []
        for node in self.Variable_node_list:
            re_var.append(self.index_to_gpu_map[node.index])
            re_m.append(self.index_to_gpu_map[(self.Variable_node_to_mv[node][0]).index])
            re_v.append(self.index_to_gpu_map[(self.Variable_node_to_mv[node][1]).index])
        re_var.reverse()
        re_m.reverse()
        re_v.reverse()
        result_output = result_output + re_var + re_m + re_v
        # 结果，计算正确率
        if Accuracy_node != None:
            result_output.append(self.index_to_gpu_map[Accuracy_node.index])

        # adam更新参数
        self.b1t[0] = self.b1t[0] * self.b1
        self.b2t[0] = self.b2t[0] * self.b2
        return result_output



def get_Variable_node_list(node):

    visited = set()
    Variable_order = []
    Variable_sort_dfs(node, visited, Variable_order)
    return Variable_order



def Variable_sort_dfs(node, visited, Variable_order):
    """Post-order DFS"""
    #
    # if isinstance(node, list):
    #     print(node[0])
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
    for i in range(len(topoorder)):
        print(i,topoorder[i])
    return topoorder