from __future__ import absolute_import
import os
import numpy as np
from . import autodiff as ad
from . import ndarray, gpu_op
import time

class GradientDescent_minimize(object):

    def __init__(self, targetnode, learning_rate, ctx=ndarray.gpu(0)):


        self.targetnode = targetnode
        self.targetnode_value = None
        self.Variable_node_list = get_Variable_node_list(self.targetnode)
        self.learning_rate = learning_rate
        self.ctx = ctx
        self.Variable_node_to_val_map = None
        self.Variable_node_feed_shapes = None


        self.Variable_node_grad_list = ad.gradients(targetnode, self.Variable_node_list)
        self.compute_list = self.Variable_node_grad_list.copy()
        self.compute_list.append(self.targetnode)
        self.compute_list.reverse()

        self.executor = TrainExecutor(self.compute_list, ctx=self.ctx)

        self.except_nodelist = None
        self.except_nodelist_len = None
        self.except_nodelist_node_val_map = {}

        # optimse
        self.index_to_VaribaleNumber = None
        self.Variable_node_feed_shapes_prefix_list = None
        self.index_count = None
        self.index_to_VaribaleNumber_cuda_pointer = None
        self.g_map = None
        self.n2_cuda_pointer = None

    def init_Variable(self,feed_dict):
        #判断是否对上
        # print("variable_start")
        self.Variable_node_to_val_map = {}
        for node in self.Variable_node_list:
            if node not in feed_dict.keys():
                assert False, "缺少初始化参数"
            value = feed_dict[node]
            # convert values to ndarray.NDArray if necessary
            if isinstance(value, np.ndarray):
                self.Variable_node_to_val_map[node] = ndarray.array(value, ctx=self.ctx)
            elif isinstance(value, ndarray.NDArray):
                self.Variable_node_to_val_map[node] = value
            else:
                assert False, "feed_dict value type not supported"
        self.Variable_node_feed_shapes = {}

        self.g_map = {}
        self.index_to_VaribaleNumber = []
        self.Variable_node_feed_shapes_prefix_list = []
        self.index_count = 0
        i = 0
        for k in range(len(self.Variable_node_list)):
            node = self.Variable_node_list[k]
            s = self.Variable_node_to_val_map[node].shape
            self.Variable_node_feed_shapes[node] = s

            node_g = self.Variable_node_grad_list[k]
            self.g_map[node_g] = ndarray.empty(s, self.ctx)

            size = gpu_op.get_shape_size(s)

            self.index_count = self.index_count + size

            for j in range(size):
                self.index_to_VaribaleNumber.append(i)
                self.Variable_node_feed_shapes_prefix_list.append(j)
            i = i + 1

        self.number = i
        self.index_to_VaribaleNumber_cuda_pointer = \
            gpu_op.get_index_to_VaribaleNumber_cuda_pointer(self.index_to_VaribaleNumber,
                                                            self.Variable_node_feed_shapes_prefix_list, self.index_count)

        Variable_val_list = []
        g_list = []

        for k in range(self.number):
            node = self.Variable_node_list[k]
            Variable_val_list.append(self.Variable_node_to_val_map[node].handle.contents.data)

            node_g = self.Variable_node_grad_list[k]
            g_list.append(self.g_map[node_g].handle.contents.data)

        self.n2_cuda_pointer = gpu_op.get_n2_cuda_pointer(Variable_val_list, g_list, self.number)

        # print("variable_end")

    def run(self,feed_dict,learning_rate="default"):

        if learning_rate == "default":
            learning_rate = self.learning_rate

        if self.Variable_node_to_val_map is None:
            assert False, "没有初始化参数"
        result_list = self.executor.run(feed_dict,self.Variable_node_to_val_map,self.Variable_node_feed_shapes, self.g_map)

        if self.except_nodelist != None:
            for i in range(self.except_nodelist_len):
                self.except_nodelist_node_val_map[self.except_nodelist[self.except_nodelist_len-i-1]] = result_list.pop()
        result_list.reverse()
        self.targetnode_value = result_list.pop()



        i = 0
        for node in self.Variable_node_list:
            print('SgdOp', self.Variable_node_to_val_map[node].shape, self.Variable_node_to_val_map[node].asnumpy())
            print('SgdOp', result_list[i].shape, result_list[i].asnumpy())
            gpu_op.sgd_update(self.Variable_node_to_val_map[node],result_list[i],learning_rate,self.executor.cudaStream)
            print('SgdOp', self.Variable_node_to_val_map[node].shape, self.Variable_node_to_val_map[node].asnumpy())
            i = i + 1

        # gpu_op.sgd_compute_o(self.n2_cuda_pointer, self.index_to_VaribaleNumber_cuda_pointer,
        #                       self.index_count, learning_rate)





    def run_get_nodelist_once(self,feed_dict,nodelist,learning_rate="default"):

        once_except_nodelist_node_val_map = {}
        once_except_nodelist_len = len(nodelist)
        once_compute_list = self.compute_list + nodelist
        once_executor = TrainExecutor(once_compute_list, ctx=self.ctx)

        if learning_rate == "default":
            learning_rate = self.learning_rate

        if self.Variable_node_to_val_map is None:
            assert False, "没有初始化参数"
        result_list = once_executor.run(feed_dict, self.Variable_node_to_val_map, self.Variable_node_feed_shapes, self.g_map)

        if nodelist != None:
            for i in range(once_except_nodelist_len):
                once_except_nodelist_node_val_map[nodelist[once_except_nodelist_len - i - 1]] = result_list.pop()

        if self.except_nodelist != None:
            for i in range(self.except_nodelist_len):
                self.except_nodelist_node_val_map[self.except_nodelist[self.except_nodelist_len-i-1]] = result_list.pop()

        self.targetnode_value = result_list.pop()



        # i = 0
        # for node in self.Variable_node_list:
        #     gpu_op.sgd_update(self.Variable_node_to_val_map[node],result_list[i],learning_rate)
        #     i = i + 1

        gpu_op.sgd_compute_o(self.n2_cuda_pointer, self.index_to_VaribaleNumber_cuda_pointer,
                             self.index_count, learning_rate)



        return once_except_nodelist_node_val_map

    def set_except_comput_nodelist(self,nodelist):
        self.except_nodelist = nodelist
        self.except_nodelist_len = len(nodelist)
        self.compute_list = self.compute_list + nodelist
        self.executor = TrainExecutor(self.compute_list, ctx=self.ctx)

    def get_except_nodelist_node_val_map(self):
        return self.except_nodelist_node_val_map

    def get_Variable_node_to_val_map(self):
        return self.Variable_node_to_val_map

    def get_loss(self):
        return self.targetnode_value


class Adam_minimize(object):

    def __init__(self, targetnode, learning_rate=0.001, ctx=ndarray.gpu(0)):


        self.targetnode = targetnode
        self.targetnode_value = None
        self.Variable_node_list = get_Variable_node_list(self.targetnode)
        self.learning_rate = learning_rate
        self.ctx = ctx
        self.Variable_node_to_val_map = None
        self.Variable_node_feed_shapes = None

        self.Variable_node_grad_list = ad.gradients(targetnode, self.Variable_node_list)
        self.compute_list = self.Variable_node_grad_list.copy()
        self.compute_list.append(self.targetnode)

        self.executor = TrainExecutor(self.compute_list, ctx=self.ctx)

        self.Mt_dict = {}
        self.Vt_dict = {}
        self.b1 =0.9
        self.b2 =0.999
        self.e = 0.00000001
        self.b1t = 0.9
        self.b2t = 0.999

        self.except_nodelist = None
        self.except_nodelist_len = None
        self.except_nodelist_node_val_map = {}

        #计算正确率
        self.once_executor = None

        #optimse
        self.index_to_VaribaleNumber = None
        self.Variable_node_feed_shapes_prefix_list = None
        self.index_count = None
        self.index_to_VaribaleNumber_cuda_pointer = None
        self.g_map = None
        self.n4_cuda_pointer = None


    def init_Variable(self,feed_dict):
        #判断是否对上
        # print("variable_start")

        self.Variable_node_to_val_map = {}


        for node in self.Variable_node_list:
            if node not in feed_dict.keys():
                assert False, "缺少初始化参数"
            value = feed_dict[node]
            # convert values to ndarray.NDArray if necessary
            if isinstance(value, np.ndarray):
                self.Variable_node_to_val_map[node] = ndarray.array(value, ctx=self.ctx)
            elif isinstance(value, ndarray.NDArray):
                self.Variable_node_to_val_map[node] = value
            else:
                assert False, "feed_dict value type not supported"



        self.Variable_node_feed_shapes = {}

        self.g_map = {}
        self.index_to_VaribaleNumber = []
        self.Variable_node_feed_shapes_prefix_list = []
        self.index_count = 0
        i = 0
        for k in range(len(self.Variable_node_list)):
            node = self.Variable_node_list[k]
            s = self.Variable_node_to_val_map[node].shape
            self.Variable_node_feed_shapes[node] = s
            self.Mt_dict[node] = ndarray.empty(s, self.ctx)
            self.Vt_dict[node] = ndarray.empty(s, self.ctx)

            node_g = self.Variable_node_grad_list[k]
            self.g_map[node_g] = ndarray.empty(s, self.ctx)

            size = gpu_op.get_shape_size(s)

            self.index_count = self.index_count + size

            for j in range(size):
                self.index_to_VaribaleNumber.append(i)
                self.Variable_node_feed_shapes_prefix_list.append(j)
            i = i + 1

        self.number = i
        self.index_to_VaribaleNumber_cuda_pointer = \
            gpu_op.get_index_to_VaribaleNumber_cuda_pointer(self.index_to_VaribaleNumber, self.Variable_node_feed_shapes_prefix_list, self.index_count)


        Variable_val_list = []
        Mt_list = []
        Vt_list = []
        g_list = []

        for k in range(self.number):
            node = self.Variable_node_list[k]
            Variable_val_list.append(self.Variable_node_to_val_map[node].handle.contents.data)
            Mt_list.append(self.Mt_dict[node].handle.contents.data)
            Vt_list.append(self.Vt_dict[node].handle.contents.data)

            node_g = self.Variable_node_grad_list[k]
            g_list.append(self.g_map[node_g].handle.contents.data)

        self.n4_cuda_pointer = gpu_op.get_n4_cuda_pointer(Variable_val_list,Mt_list,Vt_list,g_list,self.number)

        # print("variable_end")



    def run(self,feed_dict,learning_rate="default"):

        if learning_rate == "default":
            learning_rate = self.learning_rate

        if self.Variable_node_to_val_map is None:
            assert False, "没有初始化参数"
        result_list = self.executor.run(feed_dict,self.Variable_node_to_val_map,self.Variable_node_feed_shapes,self.g_map)



        if self.except_nodelist != None:
            for i in range(self.except_nodelist_len):
                self.except_nodelist_node_val_map[self.except_nodelist[self.except_nodelist_len-i-1]] = result_list.pop()

        self.targetnode_value = result_list.pop()

        # i = 0
        # for node in self.Variable_node_list:
        #
        #     gpu_op.adam_mv(self.Mt_dict[node], self.Vt_dict[node], result_list[i], self.b1, self.b2)
        #     gpu_op.adam_compute(self.Variable_node_to_val_map[node], self.Mt_dict[node], self.Vt_dict[node], self.b1t,
        #                         self.b2t, self.e, learning_rate)
        #
        #
        #     i = i + 1




        gpu_op.adam_compute_o(self.n4_cuda_pointer, self.index_to_VaribaleNumber_cuda_pointer,
                              self.index_count, self.b1, self.b2, self.b1t,
                              self.b2t, self.e, learning_rate)




        #print("1:", time.clock() - a)
        self.b1t = self.b1t * self.b1
        self.b2t = self.b2t * self.b2



    def run_get_nodelist_once(self,feed_dict,nodelist,learning_rate="default"):


        once_except_nodelist_node_val_map = {}
        once_except_nodelist_len = len(nodelist)

        if self.once_executor == None:
            once_compute_list = self.compute_list + nodelist
            self.once_executor = TrainExecutor(once_compute_list, self.executor.cudnnHandle, self.executor.cublasHandle, ctx=self.ctx)
            self.once_executor.node_to_arr_map = self.executor.node_to_arr_map
            self.once_executor.input_arr = self.executor.input_arr





        if learning_rate == "default":
            learning_rate = self.learning_rate

        if self.Variable_node_to_val_map is None:
            assert False, "没有初始化参数"
        result_list = self.once_executor.run(feed_dict,self.Variable_node_to_val_map,self.Variable_node_feed_shapes,self.g_map)


        if nodelist != None:
            for i in range(once_except_nodelist_len):
                once_except_nodelist_node_val_map[nodelist[once_except_nodelist_len-i-1]] = result_list.pop()


        if self.except_nodelist != None:
            for i in range(self.except_nodelist_len):
                self.except_nodelist_node_val_map[self.except_nodelist[self.except_nodelist_len-i-1]] = result_list.pop()

        self.targetnode_value = result_list.pop()


        # i = 0
        # for node in self.Variable_node_list:
        #
        #     gpu_op.adam_mv(self.Mt_dict[node], self.Vt_dict[node], result_list[i], self.b1, self.b2)
        #     gpu_op.adam_compute(self.Variable_node_to_val_map[node], self.Mt_dict[node], self.Vt_dict[node], self.b1t,
        #                         self.b2t, self.e, learning_rate)
        #
        #
        #     i = i + 1

        gpu_op.adam_compute_o(self.n4_cuda_pointer, self.index_to_VaribaleNumber_cuda_pointer,
                              self.index_count, self.b1, self.b2, self.b1t,
                              self.b2t, self.e, learning_rate)






        self.b1t = self.b1t * self.b1
        self.b2t = self.b2t * self.b2

        return once_except_nodelist_node_val_map

    def set_except_comput_nodelist(self,nodelist):
        self.except_nodelist = nodelist
        self.except_nodelist_len = len(nodelist)
        self.compute_list = self.compute_list + nodelist
        self.executor = TrainExecutor(self.compute_list, ctx=self.ctx)

    def get_except_nodelist_node_val_map(self):
        return self.except_nodelist_node_val_map


    def get_Variable_node_to_val_map(self):
        return self.Variable_node_to_val_map

    def get_loss(self):
        return self.targetnode_value













class TrainExecutor(object):
    """Executor computes values for given set of nodes in computation graph."""

    def __init__(self, eval_node_list, ctx=None):
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
        self.eval_node_list = eval_node_list
        self.ctx = ctx
        self.topo_order = ad.find_topo_sort(self.eval_node_list)
        # print(nodelist_to_name(self.topo_order))
        self.node_to_shape_map = None
        self.node_to_arr_map = {}
        self.feed_shapes = None
        self.mcount = 0
        self.icount = 0
        self.input_arr = {}
        self.cudaStream = gpu_op.create_cudaStream()
        self.cudnnHandle = gpu_op.create_cudnnHandle(self.cudaStream)
        self.cublasHandle = gpu_op.create_cublasHandle(self.cudaStream)


    def infer_shape(self, feed_shapes):
        """Given shapes of feed_dict nodes, infer shape for all nodes in graph.

        Implementation note:
        Iteratively calls node.op.get_predict_results to infer shapes.
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
            self.node_to_shape_map[node] = node.op.get_predict_results(node, input_shapes, self.cudnnHandle)

    def memory_plan(self, feed_shapes, g_map={}, Var_map={}):
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

        for node in self.topo_order:
            if node not in g_map.keys() and node not in Var_map.keys() and node not in self.node_to_arr_map.keys():
                self.node_to_arr_map[node] = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx)
        self.node_to_arr_map.update(g_map)


    def run(self, feed_dict, Variable_node_to_val_map={}, Variable_node_feed_shapes={}, g_map={}, convert_to_numpy_ret_vals=False):
        # print("run_start")
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

        # 转化
        # Assume self.ctx is None implies numpy array and numpy ops.
        use_numpy = self.ctx is None
        node_to_val_map = {}
        for node, value in feed_dict.items():

                if isinstance(value, np.ndarray):
                    if node in self.input_arr.keys():
                        self.input_arr[node]._sync_copyfrom(value)
                        node_to_val_map[node] = self.input_arr[node]
                    else:
                        self.icount = self.icount + 1
                        # print("icount",self.icount)
                        self.input_arr[node] = ndarray.array(value,ctx=self.ctx)
                        node_to_val_map[node] = self.input_arr[node]
                elif isinstance(value, ndarray.NDArray):
                    node_to_val_map[node] = value
                else:
                    assert False, "feed_dict value type not supported"

        # collect shapes for all placeholders
        feed_shapes = {}
        for node in node_to_val_map:
            feed_shapes[node] = node_to_val_map[node].shape

        #加入参数
        feed_shapes.update(Variable_node_feed_shapes)
        node_to_val_map.update(Variable_node_to_val_map)



        # infer shape if feed_shapes changed since last run
        # e.g. call run() on test data after trainng
        if (not are_feed_shapes_equal(feed_shapes, self.feed_shapes)):
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            # plan memory if using GPU
            if (not use_numpy):
                self.mcount = self.mcount + 1
                # print("mcount",self.mcount)
                self.memory_plan(feed_shapes,g_map,node_to_val_map)
                # print("end")
        # print("run_1")
        # Traverse graph in topo order and compute values for all nodes.
        for node in self.topo_order:

            if node in node_to_val_map:
                # Skip placeholder nodes. Values already provided by feed_dict.
                continue
            input_vals = [node_to_val_map[n] for n in node.inputs]
            if use_numpy:
                node_val = np.empty(shape=self.node_to_shape_map[node])
            else:
                node_val = self.node_to_arr_map[node]
            # node_val is modified in-place whether np.ndarray or NDArray
            node.op.compute(node, input_vals, node_val, self.cudnnHandle, self.cublasHandle, self.cudaStream, use_numpy)
            # print(node.name,":  ",node_val.asnumpy())
            # 打印计算流程
            node_to_val_map[node] = node_val

        # print("run_end")
        # Collect node values.
        if not use_numpy and convert_to_numpy_ret_vals:
            return [node_to_val_map[n].asnumpy() for n in self.eval_node_list]
        return [node_to_val_map[n] for n in self.eval_node_list]







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
