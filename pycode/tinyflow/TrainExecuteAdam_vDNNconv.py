from __future__ import absolute_import
import numpy as np
from pycode.tinyflow import ndarray, gpu_op, memoryManager
import queue, datetime, pynvml, threading
from pycode.tinyflow import autodiff_vdnn as ad

index_to_cpu_map = {}  # 在CPU中的node
index_to_cpu_flag = {}  # 标记node实际是否在CPU中
index_to_gpu_map = {}  # 在GPU中的node

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

            global index_to_cpu_map
            global index_to_gpu_map


            if move_to_gpu == 0:
                node_ndarray = index_to_gpu_map[node_index]
                node_ndarray.copyto(index_to_cpu_map[node_index], self.cudaSwapStream)
                index_to_cpu_flag[node_index] = True
                # index_to_gpu_map[node_index] = None
                # print("swap finish: node " + str(node_index) + " to " + str(move_to_gpu))
                # print("swap_out", node_index)

            else:
                node_ndarray = index_to_cpu_map[node_index]
                # time1 = datetime.datetime.now()

                node_ndarray_new = ndarray.empty(node_ndarray.shape, self.gpu_ctx)
                # time2 = datetime.datetime.now()
                if isinstance(node_ndarray_new, int):
                    print("显存超限")
                    assert 0

                node_ndarray.copyto(node_ndarray_new, self.cudaSwapStream)
                if index_to_gpu_map[node_index] is None:
                    index_to_gpu_map[node_index] = node_ndarray_new
                    index_to_cpu_flag[node_index] = False

                else:
                    print(" ", node_index, "重复swap in")
                    assert 0




class TrainExecutor(object):
    """Executor computes values for given set of nodes in computation graph."""


    def __init__(self, targetloss, learning_rate=0.001, ctx=ndarray.gpu(0)):

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

        self.ctx_gpu = ndarray.gpu(0)
        self.ctx_cpu = ndarray.cpu(0)
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

        # 按照拓扑排序设定index, 标记卷积的input
        for i in range(len(self.topo_order)):
            node = self.topo_order[i]
            node.index = i
            node.array_status = -1
            if node.name == "Convolution2DForward" or node.name == "Convolution2DBackward" or \
                node.name == "Convolution1DForward" or node.name == "Convolution1DBackward" or \
                node.name == "Convolution3DForward" or node.name == "Convolution3DBackward":
                for input_node in node.inputs:
                    if input_node.isw == 0 or input_node.isw == 2:
                        input_node.is_conv_input = 1

            for input_node in node.inputs:
                input_node.refcnt = input_node.refcnt + 1

        self.node_refcnt = {} # 记录该node是几个node的input
        for node in self.topo_order:
            self.node_refcnt[node.index] = node.refcnt


        self.will_do_queue = queue.Queue()
        self.have_done_queue = queue.Queue()
        self.memoryManager = MemoryManager(self.will_do_queue, self.have_done_queue)
        self.memoryManager.setDaemon(True)
        self.memoryManager.start()

        # 日志记录
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
                continue
            input_shapes = [self.node_to_shape_map[i] for i in node.inputs]
            assert None not in input_shapes
            self.node_to_shape_map[node] = node.op.infer_shape(node, input_shapes, self.cudnnHandle)

    #放出变量的np字典
    def init_Variable(self, feed_dict):
        self.Variable_node_np_value = feed_dict
        for node in self.Variable_node_list:
            nodem = self.Variable_node_to_mv[node][0]
            nodev = self.Variable_node_to_mv[node][1]
            self.Variable_node_np_value[nodem] = np.zeros(feed_dict[node].shape)
            self.Variable_node_np_value[nodev] = np.zeros(feed_dict[node].shape)

    # 找到预取的层
    def find_prefetch_layer(self, currId):
        if currId < self.targetloss.index: # 只有反向才开始预取,正向不预取,targetloss是正向最后一个node
            return -1

        for i in range(currId + 1, len(self.topo_order)):
            node = self.topo_order[i]
            for n in node.inputs:
                # 每个node只预取一次它的input,只要有1个input在cpu就预取
                if n.array_status == 0 and node.prefetched == 0:
                    node.prefetched = 1
                    return i
            if node.is_conv == 1 or node.is_conv == 2 or node.issgd == 1:
                return -1
        return -1

     # 对node的所有inputs进行swap_in
    def swap_in(self, node):
        for node_input in node.inputs:
            if node_input.array_status == 1:
                while index_to_cpu_flag[node_input.index] == True:  # CPU->GPU, wait直到传到GPU
                    continue
            else:
                while index_to_cpu_flag[node_input.index] == False:  # GPU->CPU, wait直到传到CPU
                    continue
                # 手动swap in
                if node_input.array_status == 0:
                    # print("start_swap_in", node_input.index)

                    ret = ndarray.empty(self.node_to_shape_map[node_input], self.ctx_gpu)
                    if isinstance(ret, int):
                        print("显存超限")
                        assert 0
                    index_to_cpu_map[node_input.index].copyto(ret, self.cudaStream)
                    index_to_gpu_map[node_input.index] = ret
                    index_to_cpu_flag[node_input] = False
                    node_input.array_status = 1

    #feed_dict为np数组
    def run(self, feed_dict, Accuracy_node = None ,convert_to_numpy_ret_vals=False):
        global index_to_gpu_map
        global index_to_cpu_map
        global index_to_cpu_flag

        if self.isfirstrun == 0:


            #第一次,把变量一起初始化了
            feed_dict.update(self.Variable_node_np_value)

            # 先确定shape

            #input的shape
            feed_shapes = {}
            for node, value in feed_dict.items():
                feed_shapes[node] = value.shape

            #把shape放进self.node_to_shape_map
            self.infer_shape(feed_shapes)

            #存已经被计算过的node
            node_computed = set()

            # 初始化index_to_cpu_map和index_to_gpu_map, 还有index_to_cpu_flag
            for node in self.node_to_shape_map:
                index_to_cpu_map[node.index] = ndarray.empty(self.node_to_shape_map[node], self.ctx_cpu)
            for node in self.topo_order:
                index_to_cpu_flag[node.index] = False
                index_to_gpu_map[node.index] = None

            # 日志记录
            self.start_finish_time = datetime.datetime.now()
            self.node_order.append("topo_order:")
            # for i in range(len(self.topo_order)):
            #     self.node_order.append("index:" + str(i) + "\t" + self.topo_order[i].name)
            self.node_order.append("\nrun:")


            #开始运行
            for i in range(len(self.topo_order)):


                node = self.topo_order[i]
                # self.node_order.append("index:" + str(i) + "\t" + node.name + "\ttime:" + str(datetime.datetime.now()))

                # print("\n", node.index, " ", node)
                # for n in node.inputs:
                #     print("input ", n.index, " ", n)
                # print(" ")


                # 是feed_dict
                if node in feed_dict.keys():
                    # 申请空间, 不进行后续操作

                    ret = ndarray.array(feed_dict[node], ctx=self.ctx_gpu)
                    while isinstance(ret, int):
                        print("显存超限")
                        assert 0
                    # 此时ret为ndarray, 放入GPU字典中
                    index_to_gpu_map[i] = ret
                    node.array_status = 1  # on GPU
                    continue


                #不是SgdOp,申请内存
                if node.issgd == 0:
                    #给这个点申请内存
                    # 申请空间

                    ret = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx_gpu)
                    while isinstance(ret, int):
                        # print(self.node_to_shape_map[node])
                        print("显存超限")
                        assert 0
                    # 此时ret为ndarray, 放入GPU字典中
                    index_to_gpu_map[i] = ret
                    node.array_status = 1  # on GPU
                else:
                    # 是SgdOp,不申请内存
                    index_to_gpu_map[i] = None

                # 预取最近的层的输入
                prefeth_node_index = self.find_prefetch_layer(i)
                if prefeth_node_index != -1:
                    prefeth_node = self.topo_order[prefeth_node_index]
                    for node_input in prefeth_node.inputs:
                        if node_input.array_status == 0:
                            # print("start_prefeth", node_input.index)
                            self.will_do_queue.put((node_input.index, 1))
                            node_input.array_status = 1

                # swap_in, 将node从CPU取到GPU上
                self.swap_in(node)

                #放inputs的ndarray，
                input_vals = []
                for input_node in node.inputs:
                    #此时要保证在gpu中
                    if node.inputs:
                        input_vals.append(index_to_gpu_map[input_node.index])
                # swap_out
                for node_input in node.inputs:
                    node_input.refcnt = node_input.refcnt - 1
                    if node_input.is_conv_input == 1 and node_input.refcnt == 0:  # 输入是卷积层的输入，并且是最后一个被使用
                        if node_input.array_status == 1:  # on GPU
                            # print("start_swap_out", node_input.index)
                            self.will_do_queue.put((node_input.index, 0))
                            node_input.array_status = 0  # GPU to CPU


                #除了SgdOp，其他的点此时要保证在gpu中
                node_val = index_to_gpu_map[i]


                memorytoSaving = node.op.compute(node, input_vals, node_val, self.cudnnHandle, self.cublasHandle, self.cudaStream)
                while memorytoSaving != 0:
                    #不等于0意味着运行需要的临时内存不够，memorytoSaving是申请失败的cudamalloc（，size）的size
                    print("显存超限")
                    assert 0

                #此点被计算过了
                node_computed.add(node)

                # 同步
                for node_input in node.inputs:  # 计算后确保卷积的输入已经移到CPU
                    if node_input.is_conv_input == 1 and node_input.refcnt == 0:
                        # print("卸载完成")
                        while (index_to_cpu_flag[node_input.index] == False):
                            continue
                        index_to_gpu_map[node_input.index].free_gpu()
                        index_to_gpu_map[node_input.index] = None

            # 把非参数的node置为不存在, 清除gpu上非参数部分没用的值
            for node in self.topo_order:
                if node.isw == 1:  # 只有参数有用
                    continue
                node.array_status = -1
                node.prefetched = 0
                if index_to_gpu_map[node.index] !=None:
                   index_to_gpu_map[node.index].free_gpu()
                index_to_gpu_map[node.index] = None

            # 不是第一次了
            self.isfirstrun = 1

            # #把结果输出了： [loss,变量按网络顺序],这里只是输出value，并不保证一定在gpu中
            # #但是如果这里value是None的话，他会报错
            # result_output = [self.node_to_arr_map[self.targetloss]]
            # re_var = []
            # for node in self.Variable_node_list:
            #     re_var.append(self.node_to_arr_map[node])
            # re_var.reverse()
            # result_output = result_output + re_var
            # #结果，计算正确率
            # if Accuracy_node !=None:
            #     result_output.append(self.node_to_arr_map[Accuracy_node])

            #adam更新参数
            self.b1t[0] = self.b1t[0] * self.b1
            self.b2t[0] = self.b2t[0] * self.b2

            # return result_output
            return []

        else:
            # 日志记录
            self.node_order.append("\nrun:")

            # 将之前算的refcnt重新赋值给node.refcnt
            for i in range(len(self.topo_order)):
                node = self.topo_order[i]
                node.refcnt = self.node_refcnt[i]

            # 开始运行
            for i in range(len(self.topo_order)):
                node = self.topo_order[i]
                self.node_order.append("index:" + str(i) + "\t" + node.name + "\ttime:" + str(datetime.datetime.now()))

            # for node in self.topo_order:
            for i in range(len(self.topo_order)):
                node = self.topo_order[i]


                # 是inputs
                if node in feed_dict.keys():

                    if index_to_gpu_map[i] is not None:
                        index_to_gpu_map[i]._sync_copyfrom(feed_dict[node])
                    else:
                        # 没在GPU中,重新在GPU申请空间
                        ret = ndarray.array(feed_dict[node], ctx=self.ctx_gpu)
                        while isinstance(ret, int):
                            print("内存超限")
                            assert 0
                        # 此时ret为ndarray
                        # value都存在self.node_to_arr_map
                        index_to_gpu_map[i] = ret

                    index_to_cpu_flag[i] = False
                    node.array_status = 1  # on GPU
                    continue

                # 如果node是变量，不用管
                if node in self.Variable_node_list:
                    continue
                # 如果node是adam要用的变量，不用管
                if node in self.mv:
                    continue

                #不是sgdop的中间点
                if node.issgd == 0:

                    # 在gpu中，可以直接拿来用，直接pass
                    if index_to_gpu_map[i] is not None:
                        pass
                    else:
                        # 不在gpu中，生成新的empty
                        # 申请空间
                        ret = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx_gpu)
                        while isinstance(ret, int):
                            print("显存超限")
                            assert 0
                        index_to_gpu_map[i] = ret
                        index_to_cpu_flag[i] = False
                        node.array_status = 1

                # 预取最近的层的输入
                prefeth_node_index = self.find_prefetch_layer(i)
                if prefeth_node_index != -1:
                    prefeth_node = self.topo_order[prefeth_node_index]
                    for node_input in prefeth_node.inputs:
                        if node_input.array_status == 0:
                            # print("start_prefeth", node_input.index)
                            self.will_do_queue.put((node_input.index, 1))
                            node_input.array_status = 1

                # swap_in, 将node从CPU取到GPU上
                self.swap_in(node)

                # 放inputs的ndarray，
                input_vals = []
                for input_node in node.inputs:
                    # 此时要保证在gpu中
                    input_vals.append(index_to_gpu_map[input_node.index])


                for node_input in node.inputs:
                    node_input.refcnt = node_input.refcnt - 1
                    if node_input.is_conv_input == 1 and node_input.refcnt == 0:  # 输入是卷积层的输入，并且是最后一个被使用
                        if node_input.array_status == 1:  # on GPU
                            self.will_do_queue.put((node_input.index, 0))
                            node_input.array_status = 0  # GPU to CPU


                # 除了SgdOp，其他的点此时要保证在gpu中
                node_val = index_to_gpu_map[i]

                memorytoSaving = node.op.compute(node, input_vals, node_val, self.cudnnHandle, self.cublasHandle, self.cudaStream)
                while memorytoSaving != 0:
                    # 不等于0意味着运行需要的临时内存不够，memorytoSaving是申请失败的cudamalloc（，size）的size
                    print("显存超限")
                    assert 0

                # 同步
                for node_input in node.inputs:  # 计算后确保卷积的输入已经移到CPU
                    if node_input.is_conv_input == 1 and node_input.refcnt == 0:
                        # print("卸载完成")
                        while (index_to_cpu_flag[node_input.index] == False):
                            continue
                        index_to_gpu_map[node_input.index].free_gpu()
                        index_to_gpu_map[node_input.index] = None
            # 把非参数的node置为不存在, 清除gpu上非参数部分没用的值
            for node in self.topo_order:
                if node.isw == 1:  # 只有参数有用
                    continue
                node.array_status = -1
                node.prefetched = 0

                if index_to_gpu_map[node.index] !=None:
                   index_to_gpu_map[node.index].free_gpu()
                index_to_gpu_map[node.index] = None
            #
            # # 把结果输出了： [loss,变量按网络顺序],这里只是输出value，并不保证一定在gpu中
            # # 但是如果这里value是None的话，他会报错
            # result_output = [self.node_to_arr_map[self.targetloss]]
            # re_var = []
            # for node in self.Variable_node_list:
            #     re_var.append(self.node_to_arr_map[node])
            # re_var.reverse()
            # result_output = result_output + re_var
            # # 结果，计算正确率
            # if Accuracy_node != None:
            #     result_output.append(self.node_to_arr_map[Accuracy_node])

            # adam更新参数
            self.b1t[0] = self.b1t[0] * self.b1
            self.b2t[0] = self.b2t[0] * self.b2
            # return result_output
            return []

    def get_start_finish_time(self):
        return self.start_finish_time

    def get_hit(self):
        return self.hit_count, self.swap_count

    def get_node_order(self):
        return self.node_order

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

    return topoorder







