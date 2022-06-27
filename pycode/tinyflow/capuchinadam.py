from __future__ import absolute_import
from operator import itemgetter


class capuchin:
    def __init__(self, topo_order):
        self.topo_order = topo_order
        # (tensor_id, access_count, timestamp)
        self.tensor_access_list = []
        self.memory_tosaving = 0
        # (id,FT)
        self.candidates = []
        # node
        self.recomps = []
        # 策略，代表每一步做什么
        # 0 什么都不做, 1，swapout, 2 swapin, 3 swap中,什么都不能做
        self.policy = [0] * len(self.tensor_access_list)
        self.policy_in = [0] * len(self.tensor_access_list)
        # 比policy后执行,然后网络等待这一步运行完再继续运行,1和3都是等这步算完了在进行 0 什么都不做，1，swapout, 2 swapin, 3 free ,4 重计算
        self.prior_policy = [0] * len(self.tensor_access_list)
        self.prior_policy_in = [0] * len(self.tensor_access_list)
        self.swap = [-1] * len(self.tensor_access_list)
        self.end_time = 0

    def add_tensor_access_info(self, tensor_id, access_count, timestamp, access_idx):
        self.tensor_access_list.append((tensor_id, access_count, timestamp, access_idx))

    def reflush(self, reflush_access):
        for id in reflush_access:
            instart_id = id - 1
            swap_node_id = self.swap[id]
            if self.topo_order[swap_node_id].post_swap_in_times == 0:
                self.topo_order[swap_node_id].post_swap_in_times = 1.05 * self.topo_order[swap_node_id].swapintime
            else:
                self.topo_order[swap_node_id].post_swap_in_times *= 1.05
            instart_time = self.tensor_access_list[id][2] - self.topo_order[swap_node_id].post_swap_in_times
            while True:
                if self.policy[instart_id] == 0 and self.policy_in[instart_id] == 0:
                    instart_id -= 1
                    if (instart_id < 0):
                        instart_id += 1
                        break
                else:
                    instart_id += 1
                    break
                if self.tensor_access_list[instart_id][2] <= instart_time and self.policy[instart_id] == 0 and \
                        self.policy_in[instart_id] == 0:
                    break
            if instart_id != id and self.policy[instart_id] == 0 and self.policy_in[instart_id] == 0:
                self.policy[instart_id] = 2
                self.swap[instart_id] = swap_node_id
                self.swap[id] = -1
                for i in range(instart_id + 1, id + 1):
                    self.policy[i] = 4

    def hybrid_policy(self, memory_tosaving, end_time, peakaccess_idx):
        self.policy = [0] * len(self.tensor_access_list)
        self.policy_in = [0] * len(self.tensor_access_list)
        self.prior_policy = [0] * len(self.tensor_access_list)
        self.prior_policy_in = [0] * len(self.tensor_access_list)
        self.swap = [-1] * len(self.tensor_access_list)
        self.memory_tosaving = memory_tosaving
        self.end_time = end_time

        def identifyAndSortCandidates():
            # access_count>1加入候选集
            for node in self.topo_order:
                if node.access_count > 1:
                    # 遍历所有access
                    for i in range(0,node.access_count):
                        # 如果下一次access出现在peak时
                        if  i+1 in node.peakaccess :
                            # (目标node_index, 本次access的FT, 本次access_index)
                            self.candidates.append((node.index,node.FT[i],i))

            # 位于高峰也加入，咋判断高峰
            # 排序,从大到小 FT
            self.candidates = sorted(self.candidates, key=itemgetter(1), reverse=True)

        def chooseSwapIntrigger(t, ti):
            if ((t, self.topo_order[t].FT[ti], ti)) in self.recomps:
                return True
            limit = len(self.tensor_access_list) - 1
            # 本次access后swap out
            swapout_id = self.topo_order[t].use_access_id[ti]
            swapouttime = self.topo_order[t].swapouttime
            swapoutend_time = self.tensor_access_list[swapout_id][2] + swapouttime

            # 从前往后数,有没得policy为0的
            if ti == len(self.topo_order[t].FT) - 1:
                out_start = swapout_id
                out_end = out_start
                while True:
                    if self.tensor_access_list[out_end][2] > swapoutend_time and self.policy[out_end] == 0:
                            break
                    out_end += 1
                    if out_end > limit:
                        return False
                self.policy[out_start] = 1
                self.swap[out_start] = t
                for i in range(out_start + 1, out_end + 1):
                    if self.tensor_access_list[i][2] < swapoutend_time:
                        self.policy[i] = 3
                self.candidates.remove((t, self.topo_order[t].FT[ti], ti))
                self.memory_tosaving = self.memory_tosaving - self.topo_order[t].memory
                return True
            # 从要evict的下一次access开始swap in
            swapin_id = self.topo_order[t].use_access_id[ti + 1]
            swapintime = self.topo_order[t].swapintime
            swapinstart_time = self.tensor_access_list[swapin_id][2] - swapintime
            out_start = swapout_id
            out_end = out_start
            while True:
                if self.tensor_access_list[out_end][2] >= swapoutend_time and self.policy[out_end] == 0:
                    break
                out_end += 1
                if out_end > limit:
                    return False
            in_end = swapin_id
            in_start = in_end - 1
            if (in_start < 0):
                return False
            while True:
                if self.tensor_access_list[in_start + 1][2] < swapinstart_time and self.policy[in_start] == 0 and \
                            self.policy_in[in_start] == 0:
                        break
                in_start -= 1
                if in_start <= out_end:
                    return False

            # 布置策略
            for i in range(out_start, out_end+1):
                if self.swap[i]!=-1 or self.policy[i]!=0:
                    return False
            for i in range(in_start, in_end):
                if self.swap[i]!=-1 or self.policy[i]!=0:
                    return False
            if self.policy_in[in_start]!=0:
                return False


            self.policy[out_start] = 1
            self.swap[out_start] = t

            for i in range(out_start + 1, out_end + 1):
                self.policy[i] = 3

            self.policy[in_start] = 2
            self.swap[in_start] = t
            for i in range(in_start + 1, in_end):
                self.policy[i] = 4
            self.policy_in[in_end] = 5
            if self.tensor_access_list[in_end][0] != t:
                print("in问题")
            # 从候选移除
            self.candidates.remove((t, self.topo_order[t].FT[ti], ti))
            self.memory_tosaving = self.memory_tosaving - self.topo_order[t].memory
            return True

        def MaxMSPS():
            maxmsps = 0
            maxmsps_id = -1
            maxmsps_use_id = -1
            for i in self.candidates:
                if self.topo_order[i[0]].MSPS > maxmsps and self.topo_order[i[0]].isw == 0 and self.topo_order[
                    i[0]].isdrop != 1:
                    maxmsps = self.topo_order[i[0]].MSPS
                    maxmsps_id = i[0]
                    maxmsps_use_id = i[2]
            return maxmsps_id, maxmsps_use_id

        def recomputation_policy(re_id, ti):

            ext_ct = 1
            t = None
            for cand in self.candidates:
                if cand[0] == re_id and cand[2] == ti:
                    t = cand
                    break
            T = self.topo_order[re_id]

            for rp in self.recomps:
                Rp = self.topo_order[rp[0]]
                if T in Rp.srcs:
                    Rp.srcs.remove(T)
                    Rp.srcs = T.srcs + Rp.srcs
                    ext_ct = ext_ct + 1
            self.recomps.append(t)
            self.candidates.remove(t)
            self.memory_tosaving = self.memory_tosaving - T.memory

            # 更新MSPS
            for i in self.candidates:
                cand = self.topo_order[i[0]]
                if T in cand.srcs:
                    cand.srcs.remove(T)
                    cand.srcs = T.srcs + cand.srcs
                    cand.rp_time = cand.rp_time + T.rp_time
                    cand.ext_time = 0
                    for rp in self.recomps:
                        Rp = self.topo_order[rp[0]]
                        if cand in Rp.srcs:
                            cand.ext_time = cand.ext_time + cand.rp_time
                    cand.MSPS = cand.memory / cand.rp_time
                if cand in T.srcs:
                    cand.ext_time = ext_ct * cand.rp_time
                    if cand.rp_time != 0:
                        cand.MSPS = cand.memory / cand.rp_time

        identifyAndSortCandidates()
        Candidates = list(self.candidates)
        for t in Candidates:
            # 没有swap了
            if self.memory_tosaving <= 0:
                break
            if chooseSwapIntrigger(t[0], t[2]) == False:
                s_overhead = self.topo_order[t[0]].swapintime
                re_id, ti = MaxMSPS()
                r_overhead = self.topo_order[re_id].ext_time + self.topo_order[re_id].rp_time
                if t[2] == len(self.topo_order[t[0]].FT) - 1:
                    s_overhead = 0
                if ti == len(self.topo_order[re_id].FT) - 1:
                    r_overhead = self.topo_order[re_id].ext_time

                if (s_overhead <= r_overhead or re_id == -1) and self.prior_policy[
                    self.topo_order[t[0]].use_access_id[t[2]]] == 0:
                    self.prior_policy[self.topo_order[t[0]].use_access_id[t[2]]] = 1
                    if t[2] != len(self.topo_order[t[0]].FT) - 1:
                        self.prior_policy_in[self.topo_order[t[0]].use_access_id[t[2] + 1]] = 2
                        # 从候选移除
                    self.memory_tosaving -= self.topo_order[t[0]].memory
                    self.candidates.remove((t[0], self.topo_order[t[0]].FT[t[2]], t[2]))

                # and re_id != -1
                elif self.prior_policy[self.topo_order[re_id].use_access_id[ti]] == 0 and re_id != -1:
                    # 布置策略
                    self.prior_policy[self.topo_order[re_id].use_access_id[ti]] = 3
                    if ti != len(self.topo_order[re_id].FT) - 1:
                        self.prior_policy_in[self.topo_order[re_id].use_access_id[ti + 1]] = 4
                    recomputation_policy(re_id, ti)

                if self.memory_tosaving <= 0:
                    break
