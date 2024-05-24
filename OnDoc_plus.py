import traceback
from environment import Task, DAG, Server, cloud, B_aver, B_c, B_e, B_u, request_list, interval_list
from collections import deque
import numpy as np
# interval_list = np.array([2.69802919e-01, 6.37062627e-01, 5.71906793e-05, 1.80006377e-01,
#        7.93547976e-02, 4.84419358e-02, 1.03057317e-01, 2.11988241e-01,
#        2.52726271e-01, 3.86979887e-01, 2.71669685e-01, 5.77939855e-01,
#        1.14362204e-01, 1.05234865e+00, 1.38848124e-02, 5.55040163e-01,
#        2.70045523e-01, 4.09003657e-01, 7.56364598e-02, 1.10386612e-01,
#        8.06583851e-01, 1.72511360e+00, 1.88019306e-01, 5.89351749e-01,
#        1.04530849e+00, 1.12502793e+00, 4.44397667e-02, 1.99189391e-02,
#        9.30626426e-02, 1.05245149e+00, 5.17626747e-02, 2.73319350e-01,
#        1.58372944e+00, 3.80890007e-01, 5.88628297e-01, 1.89544735e-01,
#        5.79979439e-01, 8.99771860e-01, 9.22878759e-03, 6.93435894e-01,
#        2.24865540e+00, 6.89491883e-01, 1.64560456e-01, 7.78610927e-01,
#        5.44757035e-02, 2.97007182e-01, 1.19623030e+00, 1.73796829e-01,
#        1.69680941e-01, 6.96474547e-02, 9.77847725e-03, 5.67900964e-01,
#        1.18892683e-01, 1.54314405e-01, 3.38216973e-01, 2.74195472e-02,
#        4.26796020e-01, 7.93387908e-02, 4.44952870e-01, 6.01583831e-01,
#        5.39788476e-02, 2.67265518e-01, 5.92739369e-01, 2.67370728e-01,
#        2.56221525e-02, 3.83823744e-01, 5.45016565e-01, 3.61688890e-01,
#        1.44654052e+00, 4.41615441e-01, 1.16859818e+00, 7.39454010e-02,
#        7.49908934e-02, 8.23547275e-01, 2.53480581e-01, 9.03739161e-02,
#        1.31214354e+00, 2.13675835e-01, 6.94774031e-01, 6.47309910e-01,
#        1.07410047e+00, 4.88647363e-01, 6.95035610e-01, 2.14544746e-01,
#        1.57305986e-01, 1.13113546e+00, 2.79387862e-01, 1.67392377e+00,
#        5.44491645e-01, 4.86028217e-01, 6.09403195e-02, 1.49278463e+00,
#        2.98838628e-01, 4.31836824e-01, 2.62239878e-01, 1.35266305e-01,
#        1.16848228e+00, 4.26281918e-01, 1.43722716e-03, 4.80049363e-01,
#        1.97741227e-01, 3.74391368e-01, 1.08552453e+00, 2.21015088e-01,
#        1.19590027e+00, 4.88232881e-01, 7.97386733e-03, 1.32562633e+00,
#        5.87040229e-01, 2.96150133e+00, 9.45767256e-02, 7.37489500e-02,
#        1.34852147e+00, 5.96711263e-01, 3.41395128e-02, 7.04194434e-01,
#        7.00960285e-01, 1.28213428e+00, 6.21573008e-01, 6.63492766e-02,
#        1.00402012e-02, 1.32803089e-02, 1.43574206e-02, 1.41321440e-01,
#        9.83156255e-01, 3.86995424e-01, 4.02399253e-01, 9.22677893e-01,
#        6.62935280e-02, 1.63685465e-01, 4.40654002e-01, 1.74658641e+00,
#        4.11662352e-01, 9.41167106e-03, 8.06303145e-01, 1.32617468e-01,
#        8.22805147e-01, 2.45397658e-01, 9.95868669e-01, 6.87423354e-01,
#        4.06235965e-01, 7.33547653e-02, 3.08939216e-02, 6.46805961e-02,
#        2.27874057e-02, 5.68610937e-02, 1.27903972e-01, 6.24117334e-01,
#        4.10168768e-01, 6.31773629e-03, 3.73479154e-02, 1.70982830e+00,
#        4.19781134e-01, 1.13634295e-01, 1.45393942e-01, 6.80948904e-01,
#        1.08723330e-01, 4.35370677e-01, 1.75361221e+00, 9.38099519e-01,
#        1.37118274e-01, 3.40381802e-01, 4.83733751e-01, 8.82990015e-01,
#        8.52704478e-02, 9.37545338e-03, 3.62972518e-02, 3.33101831e-01,
#        4.66120458e-01, 4.20651277e-01, 1.90895587e-01, 2.23777999e+00,
#        4.33447065e-01, 2.39131762e-01, 4.00308536e-01, 6.83902044e-01,
#        5.53170378e-01, 1.53887670e-01, 3.43187001e-02, 2.31084558e-01,
#        4.96744535e-01, 1.17971312e-01, 6.98688885e-01, 3.44266985e-02,
#        1.50765496e-01, 8.16748932e-01, 1.07484950e-01, 5.10077407e-01,
#        3.71873315e-01, 1.29385502e+00, 1.52785071e-01, 3.41185914e-02,
#        6.64137201e-01, 7.39595393e-01, 1.19198355e+00, 1.34391846e+00,
#        7.02490548e-03, 1.33522959e-01, 4.79570877e-01, 1.48812486e+00])
# 由于只在EST时将任务调度到目标服务器，所以不可能出现排队现象，只是在计算EST、EFT时需要考虑当前边缘服务器是否正在处理任务，有avail时间
# 远程云不用考虑，只需要计算依赖限制的EST，边缘服务器还需考虑avail时间，取最大值

Q = 100 # Number of queues
queues = []
comp = 5335
server = Server(0, comp)
processors = [server] + [cloud]      # servers编号为0-4, cloud编号为5
for processor in processors:        # 由于processors信息在environment.py中定义，所以这里需要重新初始化
    processor.task_list = []
    if processor.id == 1:
        for vm in processor.vms:
            vm.task_list = []
    
class OnDoc_plus:
    def __init__(self):
        global Q, queues, processors     
        self.num_processors = 2
        self.dags = []
        self.arrive_list = []
        self.ready_tasks = []
        self.virtual_time = 0.0        # 定义在线场景的虚拟时间，避免受机器运行时间干扰
        self.complete_task = []        # COFE算法中记录每个子任务的完成时间
        
    def advance_virtual_time(self, duration):
        if duration < 0:
            print('error')
        self.virtual_time += duration
            
    def computeRank(self, task, k):
        # np.random.seed(1)
        # f = np.random.uniform(0, 1)
        curr_rank = 0
        for succ in self.dags[k].tasks:
            if self.dags[k].graph[task.id][succ.id] != -1:
                if succ.rank is None:
                    self.computeRank(succ, k)
                # if self.dags[k].graph[task.id][succ.id] != 0 and 1-1000**(-task.avg_comp/(self.dags[k].graph[task.id][succ.id]*B_aver/10**6)) < f:
                #     pro = 0
                # else:
                #     pro = 1
                # print('DAG:{}, Task:{}, Succ:{}, pro:{}'.format(k, task.id, succ.id, pro))
                curr_rank = max(curr_rank, round(self.dags[k].graph[task.id][succ.id]*B_c/10**6, 1) + succ.rank)
        task.rank = task.avg_comp + curr_rank

    def LT(self, k):
        for t in self.dags[k].tasks:
            t.lt = self.dags[k].r + (self.dags[k].deadline - self.dags[k].r) * (self.dags[k].tasks[0].rank - t.rank) / self.dags[k].tasks[0].rank
            
    def receive_dag(self):
        k = 0                    # i is the index of the request/queue from 0
        DAGS = np.load('dag_info_6_new.npy', allow_pickle=True)  # Read DAG from file
        while k < Q:
            self.dags.append(DAG(k))
            self.dags[k].num_tasks, self.dags[k].comp_cost, self.dags[k].graph, deadline_heft = DAGS[k]   
            self.dags[k].deadline = self.virtual_time + deadline_heft * 1.3   # 以heft算法的deadline为基准，增加15%的时间作为在线场景DAG的deadline
            self.dags[k].r = self.virtual_time    # ms
            self.arrive_list.append(self.virtual_time)
            self.dags[k].tasks = ([Task(i,k) for i in range(self.dags[k].num_tasks)])
            data_in = 0
            for j in range(self.dags[k].num_tasks):
                self.dags[k].tasks[j].avg_comp = sum(self.dags[k].comp_cost[j]) / self.num_processors
                if self.dags[k].graph[0][j] != -1:
                    data_in += self.dags[k].graph[0][j]
                else:
                    data_in += 0
            self.dags[k].t_offload = round(data_in * B_u / 10**6, 1)  # 任务由用户发送到服务器需要的offload时间
            interval = interval_list[k] * 1000
            self.advance_virtual_time(interval)
            k = k + 1
            
    def arrive(self, k):
        self.computeRank(self.dags[k].tasks[0], k)
        # self.LT(k)
        result = self.dags[k].tasks
        # result.sort(key = lambda x: x.rank, reverse=True)
        queues.append(deque(result))
        self.ready_tasks.append(queues[k][0])
        
    # My algorithm is going to do as follows:
    # compute each task's l_k to sort probabilistic
    # compute EST for the task whose priority is highest
    # schedule tasks to servers according to l_k and EFT at EST to server_p

    def schedule(self): 
        k = 0
        self.virtual_time = 0.0
        self.arrive(k)
        k += 1 
        while any(queue for queue in queues) or k < len(self.arrive_list):
            while (len(self.ready_tasks) == 0 and len(self.complete_task) != 0) or (len(self.ready_tasks) == 0 and k < len(self.arrive_list)):
                if len(self.complete_task) == 0:
                    self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                    self.arrive(k)
                    k += 1
                elif k >= len(self.arrive_list):
                    self.advance_virtual_time(self.complete_task[0].duration['end'] - self.virtual_time)
                    completed_task = self.complete_task.pop(0)
                    self.find_ready_tasks(completed_task)
                elif self.arrive_list[k] <= self.complete_task[0].duration['end']:
                    self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                    self.arrive(k)
                    k += 1
                else:
                    self.advance_virtual_time(self.complete_task[0].duration['end'] - self.virtual_time)
                    completed_task = self.complete_task.pop(0)
                    self.find_ready_tasks(completed_task)
                    
            if any(queue for queue in queues):
                task, tar_p, tar_est, tar_vm = self.check_ready()    # 每当有DAG加入或者任务完成调度之后都要调用
            # 得到下一个要调度的ready_task的DAG编号、目标服务器编号、目标服务器的EST
            # 将时间推移到EST，调度ready_task，更新queues和processors，
            # 如果中间经过了某个DAG的到达时间则将其加入队列或者经过了某个任务的完成时间就需要加入ready_tasks
            # 虚拟时间一直往前走，只在EST调度任务，任务完成只添加进ready_task
            if not any(queue for queue in queues) and k < len(self.arrive_list):
                self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                self.arrive(k)
                k += 1
                task, tar_p, tar_est, tar_vm = self.check_ready()      
            while (len(self.complete_task) != 0 and self.complete_task[0].duration['end'] <= tar_est) or (k < len(self.arrive_list) and self.arrive_list[k] <= tar_est):
                if k < len(self.arrive_list):
                    if len(self.complete_task) != 0:
                        if self.arrive_list[k] <= self.complete_task[0].duration['end']:
                            self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                            self.arrive(k)
                            k += 1
                            task, tar_p, tar_est, tar_vm = self.check_ready()
                        else:
                            self.advance_virtual_time(self.complete_task[0].duration['end'] - self.virtual_time)
                            completed_task = self.complete_task.pop(0)
                            self.find_ready_tasks(completed_task)
                            task, tar_p, tar_est, tar_vm = self.check_ready()
                    else:
                        self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                        self.arrive(k)
                        k += 1
                        task, tar_p, tar_est, tar_vm = self.check_ready()
                else:
                    self.advance_virtual_time(self.complete_task[0].duration['end'] - self.virtual_time)
                    completed_task = self.complete_task.pop(0)
                    self.find_ready_tasks(completed_task)
                    task, tar_p, tar_est, tar_vm = self.check_ready()
                    
            self.advance_virtual_time(tar_est - self.virtual_time)
            self.schedule_task(task, tar_p, tar_est, tar_vm)

                
    def find_ready_tasks(self, t):
        successors = [succ for succ in queues[t.k] if self.dags[t.k].graph[t.id][succ.id] != -1]          # 得到completed任务的直接后继节点
        for succ in successors:
            if any(self.dags[t.k].graph[pre.id][succ.id] != -1 and self.virtual_time < pre.duration['end'] for pre in self.dags[t.k].tasks):
                continue
            if any(self.dags[t.k].graph[pre.id][succ.id] != -1 and (self.virtual_time == pre.duration['end'] and pre in self.complete_task) for pre in self.dags[t.k].tasks):
                continue
            self.ready_tasks.append(succ)
        
    def check_ready(self):
        tar_est = float('inf')
        tar_vm = None 
        vm = None
        task = None
        for t in self.ready_tasks:
            # p, est = self.get_tar(t)
            result = self.get_tar(t)
            if isinstance(result, tuple) and len(result) == 3:
                p, est, vm = result
            else:
                p, est = result
            if est < tar_est:
                tar_p = p
                tar_est = est
                task = t
                tar_vm = vm
        return task, tar_p, tar_est, tar_vm
        
    def schedule_task(self, task, p, est, vm=None):
        task.processor_id = p
        task.duration['start'] = est
        task.duration['end'] = task.duration['start'] + self.dags[task.k].comp_cost[task.id][p]
        try:
            queues[task.k].remove(task)
        except:
            print('DAG:{} Task {} is not in queue'.format(task.k,task.id))  
        if p == 0:
            processors[p].task_list.append(task)
        else:
            cloud.vms[vm].task_list.append(task)
        self.complete_task.append(task)
        self.complete_task.sort(key=lambda x: x.duration['end'])
        try:
            self.ready_tasks.remove(task)
        except:
            print('DAG:{} Task {} is not in ready_tasks'.format(task.k,task.id))
   
    def get_est(self, t, p, k): 
        est = max(self.dags[k].r + self.dags[k].t_offload, self.virtual_time)    # 初始化est时间为任务到达时间和offload时间之和
        # est = self.dags[k].r + self.dags[k].t_offload    # 初始化est时间为任务到达时间和offload时间之和
        for pre in self.dags[k].tasks:
            if self.dags[k].graph[pre.id][t.id] != -1:  # if pre also done on p, no communication cost
                c = self.dags[k].graph[pre.id][t.id] if pre.processor_id != p.id else 0
                est = max(est, pre.duration['end'] + round(c*B_c/10**6, 1))
        if p.id == 0 and len(p.task_list) == 0:  # 在之前没有任务则直接返回任务依赖的EST
            return est
        elif p.id == 1:
            est_cloud = float('inf')
            vm_i = None 
            for i in range(len(cloud.vms)):
                if len(cloud.vms[i].task_list) == 0:
                    return (est, i)
                else:
                    if est_cloud > cloud.vms[i].task_list[-1].duration['end']:
                        est_cloud = cloud.vms[i].task_list[-1].duration['end']
                        vm_i = i
            return (max(est, est_cloud), vm_i)
        else:
            avail = p.task_list[-1].duration['end'] # 否则需要返回当前processor任务list里最后一个任务的完成时间
            return max(est, avail)
    
    def get_tar(self, t):
        # input: object t & int k
        # return target processor's id and EST of target processor
        if t.id == 0: 
            tar_p = request_list[t.k]         # 随机从某个边缘服务器发出请求
            tar_est = self.get_est(t, processors[tar_p], t.k)
            return (tar_p, tar_est)
        elif t.id == self.dags[t.k].tasks[-1].id:
            tar_p = self.dags[t.k].tasks[0].processor_id
            tar_est = self.get_est(t, processors[tar_p], t.k)
            return (tar_p, tar_est)
        else:
            aft = float("inf")
            for processor in processors:
            # processor = processors[3]
            # est = self.get_est(t, processor, t.k)
                result = self.get_est(t, processor, t.k)
                if isinstance(result, tuple):
                    est, vm_i = result
                else:
                    est = result
                eft = est + self.dags[t.k].comp_cost[t.id][processor.id]
                if eft < aft:   # found better case of processor
                    aft = eft
                    tar_p = processor.id
                    tar_est = est
            if tar_p ==1:
                return (tar_p, tar_est, vm_i)
            else:
                return (tar_p, tar_est)
        
    def str(self):
        print_str = ""
        satisfy = 0
        Makespan = 0
        # for p in processors[0:3]:
        #     print_str += 'Processor {}:\n'.format(p.id+1)
        #     for t in p.task_list:
        #         print_str += 'Dag {}, Task {}: start = {}, end = {}\n'.format(t.k, t.id ,t.duration['start'], t.duration['end'])
        # print_str += 'Cloud:\n'
        # for i in range(len(cloud.vms)):
        #     print_str += 'VM {}:\n'.format(i+1)
        #     for t in cloud.vms[i].task_list:
        #         print_str += 'Dag {}, Task {}: start = {}, end = {}\n'.format(t.k, t.id ,t.duration['start'], t.duration['end'])
        task_e = 0
        task_c = 0
        task_e += len(processors[0].task_list)
        for vm in processors[1].vms:
            task_c += len(vm.task_list)
            # for t in vm.task_list:
            #     print('id: {}, k: {}'.format(t.id,t.k))
        for k in range(Q):
            self.Makespan = max([t.duration['end'] for t in self.dags[k].tasks]) - self.dags[k].r
            if self.Makespan < self.dags[k].deadline - self.dags[k].r:
                satisfy += 1
        print("E = {}, C = {}".format(task_e, task_c))
        SR = satisfy / Q * 100
        print(SR)
        return print_str
        
ondoc = OnDoc_plus()
ondoc.receive_dag()
ondoc.schedule()

str = ondoc.str()
# print(str)
