from environment import Task, DAG, servers, cloud, B_aver, B_c, B_e, B_u, interval_list, request_list
from collections import deque
import numpy as np

# 由于只在EST时将任务调度到目标服务器，所以不可能出现排队现象，只是在计算EST、EFT时需要考虑当前边缘服务器是否正在处理任务，有avail时间
# 远程云不用考虑，只需要计算依赖限制的EST，边缘服务器还需考虑avail时间，取最大值

Q = 200 # Number of queues
queues = []
processors = servers + [cloud]      # servers编号为0-4, cloud编号为5
for processor in processors:        # 由于processors信息在environment.py中定义，所以这里需要重新初始化
    processor.task_list = []
    
class COFE:
    def __init__(self):
        global Q, queues, processors      
        self.num_processors = 6
        self.dags = []
        self.arrive_list = []
        self.ready_tasks = []
        self.virtual_time = 0.0        # 定义在线场景的虚拟时间，避免受机器运行时间干扰
        self.complete_task = []        # COFE算法中记录每个子任务的完成时间

        
    def advance_virtual_time(self, duration):
        self.virtual_time += duration
        
    def computeRank(self, task, k):
        np.random.seed(1)
        f = np.random.uniform(0, 1)
        curr_rank = 0
        for succ in self.dags[k].tasks:
            if self.dags[k].graph[task.id][succ.id] != -1:
                if succ.rank is None:
                    self.computeRank(succ, k)
                if self.dags[k].graph[task.id][succ.id] != 0 and 1-100**(-task.avg_comp/(self.dags[k].graph[task.id][succ.id]*B_aver/10**6)) < f:
                    pro = 0
                else:
                    pro = 1
                # print('DAG:{}, Task:{}, Succ:{}, pro:{}'.format(k, task.id, succ.id, pro))
                curr_rank = max(curr_rank, round(pro*self.dags[k].graph[task.id][succ.id]*B_aver/10**6, 1) + succ.rank)
        task.rank = task.avg_comp + curr_rank

    def receive_dag(self):
        k = 0                    # i is the index of the request/queue from 0
        DAGS = np.load('dag_info.npy', allow_pickle=True)  # Read DAG from file
        while k < Q:
            self.dags.append(DAG(k))
            self.dags[k].num_tasks, self.dags[k].comp_cost, self.dags[k].graph, deadline_heft = DAGS[k]   
            self.dags[k].deadline = self.virtual_time + deadline_heft * 1.3   # 以heft算法的deadline为基准，增加15%的时间作为在线场景DAG的deadline
            self.dags[k].r = self.virtual_time    # ms
            self.arrive_list.append(self.virtual_time)
            self.dags[k].tasks = ([Task(i,k,self.dags[k].deadline) for i in range(self.dags[k].num_tasks)])
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
        result = self.dags[k].tasks
        # for t in result:
        #     print('DAG:{} id:{} rank:{}'.format(k, t.id, t.rank))
        result.sort(key = lambda x: x.rank, reverse=True)
        queues.append(deque(result))
        self.ready_tasks.append(queues[k][0])
        self.schedule_tasks()
        
    # My algorithm is going to do as follows:
    # compute each task's l_k to sort probabilistic
    # compute EST for the task whose priority is highest
    # schedule tasks to servers according to l_k and EFT at EST to server_p

    def schedule(self):
        # COFE:
        # 1. New Request Arrival: 当有新请求到达时，计算每个任务的rank，将v_0插入ready_tasks，并从V中删除，调用schedule(ready_task)
        # 2. Processing Task Completion: 当任务完成时，从V中删除，根据该任务的直接后继判断是否是ready_task，并将其加入，调用schedule(ready_task)
        # 3. Schedule Task: 对ready_tasks队列排序，先根据deadline排序，再根据rank排序中选择一个任务，计算其EST，选择一个processor，计算其EFT，将任务调度到processor上
        # schedule task at EST
        # otherwise, wait 
        k = 0
        self.virtual_time = 0.0
        self.arrive(k)
        k += 1
        while any(queue for queue in queues) or k < len(self.arrive_list):
            while len(self.complete_task) != 0 and self.virtual_time <= self.complete_task[0].duration['end']:
                while k < len(self.arrive_list) and self.virtual_time < self.arrive_list[k] and self.arrive_list[k] < self.complete_task[0].duration['end']:
                    # 这里可能还需要调整，因为EST可能会和arrive_list[k]有关
                    self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                    self.arrive(k)
                    k += 1
                self.advance_virtual_time(self.complete_task[0].duration['end'] - self.virtual_time)
                complete_task = self.complete_task.pop(0)
                self.find_ready_tasks(complete_task)
                self.schedule_tasks()
            if not any(queue for queue in queues) and k < len(self.arrive_list):
                self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                self.arrive(k)
                k += 1
        
    def find_ready_tasks(self, t):
        successors = [succ for succ in queues[t.k] if self.dags[t.k].graph[t.id][succ.id] != -1]          # 得到completed任务的直接后继节点
        for succ in successors:
            if any(self.dags[t.k].graph[pre.id][succ.id] != -1 and self.virtual_time < pre.duration['end'] for pre in self.dags[t.k].tasks):
                continue
            if any(self.dags[t.k].graph[pre.id][succ.id] != -1 and (self.virtual_time == pre.duration['end'] and pre in self.complete_task) for pre in self.dags[t.k].tasks):
                continue
            self.ready_tasks.append(succ)
                                                
    def schedule_tasks(self):
        self.ready_tasks.sort(key=lambda x: (x.deadline, x.rank), reverse=True)
        # k_id_list = [(item.k, item.id) for item in self.ready_tasks]
        # print(k_id_list)
        for t in self.ready_tasks:
            result = self.get_tar(t)
            if isinstance(result, tuple) and len(result) == 3:
                p, est, vm = result
                processors[p].vms[vm].task_list.append(t)
            else:
                p, est = result
                processors[p].task_list.append(t)
            t.processor_id = p
            t.duration['start'] = est
            t.duration['end'] = t.duration['start'] + self.dags[t.k].comp_cost[t.id][p]
            queues[t.k].remove(t)       
            self.complete_task.append(t)
        self.complete_task.sort(key=lambda x: x.duration['end'])
        self.ready_tasks.clear()
        
    def get_est(self, t, p, k): 
        est = max(self.dags[k].r + self.dags[k].t_offload, self.virtual_time)    # 初始化est时间为任务到达时间和offload时间之和
        # est = self.dags[k].r + self.dags[k].t_offload    # 初始化est时间为任务到达时间和offload时间之和
        for pre in self.dags[k].tasks:
            if self.dags[k].graph[pre.id][t.id] != -1:  # if pre also done on p, no communication cost
                c = self.dags[k].graph[pre.id][t.id] if pre.processor_id != p.id else 0
                if pre.processor_id in range(5) and p.id in range(5): 
                    est = max(est, pre.duration['end'] + round(c*B_e/10**6, 1))  # ms
                else:
                    est = max(est, pre.duration['end'] + round(c*B_c/10**6, 1))
        if p.id in range(5) and len(p.task_list) == 0:  # 在之前没有任务则直接返回任务依赖的EST
            return est
        elif p.id == 5:
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
            if tar_p == 5:
                return (tar_p, tar_est, vm_i)
            else:
                return (tar_p, tar_est)
        
    def str(self):
        print_str = ""
        satisfy = 0
        Makespan = 0
        # for p in processors:
        #     print_str += 'Processor {}:\n'.format(p.id)
        #     for t in p.task_list:
        #         print_str += 'Dag {}, Task {}: start = {}, end = {}\n'.format(t.k, t.id ,t.duration['start'], t.duration['end'])
        for k in range(Q):
            self.Makespan = max([t.duration['end'] for t in self.dags[k].tasks]) - self.dags[k].r
            Makespan += self.Makespan
            if self.Makespan < self.dags[k].deadline - self.dags[k].r:
                satisfy += 1
            # print_str += "Makespan{} = {}\n".format(k, self.Makespan)
        average_makespan = Makespan / Q
        SR = (satisfy / Q) * 100
        print_str += "COFE:SR = {}%\n".format(SR)
        print_str += "COFE:Average Makespan = {}\n".format(average_makespan)
        return print_str
        
cofe = COFE()
cofe.receive_dag()
cofe.schedule()

str = cofe.str()
print(str)