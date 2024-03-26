import traceback
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
    
class OnDoc_plus:
    def __init__(self):
        global Q, queues, processors     
        self.num_processors = 6
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
                curr_rank = max(curr_rank, round(self.dags[k].graph[task.id][succ.id]*B_aver/10**6, 1) + succ.rank)
        task.rank = task.avg_comp + curr_rank

    def LT(self, k):
        for t in self.dags[k].tasks:
            t.lt = self.dags[k].r + (self.dags[k].deadline - self.dags[k].r) * (self.dags[k].tasks[0].rank - t.rank) / self.dags[k].tasks[0].rank
            
    def receive_dag(self):
        k = 0                    # i is the index of the request/queue from 0
        DAGS = np.load('dag_info.npy', allow_pickle=True)  # Read DAG from file
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
            if any(queue for queue in queues):
                task, tar_p, tar_est = self.check_ready()    # 每当有DAG加入或者任务完成调度之后都要调用
            # 得到下一个要调度的ready_task的DAG编号、目标服务器编号、目标服务器的EST
            # 将时间推移到EST，调度ready_task，更新queues和processors，
            # 如果中间经过了某个DAG的到达时间则将其加入队列或者经过了某个任务的完成时间就需要加入ready_tasks
            # 虚拟时间一直往前走，只在EST调度任务，任务完成只添加进ready_task
            if not any(queue for queue in queues) and k < len(self.arrive_list):
                self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                self.arrive(k)
                k += 1
                task, tar_p, tar_est = self.check_ready()      
            while (len(self.complete_task) != 0 and self.complete_task[0].duration['end'] <= tar_est) or (k < len(self.arrive_list) and self.arrive_list[k] <= tar_est):
                if k < len(self.arrive_list):
                    if len(self.complete_task) != 0:
                        if self.arrive_list[k] <= self.complete_task[0].duration['end']:
                            self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                            self.arrive(k)
                            k += 1
                            task, tar_p, tar_est = self.check_ready()
                        else:
                            self.advance_virtual_time(self.complete_task[0].duration['end'] - self.virtual_time)
                            completed_task = self.complete_task.pop(0)
                            self.find_ready_tasks(completed_task)
                            task, tar_p, tar_est = self.check_ready()
                    else:
                        self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                        self.arrive(k)
                        k += 1
                        task, tar_p, tar_est = self.check_ready()
                else:
                    self.advance_virtual_time(self.complete_task[0].duration['end'] - self.virtual_time)
                    completed_task = self.complete_task.pop(0)
                    self.find_ready_tasks(completed_task)
                    task, tar_p, tar_est = self.check_ready()
                    
            self.advance_virtual_time(tar_est - self.virtual_time)
            self.schedule_task(task, tar_p, tar_est)
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
        task = None
        for t in self.ready_tasks:
            p, est = self.get_tar(t)
            if est < tar_est:
                tar_p = p
                tar_est = est
                task = t
        return task, tar_p, tar_est  
        
    def schedule_task(self, task, p, est):
        task.processor_id = p
        task.duration['start'] = est
        task.duration['end'] = task.duration['start'] + self.dags[task.k].comp_cost[task.id][p]
        try:
            queues[task.k].remove(task)
        except:
            print('DAG:{} Task {} is not in queue'.format(task.k,task.id))  
        processors[p].task_list.append(task)
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
                if pre.processor_id in range(5) and p.id in range(5): 
                    est = max(est, pre.duration['end'] + round(c*B_e/10**6, 1))  # ms
                else:
                    est = max(est, pre.duration['end'] + round(c*B_c/10**6, 1))
        if p.id == 5 or len(p.task_list) == 0:  # 在云或者之前没有任务则直接返回任务依赖的EST
            return est
        else:
            avail = p.task_list[-1].duration['end'] # 否则需要返回当前processor任务list里最后一个任务的完成时间
            return max(est, avail)
    
    def get_tar(self, t):
        # input: object t & int k
        # return target processor's id and EST of target processor
        if t.id == 0: 
            tar_p = request_list[t.k]         # 随机从某个边缘服务器发出请求
            tar_est = self.get_est(t, processors[tar_p], t.k)
            return [tar_p, tar_est]
        elif t.id == self.dags[t.k].tasks[-1].id:
            tar_p = self.dags[t.k].tasks[0].processor_id
            tar_est = self.get_est(t, processors[tar_p], t.k)
            return [tar_p, tar_est]
        else:
            aft = float("inf")
            for processor in processors:
                est = self.get_est(t, processor, t.k)
                eft = est + self.dags[t.k].comp_cost[t.id][processor.id]
                if eft < aft:   # found better case of processor
                    aft = eft
                    tar_p = processor.id
                    tar_est = est
            return [tar_p, tar_est]
        
    def str(self):
        print_str = ""
        satisfy = 0
        Makespan = 0
        # for p in processors:
            # print_str += 'Processor {}:\n'.format(p.id)
            # for t in p.task_list:
                # print_str += 'Dag {}, Task {}: start = {}, end = {}\n'.format(t.k, t.id ,t.duration['start'], t.duration['end'])
        for k in range(Q):
            self.Makespan = max([t.duration['end'] for t in self.dags[k].tasks]) - self.dags[k].r
            Makespan += self.Makespan
            if self.Makespan < self.dags[k].deadline - self.dags[k].r:
                satisfy += 1
            # print_str += "Makespan{} = {}\n".format(k, self.Makespan)
        SR = (satisfy / Q) * 100
        average_makespan = Makespan / Q
        print_str += "Mine:SR = {}%\n".format(SR)
        print_str += "Mine:Average Makespan = {}\n".format(average_makespan)
        return print_str
        
ondoc = OnDoc_plus()
ondoc.receive_dag()
ondoc.schedule()

str = ondoc.str()
print(str)
