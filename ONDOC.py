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
virtual_time = 0.0                  # 定义在线场景的虚拟时间，避免受机器运行时间干扰

def advance_virtual_time(duration):
    global virtual_time
    virtual_time += duration
    
class OnDoc:
    def __init__(self):
        global Q, queues, processors, virtual_time      
        self.num_processors = 6
        self.dags = []
        self.arrive_list = []
        
    def computeRank(self, task, k):
        curr_rank = 0
        for succ in self.dags[k].tasks:
            if self.dags[k].graph[task.id][succ.id] != -1:
                if succ.rank is None:
                    self.computeRank(succ, k)
                curr_rank = max(curr_rank, round(self.dags[k].graph[task.id][succ.id]*B_aver/10**6, 1) + succ.rank)
        task.rank = task.avg_comp + curr_rank

    def receive_dag(self):
        k = 0                    # i is the index of the request/queue from 0
        DAGS = np.load('dag_info.npy', allow_pickle=True)  # Read DAG from file
        while k < Q:
            self.dags.append(DAG(k))
            self.dags[k].num_tasks, self.dags[k].comp_cost, self.dags[k].graph, deadline_heft = DAGS[k]   
            self.dags[k].deadline = deadline_heft * 1.3   # 以heft算法的deadline为基准，增加15%的时间作为在线场景DAG的deadline
            self.dags[k].r = virtual_time    # ms
            self.arrive_list.append(virtual_time)
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
            advance_virtual_time(interval)
            k = k + 1
            
    def arrive(self, k):
        self.computeRank(self.dags[k].tasks[0], k)
        result = self.dags[k].tasks
        result.sort(key = lambda x: x.rank, reverse=True)
        queues.append(deque(result))

        
    # My algorithm is going to do as follows:
    # compute each task's l_k to sort probabilistic
    # compute EST for the task whose priority is highest
    # schedule tasks to servers according to l_k and EFT at EST to server_p

    def schedule(self):
        # OnDoc:
        # calculate each task's EST and EFT in each server from the head of queues, and choose the target server (minimum EFT)
        # choose task which has the smallest EST at target server to schedule
        # schedule task at EST
        # otherwise, wait 
        index = 0
        global virtual_time
        virtual_time = 0.0
        self.arrive(index)
        index += 1
        while any(queue for queue in queues) or index < len(self.arrive_list):
            if any(queue for queue in queues):
                min_k, tar_p, tar_est = self.check_queues()    # 每当有DAG加入或者任务完成调度之后都要调用
            # 如果所有队列现在都为空，并且还有未处理的到达事件
            # 则将时间推移到下一个到达事件，并将请求加入队列
            if not any(queue for queue in queues) and index < len(self.arrive_list):
                advance_virtual_time(self.arrive_list[index] - virtual_time)
                self.arrive(index)
                index += 1
                min_k, tar_p, tar_est = self.check_queues()
            while (index < len(self.arrive_list) and virtual_time < self.arrive_list[index] and self.arrive_list[index] < tar_est):
                advance_virtual_time(self.arrive_list[index] - virtual_time)
                self.arrive(index)
                index += 1
                min_k, tar_p, tar_est = self.check_queues()
            # if virtual_time > tar_est:
                # 由于DAG构造rank大的EST可能会比某些rank小的更大，所以可能出现虚拟时间超过EST的情况，只能在当前时间调度
                # 所以只根据rank来判断是不合理的，应该考虑COFE中当一个任务完成时有什么任务可以调度ready_tasks
                # print('DAGS:{}, id:{}, est:{}, virtual_time:{}'.format(min_k,queues[min_k][0].id,tar_est,virtual_time))
                # tar_est = virtual_time
            advance_virtual_time(tar_est - virtual_time)
            self.schedule_task(min_k, tar_p, tar_est)
                
    def check_queues(self):
        tar_est = float('inf')
        min_k = None
        for k in range(len(queues)):
            if len(queues[k]) > 0:
                p, est = self.get_tar(queues[k][0], k)
                if est < tar_est:
                    tar_p = p
                    tar_est = est
                    min_k = k
            else:
                continue   
        return min_k, tar_p, tar_est  
        
    def schedule_task(self, k, p, est):
        queue = queues[k]   # 调度第min_k个DAG的头任务
        if queue:
            t = queue[0]
            t.processor_id = p
            t.duration['start'] = est
            t.duration['end'] = t.duration['start'] + self.dags[t.k].comp_cost[t.id][p]
        queue.popleft()
        processors[p].task_list.append(t)
   
        
    def get_est(self, t, p, k): 
        global virtual_time
        est = max(self.dags[k].r + self.dags[k].t_offload, virtual_time)    # 初始化est时间为任务到达时间和offload时间之和
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
    
    def get_tar(self, t, k):
        # input: object t & int k
        # return target processor's id and EST of target processor
        if t.id == 0: 
            tar_p = request_list[k]         # 随机从某个边缘服务器发出请求
            tar_est = self.get_est(t, processors[tar_p], k)
            return [tar_p, tar_est]
        elif t.id == self.dags[t.k].tasks[-1].id:
            tar_p = self.dags[t.k].tasks[0].processor_id
            tar_est = self.get_est(t, processors[tar_p], k)
            return [tar_p, tar_est]
        else:
            aft = float("inf")
            for processor in processors:
                est = self.get_est(t, processor, k)
                eft = est + self.dags[k].comp_cost[t.id][processor.id]
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
            if self.Makespan < self.dags[k].deadline:
                satisfy += 1
            # print_str += "Makespan{} = {}\n".format(k, self.Makespan)
        SR = (satisfy / Q) * 100
        average_makespan = Makespan / Q
        print_str += "OnDoc:SR = {}%\n".format(SR)
        print_str += "OnDoc:Average Makespan = {}\n".format(average_makespan)
        return print_str
        
ondoc = OnDoc()
ondoc.receive_dag()
ondoc.schedule()

str = ondoc.str()
print(str)
