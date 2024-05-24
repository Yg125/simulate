import random
from matplotlib import pyplot as plt
from Configure import DAG, Args, Task, B_u, B_aver, B_c, B_e, TIME_STAMP, eta_vio, eta, w_3
from collections import deque
import numpy as np
from Env import ScheduleEnv, Server, Remote_cloud
import torch
import torch.nn.functional as F
import rl_utils
from MADDPG import MADDPG
     
class OnDoc_plus:
    def __init__(self, args):
        comp = np.array([5335, 5340, 5390])
        server_capacity = np.array([197, 182, 182])
        self.B_u = B_u
        self.B_aver = B_aver
        self.B_c = B_c
        self.B_e = B_e
        self.TIME_STAMP = TIME_STAMP    
        self.eta_vio = eta_vio
        self.eta = eta
        self.w_3 = w_3
        self.servers = [Server(i, comp[i], server_capacity[i]) for i in range(3)]
        self.cloud = Remote_cloud(3, 3000) 
        self.Q = 100
        self.queues = [0 for _ in range(self.Q)]
        self.num_processors = 4
        self.dags = [0 for _ in range(self.Q)]
        self.arrive_list = [0 for _ in range(self.Q)]
        self.ready_tasks = []
        self.virtual_time = 0.0        # 定义在线场景的虚拟时间，避免受机器运行时间干扰
        self.complete_task = []        # COFE算法中记录每个子任务的完成时间
        self.args = args
        self.graph = np.empty((self.Q,8,8))
        self.comp_cost = np.empty((self.Q,8,4))
        self.tasks = [0 for _ in range(self.Q)]
        self.processors = self.servers + [self.cloud]     # servers编号为0-4, cloud编号为5
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.device = args.device
        self.request_list = np.array([2, 0, 1, 1, 1, 2, 1, 1, 2, 0, 0, 2, 2, 0, 0, 0, 2, 1, 2, 1, 0, 2,
       0, 1, 0, 1, 2, 1, 1, 2, 2, 2, 1, 1, 0, 0, 2, 0, 2, 2, 0, 1, 2, 1,
       1, 1, 1, 0, 1, 0, 1, 2, 2, 2, 1, 0, 1, 0, 2, 0, 0, 2, 0, 2, 1, 2,
       1, 2, 0, 2, 2, 0, 2, 1, 1, 1, 0, 2, 2, 1, 2, 2, 0, 0, 1, 1, 2, 2,
       0, 0, 1, 2, 0, 0, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 0, 0, 2, 0, 1,
       0, 0, 1, 1, 2, 0, 0, 0, 2, 0, 1, 2, 2, 0, 2, 1, 1, 0, 0, 0, 2, 0,
       2, 2, 0, 1, 2, 1, 0, 1, 1, 2, 0, 0, 2, 0, 2, 2, 1, 2, 2, 0, 1, 0,
       0, 0, 2, 0, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 0, 0, 2,
       1, 2, 1, 0, 1, 1, 1, 1, 2, 0, 2, 2, 0, 0, 0, 2, 1, 0, 2, 1, 2, 0,
       1, 1])   # Env中的request_list 避免调用全局变量
        self.interval_list = np.array([2.69802919e-01, 6.37062627e-01, 5.71906793e-05, 1.80006377e-01,
       7.93547976e-02, 4.84419358e-02, 1.03057317e-01, 2.11988241e-01,
       2.52726271e-01, 3.86979887e-01, 2.71669685e-01, 5.77939855e-01,
       1.14362204e-01, 1.05234865e+00, 1.38848124e-02, 5.55040163e-01,
       2.70045523e-01, 4.09003657e-01, 7.56364598e-02, 1.10386612e-01,
       8.06583851e-01, 1.72511360e+00, 1.88019306e-01, 5.89351749e-01,
       1.04530849e+00, 1.12502793e+00, 4.44397667e-02, 1.99189391e-02,
       9.30626426e-02, 1.05245149e+00, 5.17626747e-02, 2.73319350e-01,
       1.58372944e+00, 3.80890007e-01, 5.88628297e-01, 1.89544735e-01,
       5.79979439e-01, 8.99771860e-01, 9.22878759e-03, 6.93435894e-01,
       2.24865540e+00, 6.89491883e-01, 1.64560456e-01, 7.78610927e-01,
       5.44757035e-02, 2.97007182e-01, 1.19623030e+00, 1.73796829e-01,
       1.69680941e-01, 6.96474547e-02, 9.77847725e-03, 5.67900964e-01,
       1.18892683e-01, 1.54314405e-01, 3.38216973e-01, 2.74195472e-02,
       4.26796020e-01, 7.93387908e-02, 4.44952870e-01, 6.01583831e-01,
       5.39788476e-02, 2.67265518e-01, 5.92739369e-01, 2.67370728e-01,
       2.56221525e-02, 3.83823744e-01, 5.45016565e-01, 3.61688890e-01,
       1.44654052e+00, 4.41615441e-01, 1.16859818e+00, 7.39454010e-02,
       7.49908934e-02, 8.23547275e-01, 2.53480581e-01, 9.03739161e-02,
       1.31214354e+00, 2.13675835e-01, 6.94774031e-01, 6.47309910e-01,
       1.07410047e+00, 4.88647363e-01, 6.95035610e-01, 2.14544746e-01,
       1.57305986e-01, 1.13113546e+00, 2.79387862e-01, 1.67392377e+00,
       5.44491645e-01, 4.86028217e-01, 6.09403195e-02, 1.49278463e+00,
       2.98838628e-01, 4.31836824e-01, 2.62239878e-01, 1.35266305e-01,
       1.16848228e+00, 4.26281918e-01, 1.43722716e-03, 4.80049363e-01,
       1.97741227e-01, 3.74391368e-01, 1.08552453e+00, 2.21015088e-01,
       1.19590027e+00, 4.88232881e-01, 7.97386733e-03, 1.32562633e+00,
       5.87040229e-01, 2.96150133e+00, 9.45767256e-02, 7.37489500e-02,
       1.34852147e+00, 5.96711263e-01, 3.41395128e-02, 7.04194434e-01,
       7.00960285e-01, 1.28213428e+00, 6.21573008e-01, 6.63492766e-02,
       1.00402012e-02, 1.32803089e-02, 1.43574206e-02, 1.41321440e-01,
       9.83156255e-01, 3.86995424e-01, 4.02399253e-01, 9.22677893e-01,
       6.62935280e-02, 1.63685465e-01, 4.40654002e-01, 1.74658641e+00,
       4.11662352e-01, 9.41167106e-03, 8.06303145e-01, 1.32617468e-01,
       8.22805147e-01, 2.45397658e-01, 9.95868669e-01, 6.87423354e-01,
       4.06235965e-01, 7.33547653e-02, 3.08939216e-02, 6.46805961e-02,
       2.27874057e-02, 5.68610937e-02, 1.27903972e-01, 6.24117334e-01,
       4.10168768e-01, 6.31773629e-03, 3.73479154e-02, 1.70982830e+00,
       4.19781134e-01, 1.13634295e-01, 1.45393942e-01, 6.80948904e-01,
       1.08723330e-01, 4.35370677e-01, 1.75361221e+00, 9.38099519e-01,
       1.37118274e-01, 3.40381802e-01, 4.83733751e-01, 8.82990015e-01,
       8.52704478e-02, 9.37545338e-03, 3.62972518e-02, 3.33101831e-01,
       4.66120458e-01, 4.20651277e-01, 1.90895587e-01, 2.23777999e+00,
       4.33447065e-01, 2.39131762e-01, 4.00308536e-01, 6.83902044e-01,
       5.53170378e-01, 1.53887670e-01, 3.43187001e-02, 2.31084558e-01,
       4.96744535e-01, 1.17971312e-01, 6.98688885e-01, 3.44266985e-02,
       1.50765496e-01, 8.16748932e-01, 1.07484950e-01, 5.10077407e-01,
       3.71873315e-01, 1.29385502e+00, 1.52785071e-01, 3.41185914e-02,
       6.64137201e-01, 7.39595393e-01, 1.19198355e+00, 1.34391846e+00,
       7.02490548e-03, 1.33522959e-01, 4.79570877e-01, 1.48812486e+00])
        
    def receive_dag(self):                    # k is the index of the request/queue from 0
        virtual_time = 0
        tasks = [0 for _ in range(self.Q)]
        DAGS = np.load('/home/yangang/DAG_Q/dag_info_6.npy', allow_pickle=True)  # Read DAG from file
        for k in range(self.Q):
            self.dags[k] = DAG(k)
            self.dags[k].num_tasks, self.comp_cost[k], self.graph[k], deadline_heft = DAGS[k]   
            self.dags[k].deadline = virtual_time + deadline_heft * 1.3   # 以heft算法的deadline为基准，增加15%的时间作为在线场景DAG的deadline
            self.dags[k].r = virtual_time    # ms
            self.arrive_list[k] = virtual_time
            num_tasks = self.dags[k].num_tasks
            tasks[k] = [Task(i,k) for i in range(num_tasks)]
            data_in = 0
            for j in range(self.dags[k].num_tasks):
                tasks[k][j].avg_comp = sum(self.comp_cost[k][j]) / self.num_processors
                if self.graph[k][0][j] != -1:
                    data_in += self.graph[k][0][j]
                else:
                    data_in += 0
            self.dags[k].t_offload = round(data_in * self.B_u / 10**6, 1)  # 任务由用户发送到服务器需要的offload时间
            interval = self.interval_list[k] * 1000
            virtual_time += interval
        self.tasks = tasks
      
    def advance_virtual_time(self, duration):
        if duration < 0:
            print('error1')
        self.virtual_time += duration
            
    def computeRank(self, task, k, computed_ranks):
        if task in computed_ranks:
            return computed_ranks[task]
        curr_rank = 0
        for succ in self.tasks[k]:
            if self.graph[k][task.id][succ.id] != -1:
                if succ.rank is None:
                    self.computeRank(succ, k, computed_ranks)
                curr_rank = max(curr_rank, round(self.graph[k][task.id][succ.id]*self.B_aver/10**6, 1) + succ.rank)
        task.rank = task.avg_comp + curr_rank
        computed_ranks[task] = task.rank
    
    def LT(self, k):
        for t in self.tasks[k]:
            t.lt = self.dags[k].r + (self.dags[k].deadline - self.dags[k].r) * (self.tasks[k][0].rank - t.rank) / self.tasks[k][0].rank
   
    def arrive(self, k):
        tasks = self.tasks[k]
        computed_ranks = {}
        self.computeRank(tasks[0], k, computed_ranks)
        self.LT(k)
        try:
            self.queues[k] = deque(tasks)
        except:
            print(1)
        self.ready_tasks.append(self.queues[k][0])

    def find_ready_tasks(self, t):
        successors = [succ for succ in self.queues[t.k] if self.graph[t.k][t.id][succ.id] != -1]          # 得到completed任务的直接后继节点
        for succ in successors:
            found_pre = False
            for pre in self.tasks[t.k]:
                if self.graph[t.k][pre.id][succ.id] != -1:
                    if self.virtual_time < pre.end:
                        found_pre = True
                        break
                    if self.virtual_time == pre.end and pre in self.complete_task:
                        found_pre = True
                        break
            if not found_pre:
                self.ready_tasks.append(succ)
        
    def check_ready(self):
        tar_est = float('inf')
        tar_vm = None 
        vm = None
        task = None 
        tar_p = None 
        if not self.ready_tasks:
            print('error')
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
        if tar_p == None:
            print('erorr')
        return task, tar_p, tar_est, tar_vm
       
    def schedule_task(self, task, p, est, vm=None):
        task.processor_id = p
        task.start = est
        task.end = task.start + self.comp_cost[task.k][task.id][p]
        try:
            self.queues[task.k].remove(task)
        except:
            print('DAG:{} Task {} is not in queue'.format(task.k,task.id))  
        if p in range(3):
            self.processors[p].task_list.append(task)
        else:
            self.cloud.vms[vm].task_list.append(task)
        if task.id != 7:
            self.complete_task.append(task)

        self.complete_task.sort(key=lambda x: x.end)
        try:
            self.ready_tasks.remove(task)
        except:
            print('DAG:{} Task {} is not in ready_tasks'.format(task.k,task.id))
   
    def get_est(self, t, p, k): 
        if (p.id in range(3) and not p.service_list[t.service_id]) and (t.id != 0 and t.id != 7):
            return float('inf')
        est = max(self.dags[k].r + self.dags[k].t_offload, self.virtual_time)    # 初始化est时间为任务到达时间和offload时间之和
        graph = self.graph[k]
        tasks = self.tasks[k]
        
        for pre in tasks:
            if graph[pre.id][t.id] != -1:  # if pre also done on p, no communication cost
                c = graph[pre.id][t.id] if pre.processor_id != p.id else 0
                if pre.processor_id in range(3) and p.id in range(3): 
                    est = max(est, pre.end + round(c*self.B_e/10**6, 1))  # ms
                else:
                    est = max(est, pre.end + round(c*self.B_c/10**6, 1))
        if p.id in range(3) and not p.task_list:  # 在之前没有任务则直接返回任务依赖的EST
            return est
        elif p.id == 3:
            est_cloud = float('inf')
            vm_i = None 
            for i, vm in enumerate(self.cloud.vms):
                if not vm.task_list:
                    return (est, i)
                else:
                    if est_cloud > vm.task_list[-1].end:
                        est_cloud = vm.task_list[-1].end
                        vm_i = i
            return (max(est, est_cloud), vm_i)
        else:
            avail = p.task_list[-1].end # 否则需要返回当前processor任务list里最后一个任务的完成时间
            return max(est, avail)
    
    def get_tar(self, t):
        # input: object t & int k
        # return target processor's id and EST of target processor
        if t.id == 0: 
            tar_p = self.request_list[t.k]         # 随机从某个边缘服务器发出请求
            tar_est = self.get_est(t, self.processors[tar_p], t.k)
            return (tar_p, tar_est)
        
        elif t.id == self.tasks[t.k][-1].id:
            tar_p = self.tasks[t.k][0].processor_id
            tar_est = self.get_est(t, self.processors[tar_p], t.k)
            return (tar_p, tar_est)
        else:
            aft = float("inf")
            for processor in self.processors:
                result = self.get_est(t, processor, t.k)
                if isinstance(result, tuple):
                    est, vm_i = result
                else:
                    est = result
                eft = est + self.comp_cost[t.k][t.id][processor.id]
                if eft < aft:   # found better case of processor
                    aft = eft
                    tar_p = processor.id
                    tar_est = est
            if tar_p == 3:
                return (tar_p, tar_est, vm_i)
            else:
                return (tar_p, tar_est)

    def str(self):
        satisfy = 0
        task_e = 0
        task_c = 0
        for p in self.processors[0:3]:
            task_e += len(p.task_list)
        for vm in self.processors[3].vms:
            task_c += len(vm.task_list)
            # for t in vm.task_list:
                # print('id: {}, k: {}'.format(t.id,t.k))
        for k in range(self.Q):
            self.Makespan = max([t.end for t in self.tasks[k]]) - self.dags[k].r
            if self.Makespan < self.dags[k].deadline - self.dags[k].r:
                satisfy += 1
        # print("E = {}, C = {}".format(task_e, task_c))
        SR = satisfy / self.Q * 100
        return SR

def stack_array(x):
    # 将原本按“列”排列的元素改为按“行”排列 并转换为tensor
    rearranged = []
    for i in range(len(x[0])):
        rearranged.append([sub_x[i] for sub_x in x])
    # rearranged = [[sub_x[i] for sub_x in x] for i in range(len(x[0]))]
    return [torch.FloatTensor(np.vstack(aa)).to(device) for aa in rearranged]

num_episodes = 10000
buffer_size = 1000000
hidden_dim = 64
actor_lr = 1e-3
critic_lr = 1e-2
gamma = 0.95
tau = 1e-2
batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
update_interval = 100

env = ScheduleEnv()
args = Args()
args.set_env_info(env.get_info())

state_dims = []
action_dims = []
for action_space in range(env.agents):
    action_dims.append(26)
for state_space in range(env.agents):
    state_dims.append(13)
critic_input_dim = sum(state_dims) + sum(action_dims)
               
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims, action_dims, critic_input_dim, gamma, tau)
ondoc_plus = OnDoc_plus(args)

def evaluate(maddpg, env, ondoc_plus):
    state = env.reset()
    ondoc_plus.receive_dag()
    for i in range(env.agents):      
        ondoc_plus.processors[i].task_list = []
        ondoc_plus.processors[i].service_list = [0,0,0,0,0]
    for vm in ondoc_plus.processors[3].vms:
        vm.task_list = []
    episode_limit = ondoc_plus.args.episode_limit
    arrive_len = len(ondoc_plus.arrive_list)
    k = 0
    ondoc_plus.virtual_time = 0.0
    ondoc_plus.arrive(k)
    k += 1
    step = -1 
    episode_reward = 0
    while (any(queue for queue in ondoc_plus.queues) or k < arrive_len) and (step < episode_limit):  # 调度还没有结束
        flag_1 ,flag_2= False, False
        if ondoc_plus.virtual_time % ondoc_plus.TIME_STAMP == 0:  # 每个时间戳执行一次动作
            if replay_buffer.states_mean is None:
                actions = maddpg.take_action(state, explore=True)
            else:
                actions = maddpg.take_action((state - replay_buffer.states_mean) / (replay_buffer.states_std + 1e-8), explore=True)
            step += 1
            next_state, reward, done = env.step(actions)
            for i in range(env.agents):
                ondoc_plus.servers[i].service_list = next_state[i][0:5]
            episode_reward += reward
            TIME_NEXT = ondoc_plus.TIME_STAMP * (step + 1)
        
        for task in ondoc_plus.complete_task:
            if task.processor_id in range(3) and task.end > TIME_NEXT:
                next_state[task.processor_id][5+task.service_id] = 1
        
        while (not ondoc_plus.ready_tasks and ondoc_plus.complete_task) or (not ondoc_plus.ready_tasks and k < arrive_len):
            if not ondoc_plus.complete_task:
                if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                    flag_1 = True
                    break
                ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
                ondoc_plus.arrive(k)
                k += 1
            elif k >= arrive_len:
                if ondoc_plus.complete_task[0].end >= TIME_NEXT:
                    flag_1 = True
                    break
                completed_task = ondoc_plus.complete_task.pop(0)
                ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
                ondoc_plus.find_ready_tasks(completed_task)
            elif ondoc_plus.arrive_list[k] <= ondoc_plus.complete_task[0].end:
                if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                    flag_1 = True
                    break
                ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
                ondoc_plus.arrive(k)
                k += 1
            else:
                if ondoc_plus.complete_task[0].end >= TIME_NEXT:
                    flag_1 = True
                    break
                completed_task = ondoc_plus.complete_task.pop(0)
                ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
                ondoc_plus.find_ready_tasks(completed_task)
        if flag_1:
            ondoc_plus.advance_virtual_time(TIME_NEXT - ondoc_plus.virtual_time)
            state = next_state
            continue    
        if any(queue for queue in ondoc_plus.queues):
            task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()    # 每当有DAG加入或者任务完成调度之后都要调用
            
        # 下一段可否删除
        if not any(queue for queue in ondoc_plus.queues) and k < arrive_len:
            if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                ondoc_plus.advance_virtual_time(TIME_NEXT - ondoc_plus.virtual_time)
                continue
            ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
            ondoc_plus.arrive(k)
            k += 1
            task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()     
            
        while (len(ondoc_plus.complete_task) != 0 and ondoc_plus.complete_task[0].end <= tar_est) or (k < arrive_len and ondoc_plus.arrive_list[k] <= tar_est):
            if k < arrive_len:
                if len(ondoc_plus.complete_task) != 0:
                    if ondoc_plus.arrive_list[k] <= ondoc_plus.complete_task[0].end:
                        if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                            flag_2 = True
                            break
                        ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
                        ondoc_plus.arrive(k)
                        k += 1
                        task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()
                    else:
                        if ondoc_plus.complete_task[0].end >= TIME_NEXT:
                            flag_2 = True
                            break
                        completed_task = ondoc_plus.complete_task.pop(0)
                        ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
                        ondoc_plus.find_ready_tasks(completed_task)
                        task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()
                else:
                    if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                        flag_2 = True
                        break
                    ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
                    ondoc_plus.arrive(k)
                    k += 1
                    task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()
            else:
                if ondoc_plus.complete_task[0].end >= TIME_NEXT:
                    flag_2 = True
                    break
                completed_task = ondoc_plus.complete_task.pop(0)
                ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
                ondoc_plus.find_ready_tasks(completed_task)
                task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()
                
        if flag_2 or tar_est >= TIME_NEXT:
            ondoc_plus.advance_virtual_time(TIME_NEXT - ondoc_plus.virtual_time)
            state = next_state
            continue
        
        ondoc_plus.advance_virtual_time(tar_est - ondoc_plus.virtual_time)
        ondoc_plus.schedule_task(task, tar_p, tar_est, tar_vm)
        
        if task.processor_id in range(3) and task.end >= TIME_NEXT:  # 判断调度之后是否会影响obs,state
            next_state[task.processor_id][5+task.service_id] = 1
        if task.processor_id in range(3):
            next_state[task.processor_id][11] = task.end // ondoc_plus.TIME_STAMP
        if task.id in range(7):
            if task.lt >= tar_est:
                episode_reward += 1
            else:
                episode_reward -= 1
        if task.id == 7:
            if task.end > ondoc_plus.dags[task.k].deadline:
                episode_reward += -ondoc_plus.eta_vio * ondoc_plus.w_3
            else:
                episode_reward += ondoc_plus.eta * ondoc_plus.w_3

    while ondoc_plus.complete_task:     # 已经调度完成但是还需要等待执行
        if ondoc_plus.virtual_time % ondoc_plus.TIME_STAMP == 0:  # 每个时间戳执行一次动作
            if replay_buffer.states_mean is None:
                actions = maddpg.take_action(state, explore=True)
            else:
                actions = maddpg.take_action((state - replay_buffer.states_mean) / (replay_buffer.states_std + 1e-8), explore=True)
            step += 1
            next_state, reward, done = env.step(actions)
            for i in range(env.agents):
                ondoc_plus.servers[i].service_list = next_state[i][0:5]
            episode_reward += reward
            TIME_NEXT = ondoc_plus.TIME_STAMP * (step + 1)
        for task in ondoc_plus.complete_task:
            if task.processor_id in range(3) and task.end > TIME_NEXT:
                next_state[task.processor_id][5+task.service_id] = 1
        if ondoc_plus.complete_task[0].end >= TIME_NEXT:
            ondoc_plus.advance_virtual_time(TIME_NEXT - ondoc_plus.virtual_time)
            state = next_state
            continue
        completed_task = ondoc_plus.complete_task.pop(0)
        ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
        if completed_task.id == 7:
            if completed_task.end > ondoc_plus.dags[completed_task.k].deadline:
                episode_reward += -ondoc_plus.eta_vio * ondoc_plus.w_3
            else:
                episode_reward += ondoc_plus.eta * ondoc_plus.w_3
    done = [True, True, True]
    return episode_reward

# @torch.no_grad() 
# def train(env, ondoc_plus, maddpg, replay_buffer, num_episodes, explore):
total_step = 0
minimal_size = 4000
return_list = []
i_episode = 0
while i_episode < num_episodes:
    ondoc_plus.virtual_time = 0.0
    state = env.reset()
    ondoc_plus.receive_dag()
    for i in range(env.agents):        # 由于self.processors信息在environment.py中定义，所以这里需要重新初始化
        ondoc_plus.processors[i].task_list = []
        ondoc_plus.processors[i].service_list = [0,0,0,0,0]
    for vm in ondoc_plus.processors[3].vms:
        vm.task_list = []
    episode_limit = ondoc_plus.args.episode_limit
    arrive_len = len(ondoc_plus.arrive_list)
    k = 0
    ondoc_plus.arrive(k)
    k += 1
    step = -1 
    next_state = None
    while (any(queue for queue in ondoc_plus.queues) or k < arrive_len) and (step < episode_limit):  # 调度还没有结束
        flag_1 ,flag_2= False, False
        if ondoc_plus.virtual_time % ondoc_plus.TIME_STAMP == 0:  # 每个时间戳执行一次动作
            if next_state is not None:
                replay_buffer.add(state, actions, reward_store, next_state, done)  # 这里存储的是上一次动作之后state经过完整时隙后的信息               
                state = next_state
            if replay_buffer.states_mean is None:
                actions = maddpg.take_action(state, explore=True)
            else:
                actions = maddpg.take_action((state - replay_buffer.states_mean) / (replay_buffer.states_std + 1e-8), explore=True)
            step += 1
            next_state, reward, done = env.step(actions)
            for i in range(env.agents):
                ondoc_plus.servers[i].service_list = next_state[i][0:5]
            total_step += 1
            if replay_buffer.size() >= minimal_size and total_step % update_interval == 0:
                replay_buffer.compute_mean_std()
                sample = replay_buffer.sample(batch_size)
                sample = [stack_array(x) for x in sample]
                for a_i in range(env.agents):
                    maddpg.update(sample, a_i)
                maddpg.update_all_targets()
                
            TIME_NEXT = ondoc_plus.TIME_STAMP * (step + 1)
        
        for task in ondoc_plus.complete_task:
            if task.processor_id in range(3) and task.end > TIME_NEXT:
                next_state[task.processor_id][5+task.service_id] = 1
        
        while (not ondoc_plus.ready_tasks and ondoc_plus.complete_task) or (not ondoc_plus.ready_tasks and k < arrive_len):
            if not ondoc_plus.complete_task:
                if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                    flag_1 = True
                    break
                ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
                ondoc_plus.arrive(k)
                k += 1
            elif k >= arrive_len:
                if ondoc_plus.complete_task[0].end >= TIME_NEXT:
                    flag_1 = True
                    break
                completed_task = ondoc_plus.complete_task.pop(0)
                ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
                ondoc_plus.find_ready_tasks(completed_task)
            elif ondoc_plus.arrive_list[k] <= ondoc_plus.complete_task[0].end:
                if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                    flag_1 = True
                    break
                ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
                ondoc_plus.arrive(k)
                k += 1
            else:
                if ondoc_plus.complete_task[0].end >= TIME_NEXT:
                    flag_1 = True
                    break
                completed_task = ondoc_plus.complete_task.pop(0)
                ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
                ondoc_plus.find_ready_tasks(completed_task)
                
        if flag_1:
            ondoc_plus.advance_virtual_time(TIME_NEXT - ondoc_plus.virtual_time)
            reward_store = [reward for i in range(env.agents)]
            continue    
        if any(queue for queue in ondoc_plus.queues):
            task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()    # 每当有DAG加入或者任务完成调度之后都要调用
        
        
        # 下一段可否删除
        if not any(queue for queue in ondoc_plus.queues) and k < arrive_len:
            if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                ondoc_plus.advance_virtual_time(TIME_NEXT - ondoc_plus.virtual_time)
                continue
            ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
            ondoc_plus.arrive(k)
            k += 1
            task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()     
            
        while (len(ondoc_plus.complete_task) != 0 and ondoc_plus.complete_task[0].end <= tar_est) or (k < arrive_len and ondoc_plus.arrive_list[k] <= tar_est):
            if k < arrive_len:
                if len(ondoc_plus.complete_task) != 0:
                    if ondoc_plus.arrive_list[k] <= ondoc_plus.complete_task[0].end:
                        if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                            flag_2 = True
                            break
                        ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
                        ondoc_plus.arrive(k)
                        k += 1
                        task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()
                    else:
                        if ondoc_plus.complete_task[0].end >= TIME_NEXT:
                            flag_2 = True
                            break
                        completed_task = ondoc_plus.complete_task.pop(0)
                        ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
                        ondoc_plus.find_ready_tasks(completed_task)
                        task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()
                else:
                    if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                        flag_2 = True
                        break
                    ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
                    ondoc_plus.arrive(k)
                    k += 1
                    task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()
            else:
                if ondoc_plus.complete_task[0].end >= TIME_NEXT:
                    flag_2 = True
                    break
                completed_task = ondoc_plus.complete_task.pop(0)
                ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
                ondoc_plus.find_ready_tasks(completed_task)
                task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()
                
        if flag_2 or tar_est >= TIME_NEXT:
            ondoc_plus.advance_virtual_time(TIME_NEXT - ondoc_plus.virtual_time)
            reward_store = [reward for i in range(env.agents)]
            continue
        
        ondoc_plus.advance_virtual_time(tar_est - ondoc_plus.virtual_time)
        ondoc_plus.schedule_task(task, tar_p, tar_est, tar_vm)
        
        if task.processor_id in range(3) and task.end >= TIME_NEXT:  # 判断调度之后是否会影响obs,state
            next_state[task.processor_id][5+task.service_id] = 1
        if task.processor_id in range(3):
            next_state[task.processor_id][11] = task.end // ondoc_plus.TIME_STAMP
        if task.id in range(7):
            if task.lt >= tar_est:
                reward += 1
            else:
                reward -= 1
        if task.id == 7:
            if task.end > ondoc_plus.dags[task.k].deadline:
                reward += -ondoc_plus.eta_vio * ondoc_plus.w_3
            else:
                reward += ondoc_plus.eta * ondoc_plus.w_3

    while ondoc_plus.complete_task:     # 已经调度完成但是还需要等待执行
        if ondoc_plus.virtual_time % ondoc_plus.TIME_STAMP == 0:  # 每个时间戳执行一次动作
            actions = maddpg.take_action(state, explore=True)
            step += 1
            next_state, reward, done = env.step(actions)
            for i in range(env.agents):
                ondoc_plus.servers[i].service_list = next_state[i][0:5]
            total_step += 1
            if replay_buffer.size() >= minimal_size and total_step % update_interval == 0:
                sample = replay_buffer.sample(batch_size)
                sample = [stack_array(x) for x in sample]
                for a_i in range(env.agents):
                    maddpg.update(sample, a_i)
                maddpg.update_all_targets()
                TIME_NEXT = ondoc_plus.TIME_STAMP * (step + 1)
            
        for task in ondoc_plus.complete_task:
            if task.processor_id in range(3) and task.end > TIME_NEXT:
                next_state[task.processor_id][5+task.service_id] = 1
                
        if ondoc_plus.complete_task[0].end >= TIME_NEXT:
            ondoc_plus.advance_virtual_time(TIME_NEXT - ondoc_plus.virtual_time)
            reward_store = [reward for i in range(env.agents)]
            replay_buffer.add(state, actions, reward_store, next_state, done)
            state = next_state
            continue
        completed_task = ondoc_plus.complete_task.pop(0)
        ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
        if completed_task.id == 7:
            if completed_task.end > ondoc_plus.dags[completed_task.k].deadline:
                reward += -ondoc_plus.eta_vio * ondoc_plus.w_3
            else:
                reward += ondoc_plus.eta * ondoc_plus.w_3
    done = [True, True, True]
    reward_store = [reward for i in range(env.agents)]
    replay_buffer.add(state, actions, reward_store, next_state, done)
    i_episode += 1
    if (i_episode+1) % 100 == 0:
        returns = evaluate(maddpg, env, ondoc_plus)
        return_list.append(returns)
        print(f"Episode:{i_episode + 1},{returns} SR:{ondoc_plus.str()}%")
        
plt.plot(range(len(return_list)), return_list)
plt.xlabel('Episodes')
plt.ylabel('Episode reward')
result = 'MADDPG-ac{}-cr-{}-batch_size{}.png'.format(actor_lr, critic_lr, batch_size)
plt.savefig(args.result_dir + result, format='png')
        

