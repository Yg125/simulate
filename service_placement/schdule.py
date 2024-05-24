import random
from Configure import Task, DAG, B_aver, B_c, B_e, B_u, interval_list, request_list, TIME_STAMP, Args, eta_vio, eta, w_3
from collections import deque
import numpy as np
from Env import ScheduleEnv, servers, cloud
from QMIX.agents import Agents
from QMIX.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import torch
import copy
import time
from QMIX.network import DRQN

random.seed(1)
processors = servers + [cloud]      # servers编号为0-4, cloud编号为5
env = ScheduleEnv()
    
class OnDoc_plus:
    def __init__(self, args, params):
        self.Q = 100
        global processors  
        self.queues = [0 for _ in range(self.Q)]
        self.num_processors = 6
        self.dags = [0 for _ in range(self.Q)]
        self.arrive_list = [0 for _ in range(self.Q)]
        self.ready_tasks = set()
        self.virtual_time = 0.0        # 定义在线场景的虚拟时间，避免受机器运行时间干扰
        self.complete_task = []        # COFE算法中记录每个子任务的完成时间
        self.args = args
        self.buffer = ReplayBuffer(args)
        self.graph = np.empty((self.Q,8,8))
        self.comp_cost = np.empty((self.Q,8,6))
        self.tasks = [0 for _ in range(self.Q)]
        self.processors = copy.deepcopy(processors)
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.device = args.device
        
        input_shape = args.obs_shape
        if args.last_action:
            input_shape += args.n_actions
        if args.reuse_network:
            input_shape += args.n_agents
        # 神经网络
        self.eval_drqn_net = DRQN(input_shape, args)  # 每个agent选动作的网络
        self.eval_drqn_net.load_state_dict(params)
        
        self.eval_hidden = torch.zeros((1, self.args.n_agents, self.args.drqn_hidden_dim))
        self.target_hidden = torch.zeros((1, self.args.n_agents, self.args.drqn_hidden_dim))
    
    def choose_action(self, obs, last_action, agent_id, avail_action, epsilon, evaluate=False):
        inputs = obs.copy()
        mask_action_index = [mask for mask in range(self.n_actions) if mask not in avail_action]
        agents = np.zeros(self.n_agents)
        agents[agent_id] = 1.
        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agents))
        hidden_state = self.eval_hidden[:, agent_id, :]

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0) # (61,) -> (1,61)
        mask_action_index_tensor = torch.tensor(mask_action_index, dtype=torch.long)

        # get q value  42dim
        q_value, self.eval_hidden[:, agent_id, :] = self.eval_drqn_net(inputs, hidden_state)
        
        # choose action form q value
        q_value[0, mask_action_index_tensor] = -float('inf')           # mask action illegal
        if np.random.uniform() < epsilon:          # random choose from avail_action
            action = random.choice(avail_action)
        else:                                      # choose the max q value
            action = q_value.argmax(dim=1).item()
        return action
        
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
                curr_rank = max(curr_rank, round(self.graph[k][task.id][succ.id]*B_aver/10**6, 1) + succ.rank)
        task.rank = task.avg_comp + curr_rank
        computed_ranks[task] = task.rank

    def LT(self, k):
        for t in self.dags[k].tasks:
            t.lt = self.dags[k].r + (self.dags[k].deadline - self.dags[k].r) * (self.dags[k].tasks[0].rank - t.rank) / self.dags[k].tasks[0].rank
            
    def receive_dag(self):                    # k is the index of the request/queue from 0
        tasks = [0 for _ in range(self.Q)]
        DAGS = np.load('/home/yangang/DAG_Scheduling/dag_info_6.npy', allow_pickle=True)  # Read DAG from file
        for k in range(self.Q):
            self.dags[k] = DAG(k)
            self.dags[k].num_tasks, self.comp_cost[k], self.graph[k], deadline_heft = DAGS[k]   
            self.dags[k].deadline = self.virtual_time + deadline_heft * 1.3   # 以heft算法的deadline为基准，增加15%的时间作为在线场景DAG的deadline
            self.dags[k].r = self.virtual_time    # ms
            self.arrive_list[k] = self.virtual_time
            num_tasks = self.dags[k].num_tasks
            tasks[k] = [Task(i,k) for i in range(num_tasks)]
            data_in = 0
            for j in range(self.dags[k].num_tasks):
                tasks[k][j].avg_comp = sum(self.comp_cost[k][j]) / self.num_processors
                if self.graph[k][0][j] != -1:
                    data_in += self.graph[k][0][j]
                else:
                    data_in += 0
            self.dags[k].t_offload = round(data_in * B_u / 10**6, 1)  # 任务由用户发送到服务器需要的offload时间
            interval = interval_list[k] * 1000
            self.advance_virtual_time(interval)
        return tasks
        
    def arrive(self, k):
        tasks = self.tasks[k]
        computed_ranks = {}
        self.computeRank(tasks[0], k, computed_ranks)
        try:
            self.queues[k] = deque(tasks)
        except:
            print(1)
        self.ready_tasks.add(self.queues[k][0])
    
    def take_actions(self, last_action, o, s, u, u_onehot, avail_u, r, terminate, padded, step, episode_reward, epsilon):
        # 在时隙开始时做出动作并更新
        n_agents = self.args.n_agents
        n_actions = self.args.n_actions
        obs = env.get_obs()  # array([[],[]])    要存储的obs，state，actions，rewards，avail_actions，avail_actions_next，done，padded
        state = env.get_state() # array([])      做了step动作之后进入下一个状态，这时候需要静待这个时隙过完才能得到正确的state_next和reward 在时隙中改变env.obs 下个时隙得到的就是正确的
        actions = [[] for _ in range(n_agents)]
        actions_onehot = [[] for _ in range(n_agents)]
        avail_actions = [[] for _ in range(n_agents)]
        for agent_id in range(n_agents):
            avail_action = env.get_avail_actions(agent_id)
            action = self.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon, evaluate=False) # 新的时隙开始先获取这个时隙的动作并step执行
            action_onehot = env.onehot(action)
            actions[agent_id] = action
            a_onehot = [0 for _ in range(n_actions)]
            for a in avail_action:
                a_onehot[a] = 1
            avail_actions[agent_id] = a_onehot
            actions_onehot[agent_id] = action_onehot
            last_action[agent_id] = action_onehot            
        _, _, reward_temp, done_temp = env.step(actions)   # 不用管这里得到的 因为会变化，下个时隙得到的才是正确的 这里得到的只是用来判断缓存策略
        o.append(obs)    # 存储的是执行动作之前的  next通过顺移即可得到
        s.append(state)
        u.append(np.reshape(actions, [n_agents, 1]))
        u_onehot.append(np.reshape(actions_onehot, [n_agents, n_actions]))
        avail_u.append(np.reshape(avail_actions, [n_agents, n_actions]))
        r.append([reward_temp])
        terminate.append([done_temp])
        padded.append([0])
        # 这里append都是加入一个元素 每个元素包含n_agents个agent的信息[[[],[],..[]],[],]
        episode_reward += reward_temp
        step += 1
        return episode_reward, step
    
    @torch.no_grad() 
    def schedule(self, evaluate=False): 
        for processor in processors:        # 由于processors信息在environment.py中定义，所以这里需要重新初始化
            processor.task_list = []
            if processor.id == 5:
                for vm in processor.vms:
                    vm.task_list = []
            else:
                processor.service_list = random.choice(env.env_avail_actions)
        n_agents = self.args.n_agents
        n_actions = self.args.n_actions
        episode_limit = self.args.episode_limit
        arrive_len = len(self.arrive_list)
        k = 0
        self.virtual_time = 0.0
        self.arrive(k)
        k += 1
        step = -1
        env.reset()   # 初始化observation state  
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        episode_reward = 0
        last_action = np.zeros((n_agents, n_actions))
        epsilon = 0 if evaluate else self.args.epsilon
        # 下面的调度都是在一个时间隙内进行调度 并及时更新时间隙和环境状态 
        # 编写时隙内的调度 加上缓存限制 在每一次推进虚拟时间的时候判断是否超过了时间隙 如果是则立即停止之后的代码并执行动作决策和放置 再重新开始新的时隙循环
        while (any(queue for queue in self.queues) or k < arrive_len) and (step < episode_limit):  
            flag_1 ,flag_2= False, False
            if self.virtual_time % TIME_STAMP == 0:  # 每个时间戳执行一次动作
                episode_reward, step = self.take_actions(last_action, o, s, u, u_onehot, avail_u, r, terminate, padded, step, episode_reward, epsilon)
                epsilon = epsilon - self.args.anneal_epsilon if epsilon > self.args.end_epsilon else epsilon
            TIME_NEXT = TIME_STAMP * (step + 1)
            
            while (not self.ready_tasks and self.complete_task) or (not self.ready_tasks and k < arrive_len):
                if not self.complete_task:
                    if self.arrive_list[k] >= TIME_NEXT:
                        flag_1 = True
                        break
                    self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                    self.arrive(k)
                    k += 1
                elif k >= arrive_len:
                    if self.complete_task[0].end >= TIME_NEXT:
                        flag_1 = True
                        break
                    completed_task = self.complete_task.pop(0)
                    self.advance_virtual_time(completed_task.end - self.virtual_time)
                    self.find_ready_tasks(completed_task)
                elif self.arrive_list[k] <= self.complete_task[0].end:
                    if self.arrive_list[k] >= TIME_NEXT:
                        flag_1 = True
                        break
                    self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                    self.arrive(k)
                    k += 1
                else:
                    if self.complete_task[0].end >= TIME_NEXT:
                        flag_1 = True
                        break
                    completed_task = self.complete_task.pop(0)
                    self.advance_virtual_time(completed_task.end - self.virtual_time)
                    self.find_ready_tasks(completed_task)
            if flag_1:
                self.advance_virtual_time(TIME_NEXT - self.virtual_time)
                continue    
            if any(queue for queue in self.queues):
                task, tar_p, tar_est, tar_vm = self.check_ready()    # 每当有DAG加入或者任务完成调度之后都要调用
                
            if not any(queue for queue in self.queues) and k < arrive_len:
                if self.arrive_list[k] >= TIME_NEXT:
                    self.advance_virtual_time(TIME_NEXT - self.virtual_time)
                    continue
                self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                self.arrive(k)
                k += 1
                task, tar_p, tar_est, tar_vm = self.check_ready()     
                
            while (len(self.complete_task) != 0 and self.complete_task[0].end <= tar_est) or (k < arrive_len and self.arrive_list[k] <= tar_est):
                if k < arrive_len:
                    if len(self.complete_task) != 0:
                        if self.arrive_list[k] <= self.complete_task[0].end:
                            if self.arrive_list[k] >= TIME_NEXT:
                                flag_2 = True
                                break
                            self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                            self.arrive(k)
                            k += 1
                            task, tar_p, tar_est, tar_vm = self.check_ready()
                        else:
                            if self.complete_task[0].end >= TIME_NEXT:
                                flag_2 = True
                                break
                            completed_task = self.complete_task.pop(0)
                            self.advance_virtual_time(completed_task.end - self.virtual_time)
                            self.find_ready_tasks(completed_task)
                            task, tar_p, tar_est, tar_vm = self.check_ready()
                    else:
                        if self.arrive_list[k] >= TIME_NEXT:
                            flag_2 = True
                            break
                        self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                        self.arrive(k)
                        k += 1
                        task, tar_p, tar_est, tar_vm = self.check_ready()
                else:
                    if self.complete_task[0].end >= TIME_NEXT:
                        flag_2 = True
                        break
                    completed_task = self.complete_task.pop(0)
                    self.advance_virtual_time(completed_task.end - self.virtual_time)
                    self.find_ready_tasks(completed_task)
                    task, tar_p, tar_est, tar_vm = self.check_ready()
                    
            if flag_2 or tar_est >= TIME_NEXT:
                self.advance_virtual_time(TIME_NEXT - self.virtual_time)
                continue
            
            self.advance_virtual_time(tar_est - self.virtual_time)
            self.schedule_task(task, tar_p, tar_est, tar_vm)
            
            for task in self.complete_task:
                if task.end >= TIME_NEXT and task.processor_id in range(5):  # 判断调度之后是否会影响obs,state
                    env.get_obs()[task.processor_id][6+task.service_id] = 1
                    env.get_state()[task.processor_id*9+7] = task.end
                    
            if task.id == 7:
                if task.end > self.dags[task.k].deadline:
                    r[-1][0] += -eta_vio * w_3
                    episode_reward += -eta_vio * w_3
                else:
                    r[-1][0] += eta * w_3
                    episode_reward += eta * w_3
        self.queues = [0 for _ in range(self.Q)] # 调度完成，清空队列 以供下次调度      
        while self.complete_task:
            if self.virtual_time % TIME_STAMP == 0:  # 每个时间戳执行一次动作
                episode_reward, step = self.take_actions(last_action, o, s, u, u_onehot, avail_u, r, terminate, padded, step, episode_reward, epsilon)
                epsilon = epsilon - self.args.anneal_epsilon if epsilon > self.args.end_epsilon else epsilon
                TIME_NEXT = TIME_STAMP * (step + 1)
            if self.complete_task[0].end >= TIME_NEXT:
                self.advance_virtual_time(TIME_NEXT - self.virtual_time)
                continue
            completed_task = self.complete_task.pop(0)
            self.advance_virtual_time(completed_task.end - self.virtual_time)
            if completed_task.id == 7:
                if completed_task.end > self.dags[completed_task.k].deadline:
                    r[-1][0] += -eta_vio * w_3
                    episode_reward += -eta_vio * w_3
                else:
                    r[-1][0] += eta * w_3
                    episode_reward += eta * w_3
        terminate[-1][0] = 1
                    
        # 执行完这个时隙后 obs会变化 需要将新的obs state更新到env中 并将新的obs state存入buffer
        # last obs
        obs = env.get_obs()  # 此时已经调度完成 obs包含最后一个状态 并可以从中得出avail_action
        state = env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(n_agents):
            avail_action = env.get_avail_actions(agent_id)
            a_onehot = np.zeros(n_actions)
            for a in avail_action:
                a_onehot[a] = 1
            avail_actions.append(a_onehot)
        avail_u.append(np.reshape(avail_actions, [n_agents, n_actions]))
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # 补齐episode_limit 用于存储
        for i in range(step+1, episode_limit):
            o.append(np.zeros((n_agents, self.args.obs_shape)))
            u.append(np.zeros((n_agents, 1)))
            s.append(np.zeros(self.args.state_shape))
            r.append([0.])
            o_next.append(np.zeros((n_agents, self.args.obs_shape)))
            s_next.append(np.zeros(self.args.state_shape))
            u_onehot.append(np.zeros((n_agents, n_actions)))   
            avail_u.append(np.zeros((n_agents, n_actions)))    # 调整 avail_actions也只存可选动作的index
            avail_u_next.append(np.zeros((n_agents, n_actions)))
            padded.append([1])
            terminate.append([1])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        # print(step)            
        return episode, episode_reward, step
    
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
                self.ready_tasks.add(succ)
        
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
        if p in range(5):
            processors[p].task_list.append(task)
        else:
            cloud.vms[vm].task_list.append(task)
        if task.id != 7:
            self.complete_task.append(task)
        
        self.complete_task.sort(key=lambda x: x.end)
        try:
            self.ready_tasks.discard(task)
        except:
            print('DAG:{} Task {} is not in ready_tasks'.format(task.k,task.id))
   
    def get_est(self, t, p, k): 
        if p.id in range(5) and t.service_id not in p.service_list and (t.id != 0 and t.id != 7):
            return float('inf')
        est = max(self.dags[k].r + self.dags[k].t_offload, self.virtual_time)    # 初始化est时间为任务到达时间和offload时间之和
        graph = self.graph[k]
        tasks = self.tasks[k]
        
        for pre in tasks:
            if graph[pre.id][t.id] != -1:  # if pre also done on p, no communication cost
                c = graph[pre.id][t.id] if pre.processor_id != p.id else 0
                if pre.processor_id in range(5) and p.id in range(5): 
                    est = max(est, pre.end + round(c*B_e/10**6, 1))  # ms
                else:
                    est = max(est, pre.end + round(c*B_c/10**6, 1))
        if p.id in range(5) and not p.task_list:  # 在之前没有任务则直接返回任务依赖的EST
            return est
        elif p.id == 5:
            est_cloud = float('inf')
            vm_i = None 
            for i, vm in enumerate(cloud.vms):
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
            tar_p = request_list[t.k]         # 随机从某个边缘服务器发出请求
            tar_est = self.get_est(t, processors[tar_p], t.k)
            return (tar_p, tar_est)
        
        elif t.id == self.tasks[t.k][-1].id:
            tar_p = self.tasks[t.k][0].processor_id
            tar_est = self.get_est(t, processors[tar_p], t.k)
            return (tar_p, tar_est)
        else:
            aft = float("inf")
            for processor in processors:
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
            if tar_p == 5:
                return (tar_p, tar_est, vm_i)
            else:
                return (tar_p, tar_est)
        
    def str(self):
        print_str = ""
        satisfy = 0
        Makespan = 0
        # for p in processors[0:5]:
        #     print_str += 'Processor {}:\n'.format(p.id+1)
        #     for t in p.task_list:
        #         print_str += 'Dag {}, Task {}: start = {}, end = {}\n'.format(t.k, t.id ,t.start, t.end)
        # print_str += 'Cloud:\n'
        # for i in range(len(cloud.vms)):
        #     print_str += 'VM {}:\n'.format(i+1)
        #     for t in cloud.vms[i].task_list:
        #         print_str += 'Dag {}, Task {}: start = {}, end = {}\n'.format(t.k, t.id ,t.start, t.end)
        for k in range(self.Q):
            self.Makespan = max([t.end for t in self.tasks[k]]) - self.dags[k].r
            Makespan += self.Makespan
            if self.Makespan < self.dags[k].deadline - self.dags[k].r:
                satisfy += 1
            # print_str += "Makespan{} = {}\n".format(k, self.Makespan)
        SR = (satisfy / self.Q) * 100
        average_makespan = Makespan / self.Q
        print_str += "Mine:SR = {}%\n".format(SR)
        print_str += "Mine:Average Makespan = {}\n".format(average_makespan)
        return print_str

    def run(self):
        tasks = self.receive_dag()
        train_steps = 0
        episode_rewards = []
        start_time = time.time()
        # for epoch in range(self.args.n_epochs):
        for epoch in range(1):
            # if epoch % self.args.evaluate_per_epoch == 0:
            #     evaluate_rewards = 0
            #     for e_epoch in range(self.args.evaluate_epoch):
            #         # print('evaluate = {}'.format(epoch))
            #         self.tasks = [[copy.copy(task) for task in subtasks] for subtasks in tasks]
            #         _, evaluate_reward, step = self.schedule(evaluate=True)
            #         evaluate_rewards += evaluate_reward
            #     evaluate_reward = evaluate_rewards / self.args.evaluate_epoch
            #     episode_rewards.append(evaluate_reward)
            #     print('epoch = {}  episode_reward = {} step = {}'.format(epoch, evaluate_reward, step))
            
            episodes = []
            for episode_idx in range(self.args.n_eposodes):
                # print('episode : {}'.format(episode_idx))
                self.tasks = [[copy.copy(task) for task in subtasks] for subtasks in tasks]
                episode, _, _= self.schedule(evaluate=False)
                episodes.append(episode)
        end_time = time.time()
        total_time = end_time - start_time
        print("Total time: ", total_time)
        #     episode_batch = episodes[0]
        #     episodes.pop(0)
        #     for episode in episodes:
        #         for key in episode_batch.keys():
        #             episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
                    
        #     self.buffer.store_episode(episode_batch)
        #     for train_step in range(self.args.train_steps):
        #         mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
        #         self.agents.train(mini_batch, train_steps)
        #         train_steps += 1
        
        # evaluate_rewards = 0
        # for e_epoch in range(self.args.evaluate_epoch):
        #     # print('evaluate = {}'.format(epoch))    
        #     self.tasks = [[copy.copy(task) for task in subtasks] for subtasks in tasks]
        #     _, evaluate_reward, step = self.schedule(evaluate=True)
        #     evaluate_rewards += evaluate_reward
        # evaluate_reward = evaluate_rewards / self.args.evaluate_epoch
        # episode_rewards.append(evaluate_reward)
        # print('epoch = {}  episode_reward = {} step = {}'.format(epoch, evaluate_reward, step))
            
        # self.plt(episode_rewards)


    def plt(self, episode_rewards):
        
        plt.plot(range(len(episode_rewards)), episode_rewards)
        plt.xlabel('Epoch*{}'.format(self.args.n_epochs))
        plt.ylabel('episode reward')
        plt.savefig(self.args.result_dir + 'result1.png', format='png')
        
# args = Args()
# args.set_env_info(env.get_info())
# ondoc = OnDoc_plus(args)
# ondoc.receive_dag()
# # episode, episode_reward = ondoc.schedule()
# ondoc.run()



# str = ondoc.str()
# print(str)

