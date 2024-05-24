import random
import time
from Configure import Task, DAG, B_aver, B_c, B_e, B_u, interval_list, request_list, TIME_STAMP, Args, eta_vio, eta, w_3
from collections import deque
import numpy as np
from Env import ScheduleEnv, servers, cloud
from QMIX.agents import Agents
from QMIX.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
import Schedule

random.seed(1)
    
class OnDoc_plus:
    def __init__(self, args):
        self.Q = 100
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
        
    def str(self):
        print_str = ""
        satisfy = 0
        Makespan = 0
        # for p in processors[0:5]:
        #     print_str += 'Processor {}:\n'.format(p.id+1)
        #     for t in p.task_list:
        #         print_str += 'Dag {}, Task {}: start = {}, end = {}\n'.format(t.k, t.id ,t.duration['start'], t.duration['end'])
        # print_str += 'Cloud:\n'
        # for i in range(len(cloud.vms)):
        #     print_str += 'VM {}:\n'.format(i+1)
        #     for t in cloud.vms[i].task_list:
        #         print_str += 'Dag {}, Task {}: start = {}, end = {}\n'.format(t.k, t.id ,t.duration['start'], t.duration['end'])
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

    def plt(self, episode_rewards):  
        plt.plot(range(len(episode_rewards)), episode_rewards)
        plt.xlabel('Epoch*{}'.format(self.args.evaluate_per_epoch))
        plt.ylabel('episode reward')
        plt.savefig(self.args.result_dir + '/result.png', format='png')
        
def show(args, episode_rewards):  
    plt.plot(range(len(episode_rewards)), episode_rewards)
    plt.xlabel('Epoch*{}'.format(args.evaluate_per_epoch))
    plt.ylabel('episode reward')
    result = 'result-lr{}-buffer_size{}-epoch{}-w1_{}-w2_{}-w3_{}.png'.format(args.lr, args.buffer_size, args.n_epochs, args.w_1, args.w_2, args.w_3)
    plt.savefig(args.result_dir + result, format='png')
    
def generate(args_params):
    args, params, tasks = args_params
    ondoc_plus = Schedule.OnDoc_plus(args, params, tasks)
    ondoc_plus.receive_dag()
    episode, _, _, epsilon= ondoc_plus.schedule(evaluate=False)
    return episode, epsilon

def evaluate(args_params):
    args, params, tasks = args_params
    ondoc_plus = Schedule.OnDoc_plus(args, params, tasks)
    ondoc_plus.receive_dag()
    _, evaluate_reward, _, _= ondoc_plus.schedule(evaluate=True)
    return evaluate_reward, ondoc_plus.str()
    
def run():
    env = ScheduleEnv()
    args = Args()
    args.set_env_info(env.get_info())
    agents = Agents(args)  # 作为参数传入子进程 这样只需要初始化一个agents 所有进程共用，更新也只更新一个
    ondoc = OnDoc_plus(args)
    tasks = ondoc.receive_dag()
    multiprocessing.set_start_method('spawn')
    train_steps = 0
    episode_rewards = []
    num_processes = 3
    loss = None
    with Pool(num_processes) as pool:
        for epoch in range(args.n_epochs):
            if epoch % args.evaluate_per_epoch == 0:
                detached_params = {k: v.detach() for k, v in agents.policy.eval_drqn_net.state_dict().items()}
                args_params_eval = [(args, detached_params, tasks) for _ in range(args.evaluate_epoch)]
                result = pool.map(evaluate, args_params_eval)
                evaluate_return = [sum(col) for col in zip(*result)]
                evaluate_reward = evaluate_return[0] / args.evaluate_epoch
                evaluate_SR = evaluate_return[1] / args.evaluate_epoch
                episode_rewards.append(evaluate_reward)
                if loss:
                    print('epoch = {}  episode_reward = {} loss = {} epsilon = {} SR = {}%'.format(epoch, evaluate_reward, loss, args.epsilon, evaluate_SR))  
                else:
                    print('epoch = {}  episode_reward = {} epsilon = {} SR = {}%'.format(epoch, evaluate_reward, args.epsilon, evaluate_SR))
                            
            detached_params = {k: v.detach() for k, v in agents.policy.eval_drqn_net.state_dict().items()}
            episodes = [0 for _ in range(args.n_eposodes)]
            args_params_gen = [(args, detached_params, tasks) for _ in range(num_processes)]
            
            results = pool.map(generate, args_params_gen)      
            args.epsilon = results[0][1]
            episodes = [result[0] for result in results]
            
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
                    
            ondoc.buffer.store_episode(episode_batch)
            if ondoc.buffer.current_size >= args.batch_size:
                mini_batch = ondoc.buffer.sample(min(ondoc.buffer.current_size, ondoc.args.batch_size))
                loss = agents.train(mini_batch, train_steps)
                train_steps += 1

    if epoch % args.evaluate_per_epoch == 0:
            evaluate_rewards = [0 for _ in args.evaluate_epoch]
            args_agents_list = [(args, agents) for _ in range(args.evaluate_epoch)]
            with Pool(num_processes) as pool:
                evaluate_rewards = pool.map(evaluate, args_agents_list)
            evaluate_reward = sum(evaluate_rewards) / args.evaluate_epoch
            episode_rewards.append(evaluate_reward)
            print('epoch = {}  episode_reward = {}'.format(epoch, evaluate_reward))
    show(args, episode_rewards)
        

if __name__ == '__main__':
    run()