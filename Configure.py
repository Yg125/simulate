import torch
import numpy as np

NUM_AGENTS = 5
NUM_TASKS = 20
Lambda = 3
Q = 200

MAX_TIME = 30000
TIME_STAMP = 1000 # ms
np.random.seed(1)
B_u = 7.81
B_c = 80
B_e = 17.77  # ns/B
B_aver = ((NUM_AGENTS-1)*B_e+B_c)/NUM_AGENTS


w_1 = 0.2 # 10
w_2 = 0.1  # 1
w_3 = 0.7 # 0.2
mu_c = 0.08
Gamma = 0.04
eta_vio = 20
eta = 10

class Task:
    def __init__(self, id, k, type=None, deadline=None):
        self.id = id
        self.k = k
        self.processor_id = None    # 0-4 is edge 5 is cloud 
        self.rank = None
        self.avg_comp = None    # average computation time of each task on all servers
        self.start = None
        self.end = float("inf")
        # self.duration = {'start': None, 'end': float("inf")}
        self.deadline = deadline
        self.lt = None
        self.service_id = type # 从0-4中随机选择一种service
        
class DAG:
    def __init__(self, k):
        self.k = k
        self.r = None  # arrival time of each DAG
        self.deadline = None
        self.num_tasks = None
        self.tasks = []
        self.comp_cost = []
        self.graph = []
        self.t_offload = None
        self.makespan = None

class Args:
    def __init__(self):
        self.train = True
        self.seed = 133
        self.cuda = True

        # train setting
        self.last_action = True  # 使用最新动作选择动作
        self.reuse_network = True  # 对所有智能体使用同一个网络
        self.n_epochs = 100000  # 20000
        self.evaluate_epoch = 2  # 20
        self.evaluate_per_epoch = 100  # 100
        self.batch_size = 64  # 32
        self.buffer_size = int(4e5)
        self.save_frequency = 5000  # 5000
        self.n_eposodes = 5  # 每个epoch有多少episodes
        self.train_steps = 1  # 每个epoch有多少train steps
        self.gamma = 0.99
        self.grad_norm_clip = 10  # prevent gradient explosion
        self.target_update_cycle = 50  # 200
        self.result_dir = './results/'
# 学习率 buffer_size 调试看weights
        # test setting
        self.load_model = False
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # model structure
        # drqn net
        self.drqn_hidden_dim = 64
        # qmix net
        # input: (batch_size, n_agents, qmix_hidden_dim)
        self.qmix_hidden_dim = 32
        self.model_dir = './models/'
        self.lr = 1e-2
        self.start_size = 1e3
        
        # coefficient
        self.w_1 = w_1 # 0.3
        self.w_2 = w_2  # 0.2
        self.w_3 = w_3 # 0.5
        
        # epsilon greedy
        self.epsilon = 1
        self.end_epsilon = 0.05
        self.anneal_steps = 1300000  # 100000
        self.anneal_epsilon = (self.epsilon - self.end_epsilon) / self.anneal_steps
        self.epsilon_anneal_scale = 'step'

    def set_env_info(self, env_info):
        self.n_actions = env_info["n_actions"]  # 26
        self.state_shape = env_info["state_shape"]
        self.n_agents = env_info["n_agents"]
        self.episode_limit = env_info["episode_limit"]
