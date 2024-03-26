import numpy as np
class Server:
    def __init__(self, id, comp):
        self.id = id
        self.task_list = []
        self.comp = comp
    
class Remote_cloud:
    def __init__(self, id, comp):
        self.id = id
        self.task_list = []
        self.comp = comp
        
np.random.seed(1)
comp = np.random.randint(low=1500, high=2001, size=5)   # modify comp variable to generate random values from a uniform distribution between 1000 and 2000
servers = [Server(i, comp[i]) for i in range(5)]  # access the servers list
cloud = Remote_cloud(5, 4000)                    
B_u = 7.81
B_c = 80
B_e = 17.77  # ns/B
B_aver = (4*B_e+B_c)/5
interval_list = np.random.exponential(0.1,200)       # 模拟每个DAG之间的时间间隔 size=200
request_list = np.random.randint(0, 5, size=200)   # 每个请求从哪个服务器发出 size=200

class Task:
    def __init__(self, id, k, deadline=None):
        self.id = id
        self.k = k
        self.processor_id = None    # 0-4 is edge 5 is cloud 
        self.rank = None
        self.avg_comp = None    # average computation time of each task on all servers
        self.duration = {'start': None, 'end': float("inf")}
        self.deadline = deadline
        self.lt = None
        
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

