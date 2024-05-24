import numpy as np
import threading
np.random.seed(1)

class SumTree:
    def __init__(self, args):
        self.data_pointer = 0
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size   #size对应有几个episode 存的一条信息就是一个episode
        self.episode_limit = self.args.episode_limit
        self.n_entries = 0
        self.tree = np.zeros(2 * self.size - 1)
        self.data = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                    'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                    's': np.empty([self.size, self.episode_limit, self.state_shape]),
                    'r': np.empty([self.size, self.episode_limit, 1]),
                    'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                    's_next': np.empty([self.size, self.episode_limit, self.state_shape]),
                    'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                    'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                    'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                    'padded': np.empty([self.size, self.episode_limit, 1]),
                    'terminated': np.empty([self.size, self.episode_limit, 1])
                    }

    def add(self, p, data):
        tree_idx = self.data_pointer + self.size - 1
        # self.data[self.data_pointer] = data  # update data_frame
        self.data['o'][self.data_pointer] = data['o']
        self.data['u'][self.data_pointer] = data['u']
        self.data['s'][self.data_pointer] = data['s']
        self.data['r'][self.data_pointer] = data['r']
        self.data['o_next'][self.data_pointer] = data['o_next']
        self.data['s_next'][self.data_pointer] = data['s_next']
        self.data['avail_u'][self.data_pointer] = data['avail_u']
        self.data['avail_u_next'][self.data_pointer] = data['avail_u_next']
        self.data['u_onehot'][self.data_pointer] = data['u_onehot']
        self.data['padded'][self.data_pointer] = data['padded']
        self.data['terminated'][self.data_pointer] = data['terminated']
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.size:  # replace when exceed the size
            self.data_pointer = 0
        
        if self.n_entries < self.size:
            self.n_entries += 1

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.size + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_p(self):
        return self.tree[0]  # the root

class Memory:
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error
    
    def __init__(self, args):
        self.tree = SumTree(args)
        self.lock = threading.Lock()
        self.current_size = 0

        # store the episode
    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]  # 存储的batch有多少条episode
        with self.lock:
            max_p = np.max(self.tree.tree[-self.tree.size:])
            if max_p == 0:
                max_p = self.abs_err_upper
            for i in range(batch_size):
                data = {}
                for key in episode_batch.keys():
                    data[key] = episode_batch[key][i]
                self.tree.add(max_p, data)
        self.current_size = min(self.current_size + batch_size, self.tree.size)

    # def sample(self, batch_size):
    #     t_idx, b_sample, ISWeights = np.empty((batch_size,), dtype=np.int32), {}, np.empty((batch_size, 1))
    #     pri_seg = self.tree.total_p / batch_size       # priority segment
    #     self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

    #     min_prob = np.min(self.tree.tree[-self.tree.size:]) / self.tree.total_p     # for later calculate ISweight
    #     idxs = np.empty((batch_size,), dtype=np.int32)
    #     for i in range(batch_size):
    #         a, b = pri_seg * i, pri_seg * (i + 1)
    #         v = np.random.uniform(a, b)
    #         idx, p, data_idx = self.tree.get_leaf(v)
    #         prob = p / self.tree.total_p
    #         ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
    #         t_idx[i], idxs[i] = idx, data_idx
    #     for key in self.tree.data.keys():
    #         b_sample[key] = self.tree.data[key][idxs]
    #     return t_idx, b_sample, ISWeights
    
    def sample(self, batch_size):
        t_idx, b_sample = np.empty((batch_size,), dtype=np.int32), {}
        segment = self.tree.total_p / batch_size       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        idxs = np.empty((batch_size,), dtype=np.int32)
        priorities = []
        
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            p = 0
            while p == 0:
                s = np.random.uniform(a, b)
                (idx, p, data_idx) = self.tree.get_leaf(s)
            priorities.append(p)
            t_idx[i], idxs[i] = idx, data_idx

        sampling_probabilities = priorities / self.tree.total_p
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        is_weight = is_weight[:, np.newaxis]
        for key in self.tree.data.keys():
            b_sample[key] = self.tree.data[key][idxs]
        # print(is_weight)
        return t_idx, b_sample, is_weight
    
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        # clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        
        scaled_td_error = np.round(np.abs(abs_errors) / 100, decimals=1)  # 缩小100倍避免total过大
        self.abs_err_upper = max(self.abs_err_upper, max(scaled_td_error))
        
        ps = np.power(scaled_td_error, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)