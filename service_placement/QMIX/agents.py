import random
import time
import numpy as np
import torch
from .policy import QMIX
from torch.distributions import Categorical

random.seed(1)
np.random.seed(1)

class Agents:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.n_actions = args.n_actions 
        self.n_agents = args.n_agents 
        self.state_shape = args.state_shape 
        self.obs_shape = args.obs_shape
        self.episode_limit = args.episode_limit
        
        self.policy = QMIX(args)

        # print("Agents inited!")
                                                            # [, , ,]
    def choose_action(self, obs, last_action, agent_id, avail_action, epsilon, evaluate=False):
        inputs = obs.copy()
        # avail_action_index = []
        # for action in avail_action:
            # avail_action_index.append(env_avail_actions.index(action)) #得到可选动作在q输出索引
        # mask_action_index = [mask for mask in range(self.n_actions) if mask not in avail_action_index]
        mask_action_index = [mask for mask in range(self.n_actions) if mask not in avail_action]
        
        # avail_action_idx = np.nonzero(avail_action)[0]
        agents = np.zeros(self.n_agents)
        agents[agent_id] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agents))
        hidden_state = self.policy.eval_hidden[:, agent_id, :]

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device) # (61,) -> (1,61)
        mask_action_index_tensor = torch.tensor(mask_action_index, dtype=torch.long).to(self.device)

        # get q value  42dim
        q_value, self.policy.eval_hidden[:, agent_id, :] = self.policy.eval_drqn_net(inputs, hidden_state)
        
        # choose action form q value
        q_value[0, mask_action_index_tensor] = -float('inf')           # mask action illegal
        if np.random.uniform() < epsilon:          # random choose from avail_action
            action = random.choice(avail_action)
        else:                                      # choose the max q value
            action = q_value.argmax(dim=1).item()
            # action = env_avail_actions[action_idx]
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch["terminated"]
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx+1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, idxs=None, memory=None, ISweights=None, epsilon=None):
        # 不同的episode的数据长度不同，因此需要得到最大长度
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        return self.policy.learn(batch, max_episode_len, train_step, idxs, memory, ISweights, epsilon)
        # if train_step > 0 and train_step % self.args.save_frequency == 0:
        #     self.policy.save_model(train_step)

 