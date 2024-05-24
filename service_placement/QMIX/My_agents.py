import random
import numpy as np
import torch
from .policy import QMIX
from torch.distributions import Categorical
from .network import DRQN
from .network import QMIXNET
from torch import nn

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
        input_shape = self.obs_shape
        self.eval_hidden = torch.zeros((1, self.args.n_agents, self.args.drqn_hidden_dim)).cuda()
        self.target_hidden = torch.zeros((1, self.args.n_agents, self.args.drqn_hidden_dim)).cuda()
        
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents
        # 神经网络
        self.eval_drqn_net = DRQN(input_shape, args)  # 每个agent选动作的网络
        self.target_drqn_net = DRQN(input_shape, args)
        self.eval_qmix_net = QMIXNET(args)  # 把agentsQ值加起来的网络
        self.target_qmix_net = QMIXNET(args)
        self.args = args
        if self.args.cuda:
            self.eval_drqn_net.cuda()
            self.target_drqn_net.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
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
        hidden_state = self.eval_hidden[:, agent_id, :]

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device) # (61,) -> (1,61)
        mask_action_index_tensor = torch.tensor(mask_action_index, dtype=torch.long).to(self.device)

        # get q value  42dim
        q_value, self.eval_hidden[:, agent_id, :] = self.eval_drqn_net(inputs, hidden_state)
        
        # choose action form q value
        q_value[0, mask_action_index_tensor] = -float('inf')           # mask action illegal
        if np.random.uniform() < epsilon:          # random choose from avail_action
            action = random.choice(avail_action)
        else:                                      # choose the max q value
            action = q_value.argmax(dim=1).item()
            # action = env_avail_actions[action_idx]
        return action



 