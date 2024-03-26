import gym
from gym import spaces
import numpy as np
from Configure import server_capacity, service_size

NUM_AGENTS = 5
NUM_ACTIONS = 6
TOTAL_TIME = 20000

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.num_agents = NUM_AGENTS
        # Define action and observation space
        action_space_each_agent = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_spaces = [action_space_each_agent for _ in range(self.num_agents)]

        self.observation_spaces = []
        for _ in range(self.num_agents):
            continuous_part = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
            binary_part = spaces.MultiBinary(6)
            single_value_part_1 = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            single_value_part_2 = spaces.Box(low=0, high=20000, shape=(1,), dtype=np.float32)
            total_space = [continuous_part, binary_part, single_value_part_1, single_value_part_2]
            self.observation_spaces.append(total_space)
            
        self.state_cur = [] # initialize the current state and will keep upodating it in the step function
        for agent_i in range(self.num_agents):
            agent_i_observation = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], server_capacity[agent_i], 0]
            self.state_cur.append(agent_i_observation)

    def reset(self):
        # Reset the environment to its initial state
        # Return the initial observation
        observations = []
        for agent_i in range(self.num_agents):
            agent_i_observation = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], server_capacity[agent_i], 0]
            observations.append(agent_i_observation)
        return observations

    def step(self, actions):
        # Execute the multi-agents' actions
        # Return the new observations, rewards, dones
        observations = []
        rewards = []
        dones = []
        for agent_i in range(self.num_agents):
            pass
            
        
        pass

