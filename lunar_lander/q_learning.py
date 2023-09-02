import random
from collections import deque

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.utils import save_model, draw_reward_history
from rl_agent.q_agent import QAgent
import os
import datetime


# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


if __name__ == "__main__":
    # Create the LunarLander environment
    env = gym.make('LunarLander-v2')
    # env = gym.make('LunarLander-v2', render_mode="human")

    # Set hyperparameters
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    base_path = os.getcwd()
    ts = str(datetime.datetime.now().timestamp()).split('.')[0]
    name = ts + "_fixed_target_replay_learning_batched"
    path = os.path.join(base_path, name)

    config = {
              'path': path,
              'episodes': 3000,
              'max_steps': 2000,
              'epsilon_start': 1,
              'epsilon_end': 0.01,
              'epsilon_decay': 0.99,
              'gamma': 0.99,
              'update_target_steps': 500,
              'terminate_at_reward_ma50': 200,
              'buffer_size': 1000,
              'batch_size': 32,
              'burst_epsilon': 0.1,
              'save_model_episode': None
    }

    load_model_path = ''

    # Create the agent
    agent = QAgent(config, env, state_size, action_size, QNetwork, None)
    agent.load_model(load_model_path)

    # Train the agent
    reward_history = agent.train()

    will_save = True
    if will_save:
        save_model(agent.q_network, config, {'reward': reward_history})
