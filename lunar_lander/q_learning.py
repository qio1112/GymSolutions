import random
from collections import deque

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.utils import save_model, draw_reward_history
from rl_agent.QAgent import QAgent
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
              'epsilon_decay': 0.996,
              'gamma': 0.99,
              'update_target_steps': 1,
              'terminate_at_ma': 50,
              'terminate_at_ma_reward': 230,
              'buffer_size': 30000,
              'experience_replay': True,
              'experience_replay_size': 300,
              'batch_size': 32,
              'save_model_episode': 20}

    load_model_path = ''

    # Create the agent
    agent = QAgent(config, env, state_size, action_size, QNetwork)
    agent.load_model(load_model_path)

    # Train the agent
    reward_history = agent.train()

    will_save = True
    if will_save:
        save_model(agent.q_network, config, {'reward': reward_history})
