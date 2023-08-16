import random
from collections import deque

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.utils import save_model, draw_reward_history
from utils.rl_trainer import train_agent
import os


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


# Define the agent
class QAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

    def load_model(self, weights_path):
        if weights_path:
            self.q_network.load_state_dict(torch.load(weights_path))
            self.target_network.load_state_dict(torch.load(weights_path))
            print(f"Load model from {weights_path}")

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            q_values = self.q_network(state)
            return torch.argmax(q_values, dim=1).item()

    def train(self, state, action, reward, next_state, done, gamma):
        state = torch.tensor(np.array(state), dtype=torch.float)  # (batch_size, num_states)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float) # (batch_size, num_states)
        action = torch.tensor(np.array(action), dtype=torch.long)  # (batch_size, )
        reward = torch.tensor(np.array(reward), dtype=torch.float)  # (batch_size, )
        done = torch.tensor(np.array(done), dtype=torch.float)  # (batch_size, )

        q_values = self.q_network(state)  # (batch_size, num_actions)
        next_q_values = self.target_network(next_state)  # fixed target (batch_size, num_actions)
        next_q_value = torch.max(next_q_values, dim=1).values  # (batch_size, )
        target_q = reward + (1 - done) * gamma * next_q_value

        q_value = q_values[torch.arange(q_values.shape[0]), action] # (batch_size, )
        loss = nn.MSELoss()(q_value, target_q.detach()) # (batch_size, )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # Create the LunarLander environment
    env = gym.make('LunarLander-v2')
    # env = gym.make('LunarLander-v2', render_mode="human")

    # Set hyperparameters
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    config = {'episodes': 1,
              'max_steps': 2000,
              'epsilon_start': 1,
              'epsilon_end': 0.01,
              'epsilon_decay': 0.996,
              'gamma': 0.99,
              'update_target_steps': 1,
              'terminate_at_reward_ma50': 230,
              'buffer_size': 30000,
              'experience_replay': True,
              'experience_replay_size': 300,
              'batch_size': 32}

    load_model_path = ''

    # Create the agent
    agent = QAgent(state_size, action_size)
    agent.load_model(load_model_path)

    # Train the agent
    reward_history = train_agent(env, agent, config)

    will_save = True
    if will_save:
        save_model(os.getcwd(), agent.q_network, 'fixed_target_replay_learning_batched', config, {'reward': reward_history})
