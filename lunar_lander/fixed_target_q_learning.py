import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.utils import save_model, draw_reward_history
import os
from collections import deque
import random

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
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
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(0)
        done = torch.tensor(done, dtype=torch.float).unsqueeze(0)

        q_values = self.q_network(state)[0]
        next_q_values = self.target_network(next_state)  # fixed target
        next_q_value = torch.max(next_q_values)
        target_q = reward + (1 - done) * gamma * next_q_value

        q_value = q_values[action]
        loss = nn.MSELoss()(q_value, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Define the training loop
def train_agent(env, agent, config):
    epsilon = config['epsilon_start']
    reward_history = []
    buffer = deque(maxlen=3000)
    steps = 0
    for i, episode in enumerate(range(config['episodes'])):
        state = env.reset()[0]
        total_reward = 0

        for step in range(config['max_steps']):
            steps += 1
            action = agent.get_action(state, epsilon)
            next_state, reward, done, *_ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            # TD method, train for each step
            agent.train(state, action, reward, next_state, done, config['gamma'])
            state = next_state
            total_reward += reward

            # update target network
            if config['update_target_steps'] and steps % config['update_target_steps'] == 0:
                agent.update_target_network()

            if done:
                break

        reward_history.append(total_reward)

        ma50 = sum(reward_history[-50:]) / 50
        if len(reward_history) % 10 == 0 or len(reward_history) >= config['episodes']-10:
            draw_reward_history(reward_history)
        # reduce the e-greedy epsilon by num of episodes
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])
        print("Episode: {}, Total Reward: {}, MA50: {}".format(episode + 1, total_reward, ma50))

        # terminate training if ma50 is larger than terminate_at_reward_ma50
        if len(reward_history) > 50 and \
                config['terminate_at_reward_ma50'] and \
                ma50 > config['terminate_at_reward_ma50']:
            draw_reward_history(reward_history)
            print(f"Terminating training with ma50={ma50}")
            break

        # experience replay for every 25 episodes
        if config['experience_replay'] and i > 50 and i % 25 == 0:
            selected_exp = random.sample(buffer, k=300)
            for exp in selected_exp:
                agent.train(exp[0], exp[1], exp[2], exp[3], exp[4], config['gamma'])
            agent.update_target_network()
            print(f"Experience Replayed")
    return reward_history


if __name__ == "__main__":
    # Create the LunarLander environment
    env = gym.make('LunarLander-v2')
    # env = gym.make('LunarLander-v2', render_mode="human")

    # Set hyperparameters
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    config = {'episodes': 3000,
              'max_steps': 1000,
              'epsilon_start': 1,
              'epsilon_end': 0.01,
              'epsilon_decay': 0.996,
              'gamma': 0.99,
              'update_target_steps': 1,
              'terminate_at_reward_ma50': 220,
              'experience_replay': True}

    load_model_path = ''

    # Create the agent
    agent = QAgent(state_size, action_size)
    agent.load_model(load_model_path)

    # Train the agent
    reward_history = train_agent(env, agent, config)

    will_save = True
    if save_model:
        save_model(os.getcwd(), agent.q_network, 'fixed_target_replay_learning', config, reward_history)
