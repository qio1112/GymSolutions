import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.utils import save_model, draw_reward_history
import os


# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

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
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

    def load_model(self, weights_path):
        if weights_path:
            self.q_network.load_state_dict(torch.load(weights_path))
            print(f"Load model from {weights_path}")

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
        next_q_values = self.q_network(next_state)
        next_q_value = torch.max(next_q_values)
        target_q = reward + (1 - done) * gamma * next_q_value

        # q_value = torch.gather(q_values, 1, action.unsqueeze(1))
        q_value = q_values[action]
        loss = nn.MSELoss()(q_value, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Define the training loop
def train_agent(env, agent, config):
    epsilon = config['epsilon_start']
    reward_history = []
    for episode in range(config['episodes']):
        state = env.reset()[0]
        total_reward = 0

        for step in range(config['max_steps']):
            action = agent.get_action(state, epsilon)
            next_state, reward, done, *_ = env.step(action)
            # TD method, train for each step
            agent.train(state, action, reward, next_state, done, config['gamma'])
            state = next_state
            total_reward += reward

            if done:
                break

        reward_history.append(total_reward)
        draw_reward_history(reward_history)
        # reduce the e-greedy epsilon by num of episodes
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])
        print("Episode: {}, Total Reward: {}".format(episode + 1, total_reward))
    return reward_history

if __name__ == "__main__":
    # Create the LunarLander environment
    env = gym.make('LunarLander-v2', render_mode="human")

    # Set random seed for reproducibility
    seed = 42
    # env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set hyperparameters
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    config = {'episodes': 700,
              'max_steps': 1000,
              'epsilon_start': 1.0,
              'epsilon_end': 0.01,
              'epsilon_decay': 0.995,
              'gamma': 0.99,
              'update_target_steps': 1}

    load_model_path = ''

    # Create the agent
    agent = QAgent(state_size, action_size)
    agent.load_model(load_model_path)

    # Train the agent
    reward_history = train_agent(env, agent, config)

    will_save = True
    if will_save:
        save_model(os.getcwd(), agent.q_network, 'dqn', config, {'reward': reward_history})
