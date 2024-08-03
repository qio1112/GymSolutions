import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
learning_rate = 1e-3
gamma = 0.99
num_episodes = 2000

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3_mu = nn.Linear(64, 2)
        self.fc3_log_sigma = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = torch.tanh(self.fc3_mu(x))
        log_sigma = self.fc3_log_sigma(x)
        sigma = torch.exp(log_sigma)
        return mu, sigma

# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value

def train():
    env = gym.make('LunarLanderContinuous-v2')
    policy_net = PolicyNetwork()
    value_net = ValueNetwork()
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=learning_rate)
    optimizer_value = optim.Adam(value_net.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    for episode in range(num_episodes):
        state = env.reset()[0]
        log_probs = []
        values = []
        rewards = []

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mu, sigma = policy_net(state_tensor)
            dist = torch.distributions.Normal(mu, sigma)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            value = value_net(state_tensor)

            next_state, reward, done, _, _ = env.step(action.numpy()[0])

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

            if done:
                break

        # Compute returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        values = torch.cat(values).squeeze()
        log_probs = torch.stack(log_probs)

        # Compute advantage
        advantage = returns - values.detach()

        # Policy loss
        policy_loss = -(log_probs * advantage).mean()

        # Value loss
        value_loss = mse_loss(values, returns)

        # Optimize policy network
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        # Optimize value network
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {sum(rewards)}")

if __name__ == '__main__':
    train()
