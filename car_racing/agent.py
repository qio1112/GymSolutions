import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class QAgent:

    def __init__(self, action_size, network, device):
        self.action_size = action_size
        self.q_network = network(action_size).to(device)
        self.target_network = network(action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.device = device

    def load_model(self, weights_path):
        if weights_path:
            self.q_network.load_state_dict(torch.load(weights_path))
            self.target_network.load_state_dict(torch.load(weights_path))
            print(f"Load model from {weights_path}")

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_action(self, input_image, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.tensor(input_image, dtype=torch.float).to(self.device).unsqueeze(0).permute(0, 3, 1, 2)
            q_values = self.q_network(state)
            return torch.argmax(q_values, dim=1).item()

    def train(self, state, action, reward, next_state, done, gamma):
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device).permute(0, 3, 1, 2)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device).permute(0, 3, 1, 2)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.int).to(self.device)

        q_values = self.q_network(state)
        next_q_values = self.target_network(next_state)

        q_value = q_values[torch.arange(q_values.shape[0]), action]
        next_q_value = torch.max(next_q_values, dim=1).values

        target_q = reward + (1 - done) * gamma * next_q_value
        loss = nn.MSELoss()(q_value, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
