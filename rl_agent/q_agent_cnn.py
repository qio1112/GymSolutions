import torch
import numpy as np
from rl_agent.q_agent import QAgent


def transform_state(state, grey=True):
    # input state (batch_size, stack_size, x, y, 3)
    # to shape (batch_size, new_channel_size, x, y)
    if not grey:
        # combine stack_size and channels into one dimension
        original_shape = state.shape
        new_shape = (original_shape[0], original_shape[1] * original_shape[4], original_shape[2], original_shape[3])
        return state.permute(0, 1, 4, 2, 3).reshape(new_shape) / 255
    else:
        state = state.permute(0, 1, 4, 2, 3)  # convert to (batch, channel, x, y)
        return torch.mean(state, dim=2, keepdim=False) / 255


class QAgentCNN(QAgent):

    def __init__(self, config, env, input_channel, action_size, QNetwork, device):
        self.lr = config.get("lr", 0.001)
        self.grey = config.get("grey", True)
        super().__init__(config, env, input_channel, action_size, QNetwork, device)

    def init_network(self, input_channel, action_size, QNetwork, device):
        self.q_network = QNetwork(input_channel, action_size).to(device)
        self.target_network = QNetwork(input_channel, action_size).to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

    def get_action(self, input_image, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_size)
        else:
            state = transform_state(torch.tensor(input_image, dtype=torch.float).unsqueeze(0)).to(self.device)
            q_values = self.q_network(state)
            return torch.argmax(q_values, dim=1).item()

    def concat_step_stack(self, state_stack):
        # cnn agent keeps the channels, returns (stack_size, x, y, channel)
        return np.array(state_stack)

    def train_batch(self, state, action, reward, next_state, done, gamma):
        # input state.shape is (batch_size, stack_size, x, y, channel)
        state = transform_state(torch.tensor(np.array(state), dtype=torch.float)).to(self.device)
        next_state = transform_state(torch.tensor(np.array(next_state), dtype=torch.float)).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.int).to(self.device)

        q_values = self.q_network(state)
        next_q_values = self.target_network(next_state)

        q_value = q_values[torch.arange(q_values.shape[0]), action]
        next_q_value = torch.max(next_q_values, dim=1).values

        target_q = reward + (1 - done) * gamma * next_q_value
        loss = torch.nn.MSELoss()(q_value, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


