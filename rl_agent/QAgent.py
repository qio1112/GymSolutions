import torch
import numpy as np
from collections import deque
import random

from utils.utils import draw_reward_history, save_model


class QAgent:
    def __init__(self, config, env, state_size, action_size, QNetwork, device):
        self.device = device
        self.config = config
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.init_network(state_size, action_size, QNetwork, device)

    def init_network(self, state_size, action_size, QNetwork, device):
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

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

    def train_batch(self, state, action, reward, next_state, done, gamma):
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
        loss = torch.nn.MSELoss()(q_value, target_q.detach()) # (batch_size, )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def train(self):
        epsilon = self.config['epsilon_start']
        batch_size = self.config.get('batch_size', 1)
        buffer_size = self.config.get('buffer_size', 30000)
        reset_options = self.config.get('reset_options', None)
        save_model_episode = self.config.get('save_model_episode', None)

        batch_count = 0
        reward_history = []
        buffer = deque(maxlen=buffer_size)
        steps = 0

        for episode in range(self.config['episodes']):
            state = self.env.reset(options=reset_options)[0]
            total_reward = 0

            for step in range(self.config['max_steps']):
                steps += 1
                batch_count += 1
                action = self.get_action(state, epsilon)
                next_state, reward, done, *_ = self.env.step(action)
                buffer.append((state, action, reward, next_state, done))

                # TD method, train for each step
                if batch_count == batch_size:
                    batch_count = 0
                    batch_data = list(buffer)[-50:]
                    self.train_batch([t[0] for t in batch_data],
                                     [t[1] for t in batch_data],
                                     [t[2] for t in batch_data],
                                     [t[3] for t in batch_data],
                                     [t[4] for t in batch_data],
                                     self.config['gamma'])

                state = next_state
                total_reward += reward

                # update target network
                if self.config['update_target_steps'] and steps % self.config['update_target_steps'] == 0:
                    self.update_target_network()

                if done:
                    break

            reward_history.append(total_reward)

            last_50_history = reward_history[-50:]
            ma50 = sum(last_50_history) / len(last_50_history)
            if len(reward_history) % 10 == 0 or len(reward_history) >= self.config['episodes'] - 10:
                draw_reward_history(reward_history)

            # reduce the e-greedy epsilon by num of episodes
            epsilon = max(self.config['epsilon_end'], epsilon * self.config['epsilon_decay'])
            print("Episode: {}, Total Reward: {:.2f}, MA50: {:.2f}, epsilon: {:.4f}".format(episode + 1, total_reward,
                                                                                            ma50, epsilon))

            # terminate training if ma50 is larger than terminate_at_reward_ma50
            if len(reward_history) > 50 and \
                    self.config['terminate_at_reward_ma50'] and \
                    ma50 > self.config['terminate_at_reward_ma50']:
                draw_reward_history(reward_history)
                print(f"Terminating training with ma50={ma50}")
                break

            if save_model_episode and (episode + 1) > save_model_episode and (episode + 1) % save_model_episode == 0:
                save_model(self.target_network, self.config, {'reward': reward_history},
                           additional_path='epi=' + str(episode + 1))

            # experience replay for every 25 episodes
            replay_after_episode = self.config.get('replay_after_episode', 50)
            replay_every_episode = self.config.get('replay_every_episode', 25)
            replay_size = self.config['experience_replay_size']
            if len(buffer) >= replay_size and \
                    self.config['experience_replay'] and episode > replay_after_episode and \
                    episode % replay_every_episode == 0:
                selected_exp = random.sample(buffer, k=replay_size)
                self.train_batch([t[0] for t in selected_exp],
                                 [t[1] for t in selected_exp],
                                 [t[2] for t in selected_exp],
                                 [t[3] for t in selected_exp],
                                 [t[4] for t in selected_exp],
                                 self.config['gamma'])
                self.update_target_network()
                print(f"Experience Replayed")
        return reward_history


class QAgentCNN(QAgent):

    def __init__(self, config, action_size, device, env, QNetwork):
        super().__init__(config, env, None, action_size, QNetwork, device)

    def init_network(self, state_size, action_size, QNetwork, device):
        self.q_network = QNetwork(action_size).to(device)
        self.target_network = QNetwork(action_size).to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    def get_action(self, input_image, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.tensor(input_image, dtype=torch.float).to(self.device).unsqueeze(0).permute(0, 3, 1, 2)
            q_values = self.q_network(state)
            return torch.argmax(q_values, dim=1).item()

    def train_batch(self, state, action, reward, next_state, done, gamma):
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
        loss = torch.nn.MSELoss()(q_value, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
