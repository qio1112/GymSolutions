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
        self.lr = config.get("lr", 0.001)
        self.init_network(state_size, action_size, QNetwork, device)

    def init_network(self, state_size, action_size, QNetwork, device):
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

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

    def update_epsilon(self, epsilon, reward_history):
        # reduce the e-greedy epsilon by num of episodes
        epsilon = max(self.config['epsilon_end'], epsilon * self.config['epsilon_decay'])
        # increase epsilon to larger number if most of recent rewards are negative after epsilon reaches minimum value
        burst_epsilon = self.config.get('burst_epsilon', None)
        if burst_epsilon and epsilon == self.config['epsilon_end'] and len(reward_history) > 100:
            last_100_returns = reward_history[-100:]
            negative_return_count = 0
            for i in range(100):
                if last_100_returns[i] < 0:
                    negative_return_count += 1
            if negative_return_count > 70:
                epsilon = burst_epsilon
                print(f"Increase epsilon to {epsilon}")
        return epsilon

    # default q_agent sinply concatenate the stack of states together into one array
    def concat_step_stack(self, step_stack):
        return np.concatenate(step_stack)

    def train(self):
        epsilon = self.config['epsilon_start']
        batch_size = self.config.get('batch_size', 1)
        buffer_size = self.config.get('buffer_size', 3000)
        reset_options = self.config.get('reset_options', None)
        save_model_episode = self.config.get('save_model_episode', None)
        # skip recording certain steps in memory
        skip_step = self.config.get('skip_step', None)
        # if an episode has too many negative rewards, stop the episode
        stop_by_negative_count = self.config.get('stop_by_negative_count', None)
        step_stack_size = self.config.get('step_stack', 1)

        batch_count = 0
        reward_history = []
        print(f"buffer_size={buffer_size}, batch_size={batch_size}")
        buffer = deque(maxlen=buffer_size)
        steps = 0

        for episode in range(self.config['episodes']):
            state = self.env.reset(options=reset_options)[0]
            step_deque = deque([state]*step_stack_size, maxlen=step_stack_size)
            state_stack = self.concat_step_stack(step_deque)
            total_reward = 0
            negative_reward_in_eps = 0
            step_count_eps = 0

            for step in range(self.config['max_steps']):
                steps += 1
                step_count_eps += 1
                batch_count += 1
                action = self.get_action(state_stack, epsilon)
                next_state, reward, done, *_ = self.env.step(action)
                step_deque.append(next_state)
                next_state_stack = self.concat_step_stack(step_deque)

                buffer.append((state_stack, action, reward, next_state_stack, done))

                # experience replay method with TD
                if len(buffer) >= batch_size:
                    selected_exp = random.sample(buffer, k=batch_size)
                    self.train_batch([t[0] for t in selected_exp],
                                     [t[1] for t in selected_exp],
                                     [t[2] for t in selected_exp],
                                     [t[3] for t in selected_exp],
                                     [t[4] for t in selected_exp],
                                     self.config['gamma'])

                state_stack = next_state_stack
                total_reward += reward

                # update target network
                if self.config['update_target_steps'] and steps % self.config['update_target_steps'] == 0:
                    self.update_target_network()

                if reward < 0:
                    negative_reward_in_eps += 1

                if stop_by_negative_count and negative_reward_in_eps > stop_by_negative_count:
                    break

                if skip_step:
                    for i in range(skip_step):
                        next_state, reward, done, *_ = self.env.step(action)
                        total_reward += reward
                        steps += 1
                        if done:
                            break

                if done:
                    break

            reward_history.append(total_reward)

            last_50_history = reward_history[-50:]
            ma50 = sum(last_50_history) / len(last_50_history)
            if len(reward_history) % 50 == 0 or len(reward_history) >= self.config['episodes'] - 10:
                draw_reward_history(reward_history, 50)

            print("Episode: {}, Total Reward: {:.2f}, MA50: {:.2f}, epsilon: {:.4f}, #step: {}".format(episode + 1, total_reward,
                                                                                            ma50, epsilon, step_count_eps))

            # update epsilon
            epsilon = self.update_epsilon(epsilon, reward_history)

            # terminate training if ma50 is larger than terminate_at_reward_ma50
            if len(reward_history) > 50 and \
                    self.config['terminate_at_reward_ma50'] and \
                    ma50 > self.config['terminate_at_reward_ma50']:
                draw_reward_history(reward_history)
                print(f"Terminating training with ma50={ma50}")
                break

            if save_model_episode and (episode + 1) > save_model_episode and (episode + 1) % save_model_episode == 0:
                draw_reward_history(reward_history)
                save_model(self.target_network, self.config, {'reward': reward_history},
                           additional_path='epi=' + str(episode + 1))
        return reward_history
