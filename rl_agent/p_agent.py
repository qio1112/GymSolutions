import gymnasium as gym
import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np

import os
import logging

from utils.utils import draw_reward_history, save_model


class PAgent:

    def __init__(self, config, env, policy_network, baseline_network=None, device="cpu"):
        self.config = config
        self.train_mode = self.config.get("train", True)
        self.device = torch.device(device)
        self.env = env
        self.base_path = self.config.get("path")
        if self.base_path:
            os.makedirs(self.base_path, exist_ok=True)
        self.logger = get_logger(config.get("log_path"))

        self.discrete = isinstance(self.env.action_space, gym.spaces.discrete.Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = (
            self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        )
        if self.discrete:
            self.log_std = nn.Parameter(torch.zeros(self.action_dim).to(self.device))

        self.gamma = self.config.get("gamma", 0.99)
        self.lr = self.config.get("learning_rate", 0.001)
        self.use_baseline = self.config.get("use_baseline", True)
        self.normalize_advantage = self.config.get("normalize_advantage", True)
        self.batch_size = self.config.get("batch_size", 1)
        self.max_ep_len = self.config.get("max_ep_len", 2000)
        self.num_batches = self.config.get("num_batches", 300)
        if self.use_baseline:
            self.baseline_network = baseline_network
            self.baseline_optim = torch.optim.Adam(self.baseline_network.parameters(), lr=self.lr)
        self.policy_network = policy_network
        self.policy_optim = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)

    def load_model(self, weights_path, baseline_weights_path=None):
        if weights_path:
            self.policy_network.load_state_dict(torch.load(weights_path))
            print(f"Load policy model from {weights_path}")
        if baseline_weights_path:
            self.baseline_network.load_state_dict(torch.load(weights_path))
            print(f"Load baseline model from {weights_path}")

    def get_distribution(self, observations):
        if self.discrete:
            return ptd.Categorical(torch.exp(self.policy_network(observations)))
        else:  # continuous
            std = torch.exp(self.log_std)
            mean = self.policy_network(observations)
            covariance_matrix = torch.diag(std ** 2)
            return ptd.MultivariateNormal(mean, covariance_matrix)

    def get_returns(self, paths):
        all_returns = []
        for path in paths:
            rewards = path["reward"]
            returns = np.copy(rewards)
            for t in reversed(range(rewards.shape[0]-1)):
                returns[t] = rewards[t] + self.gamma * returns[t+1]
            all_returns.append(returns)
        returns = np.concatenate(all_returns)
        return np2torch(returns, device=self.device)

    def get_advantages(self, returns, observations):
        if self.use_baseline:
            base_lines = self.baseline_network(observations).squeeze()
            advantages = returns - base_lines
        else:
            advantages = returns
        if self.normalize_advantage:
            advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)
        return advantages

    def act(self, observations):
        observations = np2torch(observations, device=self.device)
        distribution = self.get_distribution(observations)
        sampled_actions = distribution.sample((observations.shape[0],))
        # log_probs = distribution.log_prob(sampled_actions)
        return sampled_actions

    def sample_paths(self):
        episode = 0
        episode_rewards = []
        paths = []
        t = 0
        last_ep_finished = False  # stop when last episode finishes, use to prevent having half an episode

        while t < self.batch_size or (t >= self.batch_size and not last_ep_finished):
            state = self.env.reset()[0]
            states, actions, rewards = [], [], []
            episode_reward = 0

            for step in range(self.max_ep_len):
                states.append(state)
                action = self.act(states[-1])[0]
                state, reward, done, _, info = self.env.step(action.detach().cpu().numpy())
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if done or step == self.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    last_ep_finished = t >= (self.batch_size - 1)
                    break

            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
            }
            paths.append(path)
            episode += 1

        return paths, episode_rewards

    def train_baseline(self, returns, observations):
        loss = torch.nn.MSELoss()(returns, self.baseline_network(observations).squeeze())
        self.baseline_optim.zero_grad()
        loss.backward()

    def train_policy(self, observations, actions, advantages):
        action_distribution = self.get_distribution(observations)
        log_probs = action_distribution.log_prob(actions)  # log_probs here are negative
        loss = -torch.sum(advantages * log_probs)
        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()

    def train(self):
        self.logger.info(f"start training. device: {self.device}")
        all_total_rewards = (
            []
        )  # the returns of all episodes samples for training purposes
        averaged_total_rewards = []  # the returns for each iteration

        for t in range(self.num_batches):

            # collect a minibatch of samples
            paths, total_rewards = self.sample_paths()  # (#episodes, ()), (#episodes,)
            all_total_rewards.extend(total_rewards)
            observations = np2torch(np.concatenate([path["observation"] for path in paths]), device=self.device)  # (batch_size, *obs_space)
            actions = np2torch(np.concatenate([path["action"] for path in paths]), device=self.device)  # (batch_size,)
            rewards = np2torch(np.concatenate([path["reward"] for path in paths]), device=self.device)  # (batch_size,)
            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)  # G_t (batch_size, 1)

            # advantage will depend on the baseline implementation
            advantages = self.get_advantages(returns, observations)  # (batch_size,)

            # run training operations
            if self.train_mode:
                if self.use_baseline:
                    self.train_baseline(returns, observations)
                self.train_policy(observations, actions, advantages)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}, #epi={}, batch_size={}".format(
                t, avg_reward, sigma_reward, len(paths), len(observations)
            )
            averaged_total_rewards.append(avg_reward)
            self.logger.info(msg)

            terminate_at_ma = self.config.get("terminate_at_ma")
            if terminate_at_ma:
                terminate_at_ma_reward = self.config.get("terminate_at_reward_ma_reward", 0)
                last_history = all_total_rewards[-terminate_at_ma:]
                ma = sum(last_history) / len(last_history)
                if len(all_total_rewards) > terminate_at_ma and ma > terminate_at_ma_reward:
                    print(f"Terminating training with ma{terminate_at_ma}={ma}")
                    break

        self.logger.info("- Training done.")
        draw_reward_history(all_total_rewards)
        save_model(self.policy_network, self.config, {"rewards": all_total_rewards})


def np2torch(x, cast_double_to_float=True, device=torch.device("cpu")):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    if isinstance(x, torch.Tensor):
        return x
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x


def get_logger(filename):
    """
    Return a logger instance to a file
    """
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return logger