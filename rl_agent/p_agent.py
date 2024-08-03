import gymnasium as gym
import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np

import os
import logging

from torch.distributions import Categorical

from utils.utils import draw_reward_history, save_model


class PAgent:

    def __init__(self, config, env, policy_network, baseline_network=None, device="cpu"):
        torch.autograd.set_detect_anomaly(True)
        self.config = config
        self.train_mode = self.config.get("train", True)
        self.device_name = device
        self.device = torch.device(device)
        self.env = env
        self.base_path = self.config.get("path")
        if self.base_path:
            os.makedirs(self.base_path, exist_ok=True)
        self.logger = get_logger(config.get("log_path"))

        self.discrete = self.config.get("discrete", False)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = (
            self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        )

        self.gamma = self.config.get("gamma", 0.99)
        self.lr = self.config.get("learning_rate", 0.001)
        self.use_baseline = self.config.get("use_baseline", True)
        self.normalize_advantage = self.config.get("normalize_advantage", True)
        self.batch_size = self.config.get("batch_size", 1)
        self.max_ep_len = self.config.get("max_ep_len", 2000)
        self.num_batches = self.config.get("num_batches", 300)
        self.use_ppo = self.config.get("use_ppo", False)
        self.ppo_clip_eps = self.config.get("ppo_clip_eps", 0.2)

        if self.use_baseline:
            self.baseline_network = baseline_network.to(self.device)
            self.baseline_optim = torch.optim.Adam(self.baseline_network.parameters(), lr=self.lr)
        self.policy_network = policy_network.to(self.device)
        self.policy_optim = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr, weight_decay=1e-4)

    def load_model(self, weights_path, baseline_weights_path=None):
        if weights_path:
            self.policy_network.load_state_dict(torch.load(weights_path))
            self.policy_network.to(self.device)
            print(f"Load policy model from {weights_path}")
        if baseline_weights_path:
            self.baseline_network.load_state_dict(torch.load(weights_path))
            self.baseline_network.to(self.device)
            print(f"Load baseline model from {weights_path}")

    def get_distribution(self, observations):
        if self.discrete:
            return ptd.Categorical(torch.exp(self.policy_network(observations)))  # softmax
        else:  # continuous
            mean, std = self.policy_network(observations)
            # if self.device_name == "mps":
                # mac mps doesn't support MultivariateNormal
            if mean.dim() == 1:
                dists = [ptd.Normal(mean[i], std[i]) for i in range(self.action_dim)]
                return dists
            else:  # batched
                dists = [ptd.Normal(mean[:, i], std[:, i]) for i in range(self.action_dim)]
                return dists
            # else:
            #     covariance_matrix = torch.diag(std ** 2)
            #     return ptd.MultivariateNormal(mean, covariance_matrix)  # Gaussian

    def get_returns(self, paths):
        all_returns = []
        for path in paths:
            rewards = path["reward"]
            returns = np.copy(rewards)
            for t in reversed(range(rewards.shape[0] - 1)):
                returns[t] = rewards[t] + self.gamma * returns[t + 1]
            all_returns.append(returns)
        returns = np.concatenate(all_returns).astype(np.float32)
        return torch.tensor(returns, dtype=torch.float).to(self.device)

    def get_advantages(self, returns, observations):
        if self.use_baseline:
            baselines = self.baseline_network(observations).squeeze()
            advantages = returns - baselines
        else:
            advantages = returns
        if self.normalize_advantage:
            advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)
        return advantages

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        distribution = self.get_distribution(state)

        if isinstance(distribution, list):  # multiple Normal distributions
            sampled_action_list = [dist.sample() for dist in distribution]
            sampled_action = torch.stack(sampled_action_list, dim=-1)  # this action is before applying limits such as tanh
            if not self.discrete:
                sampled_action = self.policy_network.apply_action_limit(sampled_action)
            log_prob = self.get_log_prob(distribution, sampled_action)
        else:
            sampled_action = distribution.sample()
            # this is the log_prob for the selected actions based on the current policy
            log_prob = distribution.log_prob(sampled_action)
        return sampled_action.squeeze(0), log_prob

    def sample_paths(self):
        episode = 0
        episode_rewards = []
        paths = []
        t = 0
        last_ep_finished = False  # stop when last episode finishes, use to prevent having half an episode

        while t < self.batch_size or (t >= self.batch_size and not last_ep_finished):
            state = self.env.reset()[0]
            states, actions, rewards, log_probs = [], [], [], []
            episode_reward = 0

            for step in range(self.max_ep_len):
                states.append(state)
                action, log_prob = self.act(state)
                actual_action = action
                # if not self.discrete:
                #     limited_action = self.policy_network.apply_action_limit(action)
                #     actual_action = limited_action
                state, reward, done, _, info = self.env.step(actual_action.detach().cpu().numpy())
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                episode_reward = episode_reward + reward
                t = t + 1
                if done or step == self.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    last_ep_finished = t >= (self.batch_size - 1)
                    break

            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": torch.stack(actions),
                "log_prob": torch.stack(log_probs)
            }
            paths.append(path)
            episode += 1

        return paths, episode_rewards

    def get_log_prob(self, distribution, actions):
        log_probs = []
        if isinstance(distribution, list):
            for i, dist in enumerate(distribution):
                # Calculate the log probability for the i-th action component
                if actions.dim() == 1:
                    log_prob = dist.log_prob(actions[i])
                else:  # batched
                    log_prob = dist.log_prob(actions[:, i])
                log_probs.append(log_prob)
            total_log_prob = torch.sum(torch.stack(log_probs), dim=0)
            return total_log_prob
        else:
            return distribution.log_prob(actions)

    def train_baseline(self, returns, observations):
        baselines = self.baseline_network(observations).squeeze()
        loss = torch.nn.MSELoss()(returns, baselines)
        self.baseline_optim.zero_grad()
        loss.backward()
        self.baseline_optim.step()

    def train_policy(self, observations, actions, advantages, old_log_probs):

        action_distribution = self.get_distribution(observations)
        log_probs = self.get_log_prob(action_distribution, actions)  # log_probs here are negative

        if self.use_ppo:
            # ratio = pi(a_t|s_t)/pi_old(a_t|s_t) = exp(log(pi(a_t|s_t) - log(pi_old(a_t|s_t)))
            ratios = torch.exp(log_probs - old_log_probs)
            value1 = ratios * advantages
            value2 = torch.clamp(ratios, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps) * advantages
            loss = -torch.min(value1, value2).mean()
        else:
            loss = -torch.mean(advantages * log_probs)

        self.policy_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
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
            observations = torch.tensor(np.concatenate([path["observation"] for path in paths]), dtype=torch.float).to(
                self.device)  # (batch_size, *obs_space)

            actions = torch.cat([path["action"] for path in paths])
            old_log_probs = torch.cat([path["log_prob"] for path in paths])

            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)  # G_t (batch_size, 1)

            # advantage will depend on the baseline implementation
            advantages = self.get_advantages(returns, observations)  # (batch_size,)

            # run training operations
            if self.train_mode:
                self.train_policy(observations, actions, advantages, old_log_probs)
                if self.use_baseline:
                    self.train_baseline(returns, observations)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}, #epi={}, batch_size={}".format(
                t, avg_reward, sigma_reward, len(paths), len(observations)
            )
            averaged_total_rewards.append(avg_reward)
            self.logger.info(msg)

            early_stop_reward = self.config.get("early_stop_reward", 0)
            early_stop_ma = self.config.get("early_stop_ma", 50)
            if early_stop_reward > 0:
                last_history = all_total_rewards[-early_stop_ma:]
                ma = sum(last_history) / len(last_history)
                if len(all_total_rewards) > early_stop_ma and ma > early_stop_reward:
                    print(f"Terminating training with ma{early_stop_ma}={early_stop_reward}")
                    break

        self.logger.info("- Training done.")
        draw_reward_history(all_total_rewards, window_size=early_stop_ma)
        save_model(self.policy_network, self.config, {"rewards": all_total_rewards})


def get_logger(filename):
    """
    Return a logger instance to a file
    """
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return logger
