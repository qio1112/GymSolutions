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
        # torch.autograd.set_detect_anomaly(True)
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
        self.normalize_advantage = self.config.get("normalize_advantage", True)
        self.batch_size = self.config.get("batch_size", 1)
        self.max_ep_len = self.config.get("max_ep_len", 2000)
        self.num_batches = self.config.get("num_batches", 300)
        self.use_ppo = self.config.get("ppo", False)
        self.ppo_clip_eps = self.config.get("ppo_clip_eps", 0.2)
        self.ppo_epochs = self.config.get("ppo_epochs", 4)
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

    def sample_episode(self):
        state = self.env.reset()[0]
        states, actions, rewards, log_probs, baselines = [], [], [], [], []
        episode_reward = 0  # this is used for loggind and show the progress

        while True:
            states.append(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if not self.discrete:
                mean, std = self.policy_network(state_tensor)
                dist = ptd.Normal(mean, std)
            else:
                dist = ptd.Categorical(torch.exp(self.policy_network(state_tensor)))  # softmax
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()  # sum for all action dimensions
            baseline = self.baseline_network(state_tensor)

            next_state, reward, done, _, _ = self.env.step(action.cpu().numpy()[0])

            actions.append(action)
            log_probs.append(log_prob)
            baselines.append(baseline)
            rewards.append(reward)

            state = next_state
            episode_reward = episode_reward + reward

            if done or len(rewards) >= self.max_ep_len:
                break

        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        actions = torch.cat(actions).to(self.device)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        baselines = torch.cat(baselines).squeeze().to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        advantages = returns - baselines
        if self.normalize_advantage:
            advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)
        return actions, states, returns, baselines, log_probs, advantages, episode_reward

    def sample_paths(self):
        num_episodes = 0
        num_steps = 0
        actions_all, states_all, returns_all, baselines_all, log_probs_all, advantages_all, episode_rewards_all = [], [], [], [], [], [], []

        while num_steps < self.batch_size:
            actions, states, returns, baselines, log_probs, advantages, episode_reward = self.sample_episode()
            actions_all.append(actions)
            states_all.append(states)
            returns_all.append(returns)
            baselines_all.append(baselines)
            log_probs_all.append(log_probs)
            advantages_all.append(advantages)
            episode_rewards_all.append(episode_reward)
            num_steps += states.shape[0]
            num_episodes += 1

        actions_all = torch.cat(actions_all)
        states_all = torch.cat(states_all)
        returns_all = torch.cat(returns_all)
        baselines_all = torch.cat(baselines_all)
        log_probs_all = torch.cat(log_probs_all)
        advantages_all = torch.cat(advantages_all).detach()
        return actions_all, states_all, returns_all, baselines_all, log_probs_all, advantages_all, episode_rewards_all

    def train(self):
        self.logger.info(f"start training. device: {self.device}")
        all_total_rewards = (
            []
        )  # the returns of all episodes samples for training purposes
        averaged_total_rewards = []  # the returns for each iteration

        for t in range(self.num_batches):
            # collect a minibatch of samples
            actions, states, returns, baselines, log_probs, advantages, episode_rewards = self.sample_paths()  # (#episodes, ()), (#episodes,)
            all_total_rewards.extend(episode_rewards)

            # run training operations
            if self.train_mode:
                num_epochs = self.ppo_epochs if self.use_ppo else 1
                for epoch in range(num_epochs):
                    # train policy
                    if self.use_ppo:
                        if not self.discrete:
                            mean, std = self.policy_network(states)
                            dist = ptd.Normal(mean, std)
                            new_log_probs = dist.log_prob(actions).sum(1)  # sum on dim1 which sums all action dimensions
                        else:
                            dist = ptd.Categorical(torch.exp(self.policy_network(states)))  # softmax
                            new_log_probs = dist.log_prob(actions)  # sum on dim1 which sums all action dimensions
                        ratios = torch.exp(new_log_probs - log_probs.detach())
                        value1 = ratios * advantages
                        value2 = torch.clamp(ratios, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps) * advantages
                        policy_loss = -torch.min(value1, value2).mean()

                        new_baselines = self.baseline_network(states).squeeze()
                        baseline_loss = torch.nn.MSELoss()(returns, new_baselines)
                    else:
                        policy_loss = -(advantages * log_probs).mean()
                        baseline_loss = torch.nn.MSELoss()(returns, baselines)

                    self.policy_optim.zero_grad()
                    policy_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
                    self.policy_optim.step()

                    # train baseline
                    self.baseline_optim.zero_grad()
                    baseline_loss.backward()
                    self.baseline_optim.step()

            # compute reward statistics for this batch and log
            avg_reward = np.mean(episode_rewards)
            sigma_reward = np.sqrt(np.var(episode_rewards) / len(episode_rewards))
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}, #epi={}, batch_size={}".format(
                t, avg_reward, sigma_reward, len(episode_rewards), len(returns)
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
