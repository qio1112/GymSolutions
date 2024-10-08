import gymnasium as gym
import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np
from collections import deque
import random

import os
import logging

from torch.distributions import Categorical

from utils.utils import draw_reward_history, save_model


class PAgent:

    def __init__(self, config, env, policy_network, baseline_network=None, device="cpu"):
        # torch.autograd.set_detect_anomaly(True)
        self.config = config
        self.train_mode = self.config.get("train", True)  # train model if True
        self.device_name = device  # device, cpu/mps/cuda
        self.device = torch.device(device)
        self.env = env
        self.base_path = self.config.get("path")  # path to store logs and models
        if self.base_path:
            os.makedirs(self.base_path, exist_ok=True)
        self.logger = get_logger(config.get("log_path"))

        self.discrete = self.config.get("discrete", False)  # True if action space is discrete, False if continuous
        self.observation_dim = self.env.observation_space.shape[0]  # num of dimensions of state
        self.action_dim = (  # action dimensions
            self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        )

        self.gamma = self.config.get("gamma", 0.99)  # gamma when calculating G_t
        self.lr = self.config.get("learning_rate", 0.001)
        self.normalize_advantage = self.config.get("normalize_advantage", True)
        self.batch_size = self.config.get("batch_size", 1)  # num of steps in each batch, at least 1 episode will be in the batch
        self.max_ep_len = self.config.get("max_ep_len", 2000)  # maximum num of steps in an episode
        self.num_batches = self.config.get("num_batches", 300)  # number of batches to run
        self.actor_critic = self.config.get("actor_critic", False)  # use actor-critic method or not. If True, use bootstrapping for advantages
        self.use_ppo = self.config.get("ppo", False)
        self.ppo_clip_eps = self.config.get("ppo_clip_eps", 0.2)
        self.ppo_epochs = self.config.get("ppo_epochs", 4)
        self.ppo_mini_batch_size = self.config.get("ppo_mini_batch_size", self.batch_size)
        self.entropy_coef = self.config.get("entropy_coef", 0)  # parameter of entropy regularization used for policy loss
        self.clip_policy_grad = self.config.get("clip_policy_grad", False)  # True to clip the gradients of policy model after each training
        self.early_stop_reward = self.config.get("early_stop_reward", 0)  # if value > 0, stop training when moving averaged reward reaches this value
        self.early_stop_ma = self.config.get("early_stop_ma", 50)  # stop training based on this moving average window
        self.random_action_ratio = self.config.get("random_action_ratio", 0)  # probability of using random action
        self.random_action_decaying_rate = self.config.get("random_action_decaying_rate", 0.995)  # decaying ratio of the probability over episodes

        self.image_input = self.config.get("image_input", False)  # input is image which has a shape similar to (batch_size, L, W, dim)
        self.grey = self.config.get("grey", False)  # for image input only, for image inputs only, calculate mean of 3 RGB channels into 1 channel
        self.sample_image_stack_size = self.config.get("sample_image_stack_size", 1)  # stack size of each state
        self.sample_skip = self.config.get("sample_skip", 0)  # skip certain number of steps for sampling
        self.stop_episode_reward = self.config.get("stop_episode_reward", None)  # stop the episode if total reward is less than this number
        self.baseline_network = baseline_network.to(self.device)
        self.baseline_optim = torch.optim.Adam(self.baseline_network.parameters(), lr=self.lr)
        self.policy_network = policy_network.to(self.device)
        self.policy_optim = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)

        # used for sampling only
        self.state = None
        self.state_stack_deque = None
        self.start_next_episode = True
        self.episode_done = False
        self.action = None
        self.len_of_cur_episode = 0
        self.total_reward_of_cur_episode = 0
        self.episode_rewards_history = []
        self.cur_batch_size = 0

    def enrich_reward(self, reward, next_state, done):
        return reward

    def load_model(self, weights_path, baseline_weights_path=None):
        if weights_path:
            self.policy_network.load_state_dict(torch.load(weights_path))
            self.policy_network.to(self.device)
            print(f"Load policy model from {weights_path}")
        if baseline_weights_path:
            self.baseline_network.load_state_dict(torch.load(weights_path))
            self.baseline_network.to(self.device)
            print(f"Load baseline model from {weights_path}")

    def get_state_tensor(self):
        state_np = np.array(self.state_stack_deque)
        if self.image_input:
            state_tensor = transform_state(torch.FloatTensor(state_np).unsqueeze(0), self.grey).to(self.device)
        else:
            state_tensor = torch.FloatTensor(state_np).to(self.device)
        return state_tensor

    def sample_next_step(self, skip_this_step):
        action, log_prob, baseline, state_tensor, entropy = None, None, None, None, None
        if not skip_this_step:
            state_tensor = self.get_state_tensor()
            if not self.discrete:
                mean, std = self.policy_network(state_tensor)
                dist = ptd.Normal(mean, std)
            else:
                dist = ptd.Categorical(torch.exp(self.policy_network(state_tensor)))  # softmax
            next_action = dist.sample()
            entropy = dist.entropy()
            self.action = next_action
            # use random action
            if self.random_action_ratio > 0 and random.random() < self.random_action_ratio:
                if self.discrete:
                    self.action = torch.randint(0, 5, self.action.shape).to(self.device)
                else:
                    # TODO - implement random action for continuous action space
                    pass
            log_prob = dist.log_prob(self.action).sum(-1)  # sum for all action dimensions
            if self.discrete:
                log_prob = log_prob.unsqueeze(0)
            baseline = self.baseline_network(state_tensor).squeeze(0)
        # if the step is skipped from being recorded, use the same action as the previous action
        next_state, reward, done, _, _ = self.env.step(self.action.cpu().numpy()[0])
        reward = self.enrich_reward(reward, next_state, done)
        return state_tensor, next_state, log_prob, baseline, reward, done, entropy

    def sample_episode(self):
        if self.start_next_episode:
            if len(self.episode_rewards_history) > 0:
                self.random_action_ratio *= self.random_action_decaying_rate
            self.len_of_cur_episode = 0
            self.total_reward_of_cur_episode = 0
            self.state = self.env.reset()[0]
            self.state_stack_deque = deque([self.state] * self.sample_image_stack_size, maxlen=self.sample_image_stack_size)
        states, actions, rewards, log_probs, baselines, entropies = [], [], [], [], [], []
        episode_reward = 0  # sum of rewards of this episode or this batch
        skip_step_count = self.sample_skip + 1  # skip only happens inside a batch but won't across batches
        batch_steps_count = 0
        while True:
            skip_this_step = self.sample_skip > 0 and skip_step_count <= self.sample_skip
            state_tensor, next_state, log_prob, baseline, reward, done, entropy = self.sample_next_step(skip_this_step)
            self.episode_done = done
            self.len_of_cur_episode += 1
            self.total_reward_of_cur_episode += reward
            self.start_next_episode = done
            if not skip_this_step:
                self.cur_batch_size += 1
                skip_step_count = 0
                states.append(state_tensor)
                actions.append(self.action)
                log_probs.append(log_prob)
                baselines.append(baseline)
                rewards.append(reward)
                self.state_stack_deque.append(next_state)
                entropies.append(entropy)
                batch_steps_count += 1

            skip_step_count += 1
            self.state = next_state
            episode_reward = episode_reward + reward

            # stop the episode if episode is done, or episode is longer then the max episode length, or the reward is less than a certain reward
            early_stop = self.len_of_cur_episode >= self.max_ep_len or (self.stop_episode_reward is not None and self.total_reward_of_cur_episode < self.stop_episode_reward)
            if done or early_stop:
                self.start_next_episode = True
                self.episode_rewards_history.append(self.total_reward_of_cur_episode)
                break
            if self.actor_critic and self.cur_batch_size == self.batch_size:
                #  stop the sampling for now, but will continue the same episode after training
                break

        states_tensor = torch.cat(states).to(self.device)
        actions_tensor = torch.cat(actions).to(self.device)
        baselines_tensor = torch.cat(baselines).to(self.device)
        log_probs_tensor = torch.cat(log_probs).to(self.device)
        entropies_tensor = torch.cat(entropies).to(self.device)

        returns = []
        G = 0
        if not self.actor_critic:
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
        else:
            #  use bootstrapping to calculate G_t
            for i in reversed(range(len(rewards))):
                if i == len(rewards) - 1:
                    next_value = 0
                    if not self.episode_done:
                        next_state_tensor = self.get_state_tensor()  # get the next state if it's calculating the G_t for the last sampled state
                        next_value = self.baseline_network(next_state_tensor).item()
                    G = rewards[i] + self.gamma * next_value
                else:
                    G = rewards[i] + self.gamma * G  # bootstrapping
                returns.insert(0, G)

        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = returns_tensor - baselines_tensor
        if self.normalize_advantage and advantages_tensor.shape[0] > 1:
            advantages_tensor = (advantages_tensor - torch.mean(advantages_tensor)) / torch.std(advantages_tensor)
        return actions_tensor, states_tensor, returns_tensor, baselines_tensor, log_probs_tensor, advantages_tensor, episode_reward, entropies_tensor

    def sample_paths(self):
        self.cur_batch_size = 0
        actions_all, states_all, returns_all, baselines_all, log_probs_all, advantages_all, episode_rewards_all, entropies_all = [], [], [], [], [], [], [], []
        # sample some whole episodes until the total number of steps is greater than expected batch size
        while self.cur_batch_size < self.batch_size:
            actions, states, returns, baselines, log_probs, advantages, episode_reward, entropies = self.sample_episode()
            actions_all.append(actions)
            states_all.append(states)
            returns_all.append(returns)
            baselines_all.append(baselines)
            log_probs_all.append(log_probs)
            advantages_all.append(advantages)
            episode_rewards_all.append(episode_reward)
            entropies_all.append(entropies)
        actions_all = torch.cat(actions_all)
        states_all = torch.cat(states_all)
        returns_all = torch.cat(returns_all)
        baselines_all = torch.cat(baselines_all)
        log_probs_all = torch.cat(log_probs_all)
        advantages_all = torch.cat(advantages_all).detach()
        entropies_all = torch.cat(entropies_all).detach()
        return actions_all, states_all, returns_all, baselines_all, log_probs_all, advantages_all, episode_rewards_all, entropies_all

    def train(self):
        try:
            self.logger.info(f"start training. device: {self.device}")
            all_total_rewards = []  # the returns of all episodes samples for training purposes

            for t in range(self.num_batches):
                # collect a minibatch of samples
                actions, states, returns, baselines, log_probs, advantages, episode_rewards, entropies = self.sample_paths()  # (#episodes, ()), (#episodes,)
                all_total_rewards.extend(episode_rewards)

                # run training operations
                if self.train_mode:
                    if self.use_ppo:
                        num_epochs = self.ppo_epochs
                        for epoch in range(num_epochs):
                            # create random index for mini-batches
                            num_samples = actions.shape[0]
                            indices = np.random.permutation(num_samples)
                            for start_idx in range(0, num_samples, self.ppo_mini_batch_size):
                                end_idx = start_idx + self.ppo_mini_batch_size
                                batch_indices = indices[start_idx:end_idx]
                                mini_batch_states = states[batch_indices]
                                mini_batch_actions = actions[batch_indices]
                                mini_batch_log_probs = log_probs[batch_indices]
                                mini_batch_advantages = advantages[batch_indices]
                                mini_batch_entropies = entropies[batch_indices]
                                mini_batch_returns = returns[batch_indices]

                                if not self.discrete:
                                    mean, std = self.policy_network(mini_batch_states)
                                    dist = ptd.Normal(mean, std)
                                    new_log_probs = dist.log_prob(mini_batch_actions).sum(1)  # sum on dim1 which sums all action dimensions
                                else:
                                    dist = ptd.Categorical(torch.exp(self.policy_network(mini_batch_states)))  # softmax
                                    new_log_probs = dist.log_prob(mini_batch_actions)  # sum on dim1 which sums all action dimensions
                                ratios = torch.exp(new_log_probs - mini_batch_log_probs.detach())
                                value1 = ratios * mini_batch_advantages
                                value2 = torch.clamp(ratios, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps) * mini_batch_advantages
                                policy_loss = -torch.min(value1, value2).mean()
                                new_baselines = self.baseline_network(mini_batch_states).squeeze()
                                baseline_loss = torch.nn.MSELoss()(mini_batch_returns, new_baselines)
                                entropy = mini_batch_entropies.mean()
                                policy_loss = policy_loss - self.entropy_coef * entropy
                                self.policy_optim.zero_grad()
                                policy_loss.backward()
                                if self.clip_policy_grad:
                                    torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
                                self.policy_optim.step()
                                self.baseline_optim.zero_grad()
                                baseline_loss.backward()
                                self.baseline_optim.step()
                    else:  # not using PPO
                        policy_loss = -(advantages * log_probs).mean()
                        baseline_loss = torch.nn.MSELoss()(returns, baselines)
                        entropy = entropies.mean()
                        policy_loss = policy_loss - self.entropy_coef * entropy
                        self.policy_optim.zero_grad()
                        policy_loss.backward()
                        if self.clip_policy_grad:
                            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
                        self.policy_optim.step()
                        self.baseline_optim.zero_grad()
                        baseline_loss.backward()
                        self.baseline_optim.step()

                # logs
                history20 = self.episode_rewards_history[-20:]
                history50 = self.episode_rewards_history[-50:]
                history100 = self.episode_rewards_history[-100:]
                ma20, ma50, ma100 = 0, 0, 0
                if len(self.episode_rewards_history) > 0:
                    ma20 = sum(history20) / len(history20)
                    ma50 = sum(history50) / len(history50)
                    ma100 = sum(history100) / len(history100)
                if not self.actor_critic:
                    # compute reward statistics for this batch and log
                    avg_reward = np.mean(episode_rewards)
                    sigma_reward = np.sqrt(np.var(episode_rewards) / len(episode_rewards))
                    msg = f"[ITERATION {t}]: Average reward: {avg_reward:04.2f} +/- {sigma_reward:04.2f}, #epi={len(episode_rewards)}, batch_size={len(returns)}, #total_epi={len(self.episode_rewards_history)}, episode rewards - ma20={ma20:04.2f}, ma50={ma50:04.2f}, ma100={ma100:04.2f}"
                else:
                    batch_reward = np.sum(episode_rewards)
                    msg = f"[ITERATION {t}]: Batch reward: {batch_reward:04.2f}, #epi={len(episode_rewards)}, batch_size={len(returns)}, #total_epi={len(self.episode_rewards_history)}, episode rewards - ma20={ma20:04.2f}, ma50={ma50:04.2f}, ma100={ma100:04.2f}"
                self.logger.info(msg)

                if self.early_stop_reward > 0:
                    last_history = self.episode_rewards_history[-self.early_stop_ma:]
                    ma = sum(last_history) / len(last_history) if len(self.episode_rewards_history) > 0 else 0
                    if len(self.episode_rewards_history) > self.early_stop_ma and ma > self.early_stop_reward:
                        self.logger.info(f"Terminating training with episode rewards ma{self.early_stop_ma}={self.early_stop_reward}")
                        break

            self.logger.info("- Training done.")
            draw_reward_history(self.episode_rewards_history, window_size=self.early_stop_ma)
            save_model(self.policy_network, self.config, {"rewards": all_total_rewards})
        except (KeyboardInterrupt, ValueError) as e:
            self.logger.error(e)
            save_model(self.policy_network, self.config, history=None)


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


# used by cnn networks
def transform_state(state, grey=False):
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
