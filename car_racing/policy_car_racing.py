import torch
import torch.nn as nn
import gymnasium as gym
import os
import numpy as np

from car_racing.model import CnnNetwork2, CnnNetwork1, CnnNetwork3
from rl_agent.p_agent import PAgent
from utils.utils import save_model, get_device
import datetime


class CarRacingPolicyAgent(PAgent):
    def __init__(self, configu, env, policy_n, baseline_n=None, dev="cpu"):
        super().__init__(configu, env, policy_n, baseline_n, dev)

    def enrich_reward(self, reward, next_state, done):
        # if reward > 0:
        #     reward = reward * 1.1

        # check car on track
        top_on_track = next_state[64, 47, 1] < 120
        if top_on_track:
            reward += 0.1
        else:
            reward -= 0.1

        # penalty for green screen
        if np.mean(next_state[:, :, 1]) > 180.0:
            reward -= 0.1
        if done:
            reward = -1  # reduce the penalty of done
        return reward


if __name__ == "__main__":

    device = get_device()
    device = torch.device(device)
    print(f"Device: {device}")

    # Create the LunarLander environment
    env = gym.make("CarRacing-v2",
                   domain_randomize=False,  # color of images
                   continuous=False,
                   # render_mode="human",
                   )
    
    # env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False)

    # Set parameters
    action_dim = env.action_space.n

    base_path = os.getcwd()
    ts = str(datetime.datetime.now().timestamp()).split('.')[0]
    name = ts + "_policy_gradient_baseline"
    path = os.path.join(base_path, name)
    log_path = os.path.join(path, "log.txt")
    load_model_path = ""
    # load_model_path = "/Users/yipeng/Desktop/programs/python/GymSolutions/car_racing/1724035780_policy_gradient_baseline/model.pt"

    config = {
        "path": path,
        "log_path": log_path,
        "discrete": True,
        "train": True,
        "use_baseline": True,
        "normalize_advantage": True,
        "ppo": True,  # use proximal policy optimization
        "ppo_clip_eps": 0.2,  # clip epsilon for ppo. this value is useless if 'ppo' is False
        "ppo_epochs": 6,
        "ppo_mini_batch_size": 250,
        "actor_critic": True,
        "clip_policy_grad": True,
        "gamma": 0.99,
        "lr": 1e-3,
        "image_input": True,
        "sample_image_stack_size": 5,
        "sample_skip": 2,
        "grey": True,  # use grey for image inputs
        "num_batches": 3000,
        "batch_size": 500,
        "max_ep_len": 2000,
        "stop_episode_reward": -200,
        "early_stop_reward": 500,  # early stop when average reward reaches this value. set to None or 0 to disable
        "early_stop_ma": 50,  # window size of moving average for early stop
        "random_action_ratio": 0.1,
        "random_action_decaying_rate": 0.994,
    }

    if config.get("ppo", False):
        name += "_ppo"

    input_dim = config.get("sample_image_stack_size", 1) if config.get("grey", False) else (3 * config.get("sample_image_stack_size", 1))
    baseline_network = CnnNetwork3(input_dim, 1)
    policy_network = CnnNetwork3(input_dim, action_dim)

    p_agent = CarRacingPolicyAgent(config, env, policy_network, baseline_network, device)
    p_agent.load_model(load_model_path)

    p_agent.train()
