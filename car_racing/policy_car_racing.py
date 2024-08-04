import torch
import torch.nn as nn
import gymnasium as gym
import os
from utils.utils import save_model, get_device
from rl_agent.p_agent import PAgent
from model import CnnNetwork2, CnnNetwork1
import datetime


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


if __name__ == "__main__":

    device = get_device()
    device = torch.device(device)
    print(f"Using device: {device}")

    # Create the LunarLander environment
    env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, render_mode="human")
    # env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False)

    # Set parameters
    action_size = env.action_space.n

    base_path = os.getcwd()
    ts = str(datetime.datetime.now().timestamp()).split('.')[0]
    name = ts + "_policy_gradient_baseline"
    path = os.path.join(base_path, name)
    log_path = os.path.join(path, "log.txt")
    load_model_path = ""

    config = {
        "path": path,
        "log_path": log_path,
        "train": True,
        "use_baseline": True,
        "normalize_advantage": True,
        "ppo": True,  # use proximal policy optimization
        "ppo_clip_eps": 0.2,  # clip epsilon for ppo. this value is useless if 'ppo' is False
        "gamma": 0.99,
        "lr": 0.002,
        "num_batches": 500,
        "batch_size": 800,
        "max_ep_len": 2000,
        "early_stop_reward": 200,
        # early stop when average reward reaches this value. set to None or 0 to disable
        "early_stop_ma": 50  # window size of moving average for early stop
    }

    if config.get("ppo", False):
        name += "_ppo"

    baseline_network = CnnNetwork2()