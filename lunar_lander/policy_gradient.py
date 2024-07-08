import torch
import torch.nn as nn
import gymnasium as gym
import os
import datetime
from rl_agent.p_agent import PAgent
from utils.utils import get_device


class LinearNetwork1(nn.Module):
    def __init__(self, state_size, action_size):
        super(LinearNetwork1, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def build_mlp(input_size, output_size, n_layers, size, device):
    layers = []
    in_layer = nn.Linear(input_size, size)
    relu = nn.ReLU()
    layers.append(in_layer)
    layers.append(relu)
    for i in range(n_layers):
        layer_i = nn.Linear(size, size)
        layers.append(layer_i)
        layers.append(relu)
    out_layer = nn.Linear(size, output_size)
    layers.append(out_layer)

    return nn.Sequential(*layers).to(device)


if __name__ == "__main__":

    env = gym.make('LunarLander-v2')
    # env = gym.make('LunarLander-v2', render_mode="human")

    discrete = isinstance(env.action_space, gym.spaces.discrete.Discrete)
    observation_dim = env.observation_space.shape[0]
    action_dim = (
        env.action_space.n if discrete else env.action_space.shape[0]
    )

    device = get_device()
    print(f"Using device: {device}")

    base_path = os.getcwd()
    ts = str(datetime.datetime.now().timestamp()).split('.')[0]
    name = ts + "_policy_gradient_baseline"
    path = os.path.join(base_path, name)
    log_path = os.path.join(path, "log.txt")
    # load_model_path = "/Users/yipeng/Desktop/programs/python/GymSolutions/lunar_lander/1693171440_policy_gradient_baseline/model.pt"
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
        "num_batches": 330,
        "batch_size": 1000,
        "max_ep_len": 2000,
        "early_stop_reward": 220,  # early stop when average reward reaches this value. set to None or 0 to disable
        "early_stop_ma": 50  # window size of moving average for early stop
    }

    if config.get("ppo", False):
        name += "_ppo"

    # NETWORK=1
    NETWORK = 2

    # baseline_network = None
    # policy_network = None
    if NETWORK == 1:
        baseline_network = build_mlp(observation_dim, 1, 2, 256, device)
        policy_network = build_mlp(observation_dim, action_dim, 2, 256, device)
    elif NETWORK == 2:
        baseline_network = LinearNetwork1(observation_dim, 1)
        policy_network = LinearNetwork1(observation_dim, action_dim)

    p_agent = PAgent(config, env, policy_network, baseline_network, device)
    p_agent.load_model(load_model_path)

    p_agent.train()
