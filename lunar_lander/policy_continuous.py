import torch
import torch.nn as nn
import gymnasium as gym
import os
import datetime
from rl_agent.p_agent import PAgent
from utils.utils import get_device


class LinearNetwork(nn.Module):
    def __init__(self, state_dimension, action_dimension, param_std=True):
        super(LinearNetwork, self).__init__()
        self.param_std = param_std
        self.fc1 = nn.Linear(state_dimension, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.mean_layer = nn.Linear(64, action_dimension)
        if self.param_std:
            self.log_std = nn.Parameter(torch.zeros(action_dimension))
        else:
            self.log_std = nn.Linear(64, action_dimension)
        # self.init_weights()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mean = torch.tanh(self.mean_layer(x))
        if self.param_std:
            std = torch.exp(self.log_std)
        else:
            std = torch.exp(self.log_std(x))
        return mean, std

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class LinearNetworkBaseline(nn.Module):
    def __init__(self, state_dimension):
        super(LinearNetworkBaseline, self).__init__()
        self.fc1 = nn.Linear(state_dimension, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2",
                   # continuous=True,
                   # render_mode="human",
                   # gravity=-10.0,
                   # enable_wind=False,
                   # wind_power=15.0,
                   # turbulence_power=1.5,
                   )

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # device = get_device()
    device = "cpu"
    print(f"Using device: {device}")

    base_path = os.getcwd()
    ts = str(datetime.datetime.now().timestamp()).split('.')[0]
    name = ts + "_policy_gradient_continuous"
    path = os.path.join(base_path, name)
    log_path = os.path.join(path, "log.txt")
    # load_model_path = "/Users/yipeng/Desktop/programs/python/GymSolutions/lunar_lander/1693171440_policy_gradient_baseline/model.pt"
    load_model_path = ""

    config = {
        "path": path,
        "log_path": log_path,
        "param_std": False,  # if true, use nn.Parameter for std, else use a layer
        "discrete": False,
        "train": True,
        "use_baseline": True,
        "normalize_advantage": True,
        "ppo": True,  # use proximal policy optimization
        "ppo_clip_eps": 0.2,  # clip epsilon for ppo. this value is useless if 'ppo' is False
        "ppo_epochs": 4,
        "gamma": 0.99,
        "lr": 5e-4,
        "num_batches": 2000,  # number of sampling and train times
        "batch_size": 500,  # steps in each batch, the last episode may exceed this size
        "max_ep_len": 3000,  # max length of each episode
        "early_stop_reward": 200,  # early stop when average reward reaches this value. set to None or 0 to disable
        "early_stop_ma": 50  # window size of moving average for early stop
    }

    # continuous action has 2 dimensions with range [-1, 1], so use tanh
    policy_network = LinearNetwork(observation_dim, action_dim, config.get("param_std", True))
    baseline_network = LinearNetworkBaseline(observation_dim)

    p_agent = PAgent(config, env, policy_network, baseline_network, device)
    p_agent.load_model(load_model_path)

    p_agent.train()
