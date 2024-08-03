import torch
import torch.nn as nn
import gymnasium as gym
import os
import datetime
from rl_agent.p_agent import PAgent
from utils.utils import get_device


class LinearNetwork(nn.Module):
    def __init__(self, state_dimension, action_dimension, action_activation_functions=None, fixed_std=None):
        super(LinearNetwork, self).__init__()
        self.fixed_std = fixed_std
        self.action_activation_functions = action_activation_functions
        self.fc1 = nn.Linear(state_dimension, 64)
        self.fc2 = nn.Linear(64, 128)
        # self.fc2_1 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.mean_layer = nn.Linear(64, action_dimension)
        if fixed_std is None:
            self.log_std_layer = nn.Linear(64, action_dimension)
        self.init_weights()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc2_1(x))
        x = torch.relu(self.fc3(x))
        mean = self.mean_layer(x)
        mean = self.apply_action_limit(mean)
        if self.fixed_std is None:
            log_std = self.log_std_layer(x)
            std = torch.exp(log_std)  # to make sure std is positive
        else:
            std = torch.full_like(mean, self.fixed_std)
        # if torch.isnan(mean).any() or torch.isnan(std).any():
        #     print("Nan values found in mean or std")
        #     print(f"Mean: {mean}")
        #     print(f"Std: {std}")
        return mean, std

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def apply_action_limit(self, data, for_actions=False):
        if self.action_activation_functions:
            for i, activation_function in enumerate(self.action_activation_functions):
                if activation_function is not None:
                    # for actions, if the range is [0, 1], then simply truncate the data because sigmoid will distort the data near 0
                    if for_actions:
                        if activation_function == torch.sigmoid:
                            if data.dim() == 1:
                                data[i] = torch.clamp(data[i], min=0, max=1.0)
                            else:  # batched
                                data[:, i] = torch.clamp(data[:, i], min=0, max=1.0)
                        elif activation_function == torch.tanh:
                            if data.dim() == 1:
                                data[i] = torch.clamp(data[i], min=-1.0, max=1.0)
                            else:  # batched
                                data[:, i] = torch.clamp(data[:, i], min=-1.0, max=1.0)
                    else:
                        if data.dim() == 1:
                            data[i] = activation_function(data[i])
                        else:  # batched
                            data[:, i] = activation_function(data[:, i])
        return data


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
    # env = gym.make('LunarLander-v2', render_mode="human")

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
        "fixed_std": 0.3,  # use fixed std for Gaussian distributions, set to None or don't set it to use trained std
        "discrete": False,
        "train": True,
        "use_baseline": True,
        "normalize_advantage": True,
        "ppo": True,  # use proximal policy optimization
        "ppo_clip_eps": 0.2,  # clip epsilon for ppo. this value is useless if 'ppo' is False
        "gamma": 0.99,
        "lr": 0.001,
        "num_batches": 2000,
        "batch_size": 400,
        "max_ep_len": 800,
        "early_stop_reward": 200,  # early stop when average reward reaches this value. set to None or 0 to disable
        "early_stop_ma": 50  # window size of moving average for early stop
    }

    # continuous action has 2 dimensions with range [-1, 1], so use tanh
    policy_network = LinearNetwork(observation_dim, action_dim, (torch.tanh, torch.tanh), fixed_std=config.get("fixed_std", None))
    baseline_network = LinearNetworkBaseline(observation_dim)

    p_agent = PAgent(config, env, policy_network, baseline_network, device)
    p_agent.load_model(load_model_path)

    p_agent.train()
