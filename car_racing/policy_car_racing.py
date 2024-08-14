import torch
import torch.nn as nn
import gymnasium as gym
import os

from car_racing.model import CnnNetwork2
from rl_agent.p_agent import PAgent
from utils.utils import save_model, get_device
import datetime


class CnnNetwork(nn.Module):

    def __init__(self, input_channel, action_size):
        super(CnnNetwork, self).__init__()
        # input shape (batch_size, <channel>, 96, 96)
        # cnn_out_size = ((input_width - kernel_size + 2 * padding) / stride) + 1
        # pooling_out_size = ((input_size - kernel_size) / stride) + 1 ,  if kernel_size=2 and stride=2, pooling_out_size = input_size / 2
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=8, stride=4, padding=4),  # (batch_size, 12, 96, 96)
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 12, 48, 48)
            nn.Dropout(p=0.05)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=3, padding=2),  # (batch_size, 24, 48, 48)
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 24, 24, 24)
            nn.Dropout(p=0.05)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (batch_size, 48, 24, 24)
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 48, 12, 12)
            nn.Dropout(p=0.05)
        )
        self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(48 * 12 * 12, 256)
        self.linear1 = nn.Linear(128 * 9 * 9, 256)
        self.linear3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = torch.relu(self.linear1(x))
        # x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class CnnNetworkNew(nn.Module):
    def __init__(self, input_channel, action_size):
        super(CnnNetworkNew, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(input_channel, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.flatten = nn.Flatten()
        self.l1 = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, action_size))
        # self.apply(self._weights_init)
        self.init_weights()

    # @staticmethod
    # def _weights_init(m):
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
    #         nn.init.constant_(m.bias, 0.1)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.cnn_base(x)
        x = self.flatten(x)
        x = self.l1(x)
        return x


if __name__ == "__main__":

    device = get_device()
    device = torch.device(device)
    print(f"Device: {device}")

    # Create the LunarLander environment
    env = gym.make("CarRacing-v2",
                   domain_randomize=False,  # color of images
                   continuous=False,
                   # )
                   render_mode="human")
    
    # env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False)

    # Set parameters
    action_dim = env.action_space.n

    base_path = os.getcwd()
    ts = str(datetime.datetime.now().timestamp()).split('.')[0]
    name = ts + "_policy_gradient_baseline"
    path = os.path.join(base_path, name)
    log_path = os.path.join(path, "log.txt")
    load_model_path = ""

    config = {
        "path": path,
        "log_path": log_path,
        "discrete": True,
        "train": True,
        "use_baseline": True,
        "normalize_advantage": True,
        "ppo": True,  # use proximal policy optimization
        "ppo_clip_eps": 0.3,  # clip epsilon for ppo. this value is useless if 'ppo' is False
        "ppo_epochs": 4,
        "clip_policy_grad": True,
        "gamma": 0.99,
        "lr": 3e-5,
        "image_input": True,
        "sample_image_stack_size": 3,
        "sample_skip": 3,
        "grey": True,  # use grey for image inputs
        "num_batches": 3000,
        "sample_whole_episodes": False,  # True if sample every certain steps, False if train only whole episodes
        "batch_size": 50,
        "max_ep_len": 1500,
        "stop_episode_reward": -50,
        "stop_batch_reward": -35,
        "early_stop_reward": 500,  # early stop when average reward reaches this value. set to None or 0 to disable
        "early_stop_ma": 50,  # window size of moving average for early stop
        # "is_continue": 300
        "random_action_ratio": 0.15,
        "random_action_decaying_rate": 0.99,
    }

    if config.get("ppo", False):
        name += "_ppo"

    input_dim = config.get("sample_image_stack_size", 1) if config.get("grey", False) else (3 * config.get("sample_image_stack_size", 1))
    baseline_network = CnnNetwork(input_dim, 1)
    policy_network = CnnNetworkNew(input_dim, action_dim)

    p_agent = PAgent(config, env, policy_network, baseline_network, device)
    p_agent.load_model(load_model_path)

    p_agent.train()
