# -*- coding: UTF-8 -*-

import argparse
import numpy as np
import torch
import gymnasium as gym
from policy_gradient import PolicyGradient
from ppo import PPO
from config import get_config
import random

import pdb

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env-name", required=True, type=str, choices=["cartpole", "pendulum", "cheetah"]
)
parser.add_argument("--baseline", dest="use_baseline", action="store_true")
parser.add_argument("--no-baseline", dest="use_baseline", action="store_false")
parser.add_argument("--ppo", dest="ppo", action="store_true")
parser.add_argument("--seed", type=int, default=1)

parser.set_defaults(use_baseline=True)


if __name__ == "__main__":
    # args = parser.parse_args()
    #
    # torch.random.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    # env_name = "cartpole"
    env_name = "LunarLander-v2"
    use_baseline = True
    ppo = False
    seed = 1

    config = get_config(env_name, use_baseline, ppo, seed)
    env = gym.make(config.env_name)
    # train model
    model = PolicyGradient(env, config, seed) if not ppo else PPO(env, config, seed)
    model.run()
