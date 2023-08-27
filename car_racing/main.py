import torch
import gymnasium as gym
import os
from utils.utils import save_model
from rl_agent.QAgent import QAgentCNN
from model import QNetworkCNN1
import datetime


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create the LunarLander environment
    # env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, render_mode="human")
    env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False)

    # Set parameters
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    base_path = os.getcwd()
    ts = str(datetime.datetime.now().timestamp()).split('.')[0]
    name = ts + "_car-racing_q-learning"
    path = os.path.join(base_path, name)
    config = {'path': path,
              'episodes': 1000,
              'max_steps': 2000,
              'epsilon_start': 1,
              'epsilon_end': 0.02,
              'epsilon_decay': 0.996,
              'gamma': 0.99,
              'update_target_steps': 200,
              'terminate_at_reward_ma50': 230,
              'buffer_size': 3000,
              'experience_replay': True,
              'experience_replay_size': 300,
              'batch_size': 32,
              'save_model_episode': 10}

    load_model_path = ''

    # Create the agent
    agent = QAgentCNN(config, action_size, device, env, QAgentCNN)
    agent.load_model(load_model_path)

    # Train the agent
    reward_history = agent.train()

    will_save = True
    if will_save:
        save_model(agent.q_network, config, {'reward': reward_history})
