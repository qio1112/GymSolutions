import torch
import gymnasium as gym
import os
from utils.utils import save_model
from agent import QAgent
from model import QNetworkCNN1
from utils.rl_trainer import train_agent


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create the LunarLander environment
    env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, render_mode="human")
    # env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False)

    # Set parameters
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    config = {'episodes': 3000,
              'max_steps': 3000,
              'epsilon_start': 0.5,
              'epsilon_end': 0.01,
              'epsilon_decay': 0.996,
              'gamma': 0.99,
              'update_target_steps': 200,
              'terminate_at_reward_ma50': 200,
              'buffer_size': 30000,
              'experience_replay': True,
              'experience_replay_size': 2000,
              'replay_after_episode': 3,
              'replay_every_episode': 1,
              'batch_size': 64}

    load_model_path = ''

    # Create the agent
    agent = QAgent(action_size, QNetworkCNN1, device)
    agent.load_model(load_model_path)

    # Train the agent
    reward_history = train_agent(env, agent, config)

    will_save = True
    if will_save:
        save_model(os.getcwd(), agent.q_network, 'fixed_target_replay_learning_batched', config, {'reward': reward_history})
