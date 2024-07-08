import torch
import gymnasium as gym
import os
from utils.utils import save_model, get_device
from rl_agent.q_agent_cnn import QAgentCNN
from model import QNetworkCNN2
import datetime


if __name__ == "__main__":

    device = get_device()
    device = torch.device(device)
    print(f"Device: {device}")

    # Create the LunarLander environment
    env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, render_mode="human")
    # env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False)

    # Set parameters
    action_size = env.action_space.n

    base_path = os.getcwd()
    ts = str(datetime.datetime.now().timestamp()).split('.')[0]
    name = ts + "_car-racing_q-learning"
    path = os.path.join(base_path, name)
    config = {'path': path,
              'episodes': 1000,
              'max_steps': 2000,
              'grey': True,
              'step_stack': 3,  # combine certain steps (images) together as a state
              'lr': 0.002,
              'epsilon_start': 1.0,
              'epsilon_end': 0.01,
              'epsilon_decay': 0.993,
              'gamma': 0.99,
              'update_target_steps': 3000,  # update target network for every certain steps
              'terminate_at_reward_ma50': 230,
              'experience_replay': True,
              'skip_step': 2,  # skip certain steps without recording or learning
              'buffer_size': 3000,
              'batch_size': 64,
              'stop_by_negative_count': 300,  # stop an episode if the count of negative reward steps reaches certain number
              'burst_epsilon': 0.1,  # increase epsilon to this number if most of the recent episodes have negative return and the epsilon has reached epsilon_end
              'save_model_episode': 100
              }

    load_model_path = ''

    # Create the agent
    agent = QAgentCNN(config, env, config.get("step_stack"), action_size,
                      QNetworkCNN2, device)
    agent.load_model(load_model_path)

    # Train the agent
    reward_history = agent.train()

    will_save = True
    if will_save:
        save_model(agent.q_network, config, {'reward': reward_history})
