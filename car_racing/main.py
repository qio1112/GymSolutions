import torch
import gymnasium as gym
import os
import random
from collections import deque
from utils.utils import save_model, draw_reward_history
from agent import QAgent
from model import QNetworkCNN1


def train_agent(env, agent, config):
    epsilon = config['epsilon_start']
    batch_size = config['batch_size']
    if not batch_size:
        batch_size = 1
    batch_count = 0
    reward_history = []
    buffer = deque(maxlen=3000)
    steps = 0

    for episode in range(config['episodes']):
        state = env.reset(options={"randomize": False})[0]
        total_reward = 0

        for step in range(config['max_steps']):
            steps += 1
            batch_count += 1
            action = agent.get_action(state, epsilon)
            next_state, reward, done, *_ = env.step(action)
            buffer.append((state, action, reward, next_state, done))

            # TD method, train for each step
            if batch_count == batch_size:
                batch_count = 0
                batch_data = list(buffer)[-50:]
                agent.train([t[0] for t in batch_data],
                            [t[1] for t in batch_data],
                            [t[2] for t in batch_data],
                            [t[3] for t in batch_data],
                            [t[4] for t in batch_data],
                            config['gamma'])

            state = next_state
            total_reward += reward

            # update target network
            if config['update_target_steps'] and steps % config['update_target_steps'] == 0:
                agent.update_target_network()

            if done:
                break

        reward_history.append(total_reward)

        ma50 = sum(reward_history[-50:]) / 50
        if len(reward_history) % 10 == 0 or len(reward_history) >= config['episodes']-10:
            draw_reward_history(reward_history)
        # reduce the e-greedy epsilon by num of episodes
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])
        print("Episode: {}, Total Reward: {}, MA50: {}".format(episode + 1, total_reward, ma50))

        # terminate training if ma50 is larger than terminate_at_reward_ma50
        if len(reward_history) > 50 and \
                config['terminate_at_reward_ma50'] and \
                ma50 > config['terminate_at_reward_ma50']:
            draw_reward_history(reward_history)
            print(f"Terminating training with ma50={ma50}")
            break

        # experience replay for every 25 episodes
        if config['experience_replay'] and episode > 50 and episode % 25 == 0:
            replay_size = config['experience_replay_size']
            selected_exp = random.sample(buffer, k=300)
            agent.train([t[0] for t in selected_exp],
                        [t[1] for t in selected_exp],
                        [t[2] for t in selected_exp],
                        [t[3] for t in selected_exp],
                        [t[4] for t in selected_exp],
                        config['gamma'])
            agent.update_target_network()
            print(f"Experience Replayed")
    return reward_history


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create the LunarLander environment
    env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, render_mode="human")

    # Set parameters
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    config = {'episodes': 5000,
              'max_steps': 2000,
              'epsilon_start': 1,
              'epsilon_end': 0.01,
              'epsilon_decay': 0.996,
              'gamma': 0.99,
              'update_target_steps': 1,
              'terminate_at_reward_ma50': 230,
              'experience_replay': False,
              # 'experience_replay_size': 300,
              'batch_size': 1}

    load_model_path = ''

    # Create the agent
    agent = QAgent(action_size, QNetworkCNN1, device)
    agent.load_model(load_model_path)

    # Train the agent
    reward_history = train_agent(env, agent, config)

    will_save = True
    if will_save:
        save_model(os.getcwd(), agent.q_network, 'fixed_target_replay_learning_batched', config, {'reward': reward_history})
