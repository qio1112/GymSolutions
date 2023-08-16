from collections import deque
import random
from utils.utils import draw_reward_history, save_model


def train_agent(env, agent, config):
    epsilon = config['epsilon_start']
    batch_size = config.get('batch_size', 1)
    buffer_size = config.get('buffer_size', 30000)
    reset_options = config.get('reset_options', None)
    save_model_episode = config.get('save_model_episode', None)

    batch_count = 0
    reward_history = []
    buffer = deque(maxlen=buffer_size)
    steps = 0

    for episode in range(config['episodes']):
        state = env.reset(options=reset_options)[0]
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

        last_50_history = reward_history[-50:]
        ma50 = sum(last_50_history) / len(last_50_history)
        if len(reward_history) % 10 == 0 or len(reward_history) >= config['episodes']-10:
            draw_reward_history(reward_history)

        # reduce the e-greedy epsilon by num of episodes
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])
        print("Episode: {}, Total Reward: {:.2f}, MA50: {:.2f}, epsilon: {:.4f}".format(episode + 1, total_reward, ma50, epsilon))

        # terminate training if ma50 is larger than terminate_at_reward_ma50
        if len(reward_history) > 50 and \
                config['terminate_at_reward_ma50'] and \
                ma50 > config['terminate_at_reward_ma50']:
            draw_reward_history(reward_history)
            print(f"Terminating training with ma50={ma50}")
            break

        if save_model_episode and (episode+1) > save_model_episode and (episode+1) % save_model_episode == 0:
            save_model(agent.target_network, config, {'reward': reward_history}, additional_path='epi='+str(episode+1))

        # experience replay for every 25 episodes
        if config['experience_replay'] and episode > 50 and episode % 25 == 0:
            replay_size = config['experience_replay_size']
            selected_exp = random.sample(buffer, k=replay_size)
            agent.train([t[0] for t in selected_exp],
                        [t[1] for t in selected_exp],
                        [t[2] for t in selected_exp],
                        [t[3] for t in selected_exp],
                        [t[4] for t in selected_exp],
                        config['gamma'])
            agent.update_target_network()
            print(f"Experience Replayed")
    return reward_history
