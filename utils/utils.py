import matplotlib.pyplot as plt
import torch
import numpy as np
import datetime
import os
import json


def draw_reward_history(rewards):
    window_size = 100
    ma = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.ion()
    plt.close()
    plt.figure(figsize=(15, 4))
    # plt.scatter(range(len(rewards)), rewards, color='red', marker='o', s=4)
    plt.plot(rewards, color='red', linestyle='--', linewidth=0.5, marker='o', markersize=2, label='reward')
    if len(rewards) > window_size:
        plt.plot(range(window_size - 1, len(rewards)), ma, color="blue", linewidth=1,
                 label=f'moving_average {window_size}')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('Reward History')
    plt.grid(True)
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)


def save_model(base_path, model, name, config, reward_history):
    ts = str(datetime.datetime.now().timestamp()).split('.')[0]
    dir_name = ts + '_' + name
    os.mkdir(os.path.join(base_path, dir_name))
    # save fig
    reward_fig_path = os.path.join(base_path, dir_name, 'reward_history.svg')
    plt.savefig(reward_fig_path)
    print(f"Save rewards figure to path: {reward_fig_path}")
    # save model
    model_path = os.path.join(base_path, dir_name, 'model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Save model to path: {model_path}")
    # save config
    config_path = os.path.join(base_path, dir_name, 'config.txt')
    json_str = json.dumps(config)
    with open(config_path, 'w') as f:
        f.write(json_str)
        print(f"Save config to path: {config_path}")
    # save reward history
    reward_history_path = os.path.join(base_path, dir_name, 'reward_history.txt')
    with open(reward_history_path, 'w') as file:
        for r in reward_history:
            file.write(str(r) + '\n')
        print(f"Save reward history to path: {reward_history_path}")


def draw_train_test_loss_history(train_loss, test_loss):
    plt.plot(train_loss, "train loss")
    plt.plot(test_loss, "test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()