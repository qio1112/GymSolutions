import matplotlib.pyplot as plt
import torch
import numpy as np
import datetime
import os
import json


def draw_reward_history(rewards, window_size=100, block=False):
    ma = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
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
    if not block:
        plt.show(block=False)
        plt.pause(0.1)


def save_model(model, config, history, path=None, additional_path=None, save_fig=True):
    if not path:
        path = config.get('path')
    if additional_path:
        path = os.path.join(path, additional_path)
    os.makedirs(path, exist_ok=True)
    # save fig
    if save_fig:
        fig_path = os.path.join(path, 'history.png')
        plt.savefig(fig_path)
        print(f"Save figure to path: {fig_path}")
    # save model
    model_path = os.path.join(path, 'model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Save model to path: {model_path}")
    # save config
    if config:
        config_path = os.path.join(path, 'config.txt')
        json_str = json.dumps(config)
        with open(config_path, 'w') as f:
            f.write(json_str)
            print(f"Save config to path: {config_path}")
    # save reward history
    if history:
        for name, data in history.items():
            filename = name + '.txt'
            history_path = os.path.join(path, filename)
            with open(history_path, 'w') as file:
                for r in data:
                    file.write(str(r) + '\n')
                print(f"Save {name} to path: {history_path}")
    return path


def draw_loss_accuracy_history(train_loss, test_loss, train_acc, test_acc, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_loss, label="train-loss")
    ax1.plot(test_loss, label="test-loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(train_acc, label="train-accuracy")
    ax2.plot(test_acc, label="test-accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "loss_accuracy.png"))
    plt.show()


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device
