import torch.nn as nn


class CnnNetwork1(nn.Module):

    def __init__(self, action_size):
        super(CnnNetwork1, self).__init__()
        # input shape (batch_size, 3, 96, 96)
        # cnn_out_width = ((input_width - kernel_size + 2 * padding) / stride) + 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=7, stride=1, padding=3),  # (batch_size, 12, 96, 96)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 12, 48, 48)
            nn.Dropout(p=0.05)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=5, stride=1, padding=2),  # (batch_size, 24, 48, 48)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 24, 24, 24)
            nn.Dropout(p=0.05)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (batch_size, 64, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 64, 12, 12)
            nn.Dropout(p=0.05)
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64*12*12, 256)
        self.linear2 = nn.Linear(256, action_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class CnnNetwork2(nn.Module):

    def __init__(self, input_channel, action_size):
        super(CnnNetwork2, self).__init__()
        # input shape (batch_size, 3, 96, 96)
        # cnn_out_width = ((input_width - kernel_size + 2 * padding) / stride) + 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, 8, kernel_size=8, stride=3, padding=1),  # (batch_size, 8, 31, 31)
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 8, 48, 48)
            nn.Dropout(p=0.05)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # (batch_size, 16, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 16, 8, 8)
            nn.Dropout(p=0.05)
        )

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(16*8*8, 128)
        self.linear2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

