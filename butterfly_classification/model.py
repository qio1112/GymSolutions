import torch
import torch.nn as nn
import torch.optim as optim


class CNNModel(nn.Module):

    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        # out_width = ((input_width - kernel_size + 2 * padding) / stride) + 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # (batch, 3, 224, 224) -> (batch, 16, 224, 224)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, 16, 224, 224) -> (batch, 16, 112, 112)
            nn.Dropout(p=0.95)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # (batch, 16, 112, 112) -> (batch, 32, 112, 112)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, 32, 112, 112) -> (batch, 32, 56, 56)
            nn.Dropout(p=0.95)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (batch, 32, 56, 56) -> (batch, 64, 56, 56)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, 64, 56, 56) -> (batch, 64, 28, 28)
            nn.Dropout(p=0.95)
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64*28*28, 256)
        self.linear2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


