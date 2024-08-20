import torch.nn as nn
import torch


class CnnNetwork1(nn.Module):

    def __init__(self, input_dim, action_dim):
        super(CnnNetwork1, self).__init__()
        activation = nn.LeakyReLU(negative_slope=0.01)
        # input shape (batch_size, 3, 96, 96)
        # cnn_out_width = ((input_width - kernel_size + 2 * padding) / stride) + 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=7, stride=1, padding=3),  # (batch_size, 12, 96, 96)
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 12, 48, 48)
            nn.Dropout(p=0.05)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),  # (batch_size, 24, 48, 48)
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 24, 24, 24)
            nn.Dropout(p=0.05)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (batch_size, 64, 24, 24)
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 64, 12, 12)
            nn.Dropout(p=0.05)
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64*12*12, 256)
        self.linear2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = nn.ReLU()(self.linear1(x))
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


class CnnNetwork3(nn.Module):
    def __init__(self, input_channel, action_size):
        super(CnnNetwork3, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(input_channel, 8, kernel_size=4, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.LeakyReLU(negative_slope=0.01),
        )  # output shape (256, 1, 1)
        self.flatten = nn.Flatten()
        self.l1 = nn.Sequential(nn.Linear(256, 100), nn.LeakyReLU(negative_slope=0.01), nn.Linear(100, action_size))
        # self.apply(self._weights_init)
        self.init_weights()

    # @staticmethod
    # def _weights_init(m):
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
    #         nn.init.constant_(m.bias, 0.1)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.cnn_base(x)
        x = self.flatten(x)
        x = self.l1(x)
        return x


class CnnNetwork4(nn.Module):

    def __init__(self, input_channel, action_size):
        super(CnnNetwork, self).__init__()
        # input shape (batch_size, <channel>, 96, 96)
        # cnn_out_size = ((input_width - kernel_size + 2 * padding) / stride) + 1
        # pooling_out_size = ((input_size - kernel_size) / stride) + 1 ,  if kernel_size=2 and stride=2, pooling_out_size = input_size / 2
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=8, stride=4, padding=4),  # (batch_size, 12, 96, 96)
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 12, 48, 48)
            nn.Dropout(p=0.1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=3, padding=2),  # (batch_size, 24, 48, 48)
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 24, 24, 24)
            nn.Dropout(p=0.03)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (batch_size, 48, 24, 24)
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 48, 12, 12)
            nn.Dropout(p=0.03)
        )
        self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(48 * 12 * 12, 256)
        self.linear1 = nn.Linear(64 * 9 * 9, 256)
        self.linear3 = nn.Linear(256, action_size)
        self.init_weights()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = torch.relu(self.linear1(x))
        # x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)