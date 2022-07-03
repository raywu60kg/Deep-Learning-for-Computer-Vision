import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, input_shape, num_label):
        super(LeNet, self).__init__()
        channel = input_shape[0]
        height = input_shape[1]
        weight = input_shape[2]
        self.conv1 = nn.Conv2d(
            in_channels=channel, out_channels=6, kernel_size=5, padding=2
        )
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(int(16 * (height / 4 - 2) * (weight / 4 - 2)), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_label)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = self.avg_pool1(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = self.avg_pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        return x
