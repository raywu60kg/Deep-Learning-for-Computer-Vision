from typing import Tuple, Annotated
import torch
import torch.nn as nn
import math


class AlexNet(nn.Module):
    def __init__(self, input_shape: Annotated[Tuple[int], 3], num_label: int) -> None:
        super(AlexNet, self).__init__()
        channel = input_shape[0]
        # height = input_shape[1]
        # width = input_shape[2]

        self.conv1 = nn.Conv2d(
            in_channels=channel, out_channels=96, kernel_size=11, stride=4, padding=1
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, padding=2
        )
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, padding=1
        )
        self.conv5 = nn.Conv3d(
            in_channels=384, out_channels=256, kernel_size=3, padding=1
        )
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.LazyLinear(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_label)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.conv5(x)
        x = torch.relu(x)
        x = self.max_pool3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = torch.dropout(0.5)
        x = self.fc2(x)
        x = torch.relu(x)
        x = torch.dropout(0.5)
        x = self.fc3(x)
        return x
