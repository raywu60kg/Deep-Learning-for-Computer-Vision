from typing import Tuple, Annotated
import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, input_shape: Annotated[Tuple[int], 3], num_label: int) -> None:
        super(LeNet, self).__init__()
        channel = input_shape[0]
        # height = input_shape[1]
        # width = input_shape[2]

        self.conv1 = nn.Conv2d(
            in_channels=channel, out_channels=6, kernel_size=5, padding=2
        )
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_label)

    def forward(self, x: torch.tensor) -> torch.tensor:
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
        self.conv5 = nn.Conv2d(
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
        x = torch.dropout(x,p=0.5,train=True)
        x = self.fc2(x)
        x = torch.relu(x)
        x = torch.dropout(x,p=0.5,train=True)
        x = self.fc3(x)
        return x


class VGG(nn.Module):
    def __init__(self, input_shape: Annotated[Tuple[int], 3], num_label: int) -> None:
        super(VGG, self).__init__()
        channel = input_shape[0]
        # height = input_shape[1]
        # width = input_shape[2]
        in_channel = channel
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        conv_blks = []
        for num_convs, out_channels in conv_arch:
            conv_blks.append(
                self.get_vgg_block(
                    num_convs=num_convs,
                    in_channels=in_channel,
                    out_channels=out_channels,
                )
            )
            in_channel = out_channels
        self.features = nn.Sequential(*conv_blks)
        self.fc1 = nn.LazyLinear(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_label)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = torch.dropout(x,p=0.5,train=True)
        x = self.fc2(x)
        x = torch.relu(x)
        x = torch.dropout(x,p=0.5,train=True)
        x = self.fc3(x)
        return x

    def get_vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

class NiN(nn.Module):
    def __init__(self, input_shape: Annotated[Tuple[int], 3], num_label: int) -> None:
        super(VGG, self).__init__()
        channel = input_shape[0]
    
    def get_nin_black(self):
        pass