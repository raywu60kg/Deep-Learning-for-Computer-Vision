from typing import Tuple, Annotated
import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, input_shape: Annotated[Tuple[int], 3], num_label: int) -> None:
        super(LeNet, self).__init__()
        channel = input_shape[0]
        # height = input_shape[1]
        # width = input_shape[2]

        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=6, kernel_size=5, padding=2)
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
            in_channels=channel,
            out_channels=96,
            kernel_size=11,
            stride=4,
            padding=1,
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
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
        x = torch.dropout(x, p=0.5, train=True)
        x = self.fc2(x)
        x = torch.relu(x)
        x = torch.dropout(x, p=0.5, train=True)
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
        x = torch.dropout(x, p=0.5, train=True)
        x = self.fc2(x)
        x = torch.relu(x)
        x = torch.dropout(x, p=0.5, train=True)
        x = self.fc3(x)
        return x

    def get_vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)


class NiN(nn.Module):
    def __init__(self, input_shape: Annotated[Tuple[int], 3], num_label: int) -> None:
        super(NiN, self).__init__()
        channel = input_shape[0]
        self.nin_feature_block1 = self.get_nin_block(
            in_channels=channel,
            out_channels=96,
            kernel_size=11,
            strides=4,
            padding=0,
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.nin_feature_block2 = self.get_nin_block(
            in_channels=96,
            out_channels=256,
            kernel_size=5,
            strides=1,
            padding=2,
        )
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.nin_feature_block3 = self.get_nin_block(
            in_channels=256,
            out_channels=384,
            kernel_size=5,
            strides=1,
            padding=2,
        )
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.nin_fc_block = self.get_nin_block(
            in_channels=384,
            out_channels=num_label,
            kernel_size=3,
            strides=1,
            padding=1,
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.nin_feature_block1(x)
        x = self.max_pool1(x)
        x = self.nin_feature_block2(x)
        x = self.max_pool2(x)
        x = self.nin_feature_block3(x)
        x = self.max_pool3(x)
        x = torch.dropout(x, p=0.5, train=True)
        x = self.nin_fc_block(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return x

    def get_nin_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        strides: int,
        padding: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
        )


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels1: int,
        out_channels2: Annotated[Tuple[int], 2],
        out_channels3: Annotated[Tuple[int], 2],
        out_channels4: int,
    ):
        super(InceptionBlock, self).__init__()
        self.p1_1 = nn.Conv2d(in_channels, out_channels1, kernel_size=1)

        self.p2_1 = nn.Conv2d(in_channels, out_channels2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(out_channels2[0], out_channels2[1], kernel_size=3, padding=1)

        self.p3_1 = nn.Conv2d(in_channels, out_channels3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(out_channels3[0], out_channels3[1], kernel_size=5, padding=2)

        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, out_channels4, kernel_size=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        p1 = torch.relu(self.p1_1(x))
        p2 = torch.relu(self.p2_2(torch.relu(self.p2_1(x))))
        p3 = torch.relu(self.p3_2(torch.relu(self.p3_1(x))))
        p4 = torch.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


class InceptionNet(nn.Module):
    def __init__(self, input_shape: Annotated[Tuple[int], 3], num_label: int) -> None:
        super(InceptionNet, self).__init__()
        channel = input_shape[0]
        self.block1 = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block3 = nn.Sequential(
            InceptionBlock(192, 64, (96, 128), (16, 32), 32),
            InceptionBlock(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block4 = nn.Sequential(
            InceptionBlock(480, 192, (96, 208), (16, 48), 64),
            InceptionBlock(512, 160, (112, 224), (24, 64), 64),
            InceptionBlock(512, 128, (128, 256), (24, 64), 64),
            InceptionBlock(512, 112, (144, 288), (32, 64), 64),
            InceptionBlock(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block5 = nn.Sequential(
            InceptionBlock(832, 256, (160, 320), (32, 128), 128),
            InceptionBlock(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(1024, num_label)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            num_channels,
            kernel_size=3,
            padding=1,
            stride=strides,
        )
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = torch.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return torch.relu(Y)


class ResNet(nn.Module):
    def __init__(self, input_shape: Annotated[Tuple[int], 3], num_label: int) -> None:
        super(ResNet, self).__init__()
        channel = input_shape[0]
        self.b1 = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.b2 = nn.Sequential(
            *self.get_resnet_block(
                input_channels=64,
                num_channels=64,
                num_residuals=2,
                use_1x1conv_on_first_block=False,
            )
        )
        self.b3 = nn.Sequential(
            *self.get_resnet_block(
                input_channels=64,
                num_channels=128,
                num_residuals=2,
                use_1x1conv_on_first_block=True,
            )
        )
        self.b4 = nn.Sequential(
            *self.get_resnet_block(
                input_channels=128,
                num_channels=256,
                num_residuals=2,
                use_1x1conv_on_first_block=True,
            )
        )
        self.b5 = nn.Sequential(
            *self.get_resnet_block(
                input_channels=256,
                num_channels=512,
                num_residuals=2,
                use_1x1conv_on_first_block=True,
            )
        )
        self.fc = nn.Linear(512, num_label)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = torch._adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def get_resnet_block(
        self,
        input_channels,
        num_channels,
        num_residuals,
        use_1x1conv_on_first_block,
    ):
        blk = []
        for i in range(num_residuals):
            if i == 0 and use_1x1conv_on_first_block:
                blk.append(
                    ResidualBlock(
                        input_channels,
                        num_channels,
                        use_1x1conv=True,
                        strides=2,
                    )
                )
            else:
                blk.append(ResidualBlock(num_channels, num_channels))
        return blk


class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super().__init__()
        layer = []
        for i in range(num_convs):
            layer.append(self.conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def conv_block(self, input_channels, num_channels):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X


class DenseNet(nn.Module):
    def __init__(self, input_shape: Annotated[Tuple[int], 3], num_label: int) -> None:
        super().__init__()
        channel = input_shape[0]
        self.b1 = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.blks = []
        num_channels, growth_rate = 64, 32
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            self.blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            num_channels += num_convs * growth_rate

            if i != len(num_convs_in_dense_blocks) - 1:
                self.blks.append(self.transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2

        self.bn = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(num_channels, num_label)

    def forward(self, x):
        x = self.b1(x)
        for blk in self.blks:
            x = blk(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = torch._adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def transition_block(self, input_channels: int, num_channels: int):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
