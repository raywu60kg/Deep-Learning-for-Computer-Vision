# VGG
Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

## Achievements
- Repeat VGG block

## Model structure
```python
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1

    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),

        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, num_label))
```
## Layer shapes
Conv2d output shape:         torch.Size([1, 96, 54, 54])
ReLU output shape:   torch.Size([batch_size, 96, 54, 54])
MaxPool2d output shape:      torch.Size([batch_size, 96, 26, 26])
Conv2d output shape:         torch.Size([batch_size, 256, 26, 26])
ReLU output shape:   torch.Size([batch_size, 256, 26, 26])
MaxPool2d output shape:      torch.Size([batch_size, 256, 12, 12])
Conv2d output shape:         torch.Size([batch_size, 384, 12, 12])
ReLU output shape:   torch.Size([batch_size, 384, 12, 12])
Conv2d output shape:         torch.Size([batch_size, 384, 12, 12])
ReLU output shape:   torch.Size([batch_size, 384, 12, 12])
Conv2d output shape:         torch.Size([batch_size, 256, 12, 12])
ReLU output shape:   torch.Size([batch_size, 256, 12, 12])
MaxPool2d output shape:      torch.Size([batch_size, 256, 5, 5])
Flatten output shape:        torch.Size([batch_size, 6400])
Linear output shape:         torch.Size([batch_size, 4096])
ReLU output shape:   torch.Size([batch_size, 4096])
Dropout output shape:        torch.Size([batch_size, 4096])
Linear output shape:         torch.Size([batch_size, 4096])
ReLU output shape:   torch.Size([1, 4096])
Dropout output shape:        torch.Size([batch_size, 4096])
Linear output shape:         torch.Size([batch_size, num_label])

## Optimizer
Stochastic gradient descent