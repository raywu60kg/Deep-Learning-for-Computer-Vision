# DenseNet
Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4700–4708).

## Achievements
dense block

## Model structure

```python
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    num_channels += num_convs * growth_rate
    
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

```
## Layer shapes
Conv2d output shape:         torch.Size([batch_size, 6, 28, 28])
Sigmoid output shape:        torch.Size([batch_size, 6, 28, 28])
AvgPool2d output shape:      torch.Size([batch_size, 6, 14, 14])
Conv2d output shape:         torch.Size([batch_size, 16, 10, 10])
Sigmoid output shape:        torch.Size([batch_size, 16, 10, 10])
AvgPool2d output shape:      torch.Size([batch_size, 16, 5, 5])
Flatten output shape:        torch.Size([batch_size, 400])
Linear output shape:         torch.Size([batch_size, 120])
Sigmoid output shape:        torch.Size([batch_size, 120])
Linear output shape:         torch.Size([batch_size, 84])
Sigmoid output shape:        torch.Size([batch_size, 84])
Linear output shape:         torch.Size([batch_size, num_label])

## Optimizer
Stochastic gradient descent