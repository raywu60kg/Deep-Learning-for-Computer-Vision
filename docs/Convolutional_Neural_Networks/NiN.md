# NiN
Lin, M., Chen, Q., & Yan, S. (2013). Network in network. arXiv preprint arXiv:1312.4400.

## Innovation
- NiN block
- Global average pooling layer
- No fully connected layer

## Model structure
```python
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
  
    nin_block(384, num_label, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten())
```
## Layer shapes
Sequential output shape:     torch.Size([batch_size, 96, 54, 54])
MaxPool2d output shape:      torch.Size([batch_size, 96, 26, 26])
Sequential output shape:     torch.Size([batch_size, 256, 26, 26])
MaxPool2d output shape:      torch.Size([batch_size, 256, 12, 12])
Sequential output shape:     torch.Size([batch_size, 384, 12, 12])
MaxPool2d output shape:      torch.Size([batch_size, 384, 5, 5])
Dropout output shape:        torch.Size([batch_size, 384, 5, 5])
Sequential output shape:     torch.Size([batch_size, 10, 5, 5])
AdaptiveAvgPool2d output shape:      torch.Size([batch_size, 10, 1, 1])
Flatten output shape:        torch.Size([batch_size, num_label])

## Optimizer
Stochastic gradient descent