# LeNet
LeCun, Y., Bottou, L., Bengio, Y., Haffner, P., & others. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278–2324.

## Achievements
- Convolutional layer
- Pooling layer

## Model structure

```python
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, num_label))
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
