# AlexNet
Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems 

## Achievements
- GPU accelerator
- Relu activation function
- Max pooling
- Dropout layer
- Image data augmentation
- Much bigger number of parameters compare to LeNet

## Model structure
```
net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, num_label))
```
## Layer shapes
Conv2d output shape:         torch.Size([batch_size, 96, 54, 54])
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
ReLU output shape:   torch.Size([batch_size, 4096])
Dropout output shape:        torch.Size([batch_size, 4096])
Linear output shape:         torch.Size([batch_size, num_label])

## input image
3 * 224 * 224