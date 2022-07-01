import torch.nn as nn
import torch.nn.functional as F

class LeNet():
    def __init__(self, input_shape, num_label):
        self.channel = input_shape[0]
        self.num_label = num_label
        
    def get_model(self):
        return nn.Sequential(
            nn.Conv2d(self.channel, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, self.num_label))
