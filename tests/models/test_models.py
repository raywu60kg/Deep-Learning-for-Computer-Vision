from src.models.leNet import LeNet
import torch
import unittest

class TestModels(unittest.TestCase):

    def test_lenet_forword(self):
        model1 = LeNet(input_shape=(1,28,28), num_label=10)
        model2 = LeNet(input_shape=(1,32,16), num_label=10)
        test_image1 = torch.zeros(size=(1, 1, 28, 28), dtype=torch.float32)
        test_image2 = torch.zeros(size=(1, 1, 32, 16), dtype=torch.float32)
        output_image1 = model1.forward(test_image1)
        output_image2 = model2.forward(test_image2)
        assert output_image1.shape == torch.Size([1,10])
        assert output_image2.shape == torch.Size([1,10])