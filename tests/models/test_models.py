from src.models.cnn import LeNet, AlexNet, VGG, NiN
import torch
import unittest


class TestModels(unittest.TestCase):
    def test_lenet_forword(self) -> None:
        model1 = LeNet(input_shape=(1, 28, 28), num_label=10)
        model2 = LeNet(input_shape=(1, 32, 16), num_label=10)
        model3 = LeNet(input_shape=(3, 224, 224), num_label=10)
        test_image1 = torch.zeros(size=(1, 1, 28, 28), dtype=torch.float32)
        test_image2 = torch.zeros(size=(1, 1, 32, 16), dtype=torch.float32)
        test_image3 = torch.zeros(size=(1, 3, 224, 224), dtype=torch.float32)
        output_image1 = model1.forward(test_image1)
        output_image2 = model2.forward(test_image2)
        output_image3 = model3.forward(test_image3)
        print(type(output_image1))
        print(type(test_image1))
        assert output_image1.shape == torch.Size([1, 10])
        assert output_image2.shape == torch.Size([1, 10])
        assert output_image3.shape == torch.Size([1, 10])

    def test_alexnet_forword(self) -> None:
        model1 = AlexNet(input_shape=(1, 128, 128), num_label=10)
        model2 = AlexNet(input_shape=(1, 224, 128), num_label=10)
        model3 = AlexNet(input_shape=(3, 224, 224), num_label=10)
        test_image1 = torch.zeros(size=(1, 1, 128, 128), dtype=torch.float32)
        test_image2 = torch.zeros(size=(1, 1, 224, 128), dtype=torch.float32)
        test_image3 = torch.zeros(size=(1, 3, 224, 224), dtype=torch.float32)
        output_image1 = model1.forward(test_image1)
        output_image2 = model2.forward(test_image2)
        output_image3 = model3.forward(test_image3)
        # print(type(output_image1))
        # print(type(test_image1))
        assert output_image1.shape == torch.Size([1, 10])
        assert output_image2.shape == torch.Size([1, 10])
        assert output_image3.shape == torch.Size([1, 10])

    def test_vgg_forword(self) -> None:
        model1 = VGG(input_shape=(1, 128, 128), num_label=10)
        model2 = VGG(input_shape=(1, 224, 128), num_label=10)
        model3 = VGG(input_shape=(3, 224, 224), num_label=10)
        test_image1 = torch.zeros(size=(1, 1, 128, 128), dtype=torch.float32)
        test_image2 = torch.zeros(size=(1, 1, 224, 128), dtype=torch.float32)
        test_image3 = torch.zeros(size=(1, 3, 224, 224), dtype=torch.float32)
        output_image1 = model1.forward(test_image1)
        output_image2 = model2.forward(test_image2)
        output_image3 = model3.forward(test_image3)
        print(type(output_image1))
        print(type(test_image1))
        assert output_image1.shape == torch.Size([1, 10])
        assert output_image2.shape == torch.Size([1, 10])
        assert output_image3.shape == torch.Size([1, 10])

    def test_nin_forword(self) -> None:
        model1 = NiN(input_shape=(1, 128, 128), num_label=10)
        model2 = NiN(input_shape=(1, 224, 128), num_label=10)
        model3 = NiN(input_shape=(3, 224, 224), num_label=10)
        test_image1 = torch.zeros(size=(1, 1, 128, 128), dtype=torch.float32)
        test_image2 = torch.zeros(size=(1, 1, 224, 128), dtype=torch.float32)
        test_image3 = torch.zeros(size=(1, 3, 224, 224), dtype=torch.float32)
        output_image1 = model1.forward(test_image1)
        output_image2 = model2.forward(test_image2)
        output_image3 = model3.forward(test_image3)
        print(type(output_image1))
        print(type(test_image1))
        assert output_image1.shape == torch.Size([1, 10])
        assert output_image2.shape == torch.Size([1, 10])
        assert output_image3.shape == torch.Size([1, 10])
