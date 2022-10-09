from src.dataloader.dataloader import GetFashionMnist
import torch
import unittest


class TestDataLoader(unittest.TestCase):
    def test_fashion_mnist(self) -> None:
        get_data = GetFashionMnist()
        training_data_loader = get_data.get_training_data(batch_size=1)
        print(type(training_data_loader))
        for X, y in training_data_loader:
            # print(y.size())
            # print(X.size())
            break
        # print(training_data_loader[0])
        assert X.size() == torch.Size([1, 1, 28, 28])

    def test_fashion_mnist_resize(self) -> None:
        get_data = GetFashionMnist()
        training_data_loader = get_data.get_training_data(batch_size=1, resize=(224, 224))
        print(type(training_data_loader))
        for X, y in training_data_loader:
            # print(y.size())
            # print(X.size())
            break
        # print(training_data_loader[0])
        assert X.size() == torch.Size([1, 1, 224, 224])
