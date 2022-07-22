from src.dataloader.dataloader import GetFashionMnist
import torch
import unittest

class TestDataLoader(unittest.TestCase):
    def test_fashion_mnist(self) -> None:
        get_data = GetFashionMnist()
        training_data_loader = get_data.get_training_data(batch_size=1)
        print(type(training_data_loader))
        print(training_data_loader[0].shape)
        assert 1 == 2
