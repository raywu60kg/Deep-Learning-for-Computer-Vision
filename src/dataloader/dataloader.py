from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class GetData:
    def get_training_data(self):
        raise NotImplementedError

    def get_testing_data(self):
        raise NotImplementedError


class GetFashionMnist(GetData):
    """
    image shape: (1,28,28)
    """
    def get_training_data(self, batch_size):

        return DataLoader(
            datasets.FashionMNIST(
                root="data", train=True, download=True, transform=ToTensor()
            ),
            batch_size=batch_size,
        )

    def get_testing_data(self, batch_size):

        return DataLoader(
            datasets.FashionMNIST(
                root="data", train=False, download=True, transform=ToTensor()
            ),
            batch_size=batch_size,
        )
