from torchvision import datasets
from torchvision.transforms import ToTensor


class GetData:
    def get_training_data(self):
        raise NotImplementedError

    def get_testing_data(self):
        raise NotImplementedError


class GetFashionMnist(GetData):
    def get_testing_data(self):

        return datasets.FashionMNIST(
            root="data", train=True, download=True, transform=ToTensor()
        )

    def get_testing_data(self):

        return datasets.FashionMNIST(
            root="data", train=False, download=True, transform=ToTensor()
        )
