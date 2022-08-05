from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize
from torch.utils.data import DataLoader
from typing import Tuple, Annotated, Union


class GetData:
    def get_training_data(self):
        raise NotImplementedError

    def get_testing_data(self):
        raise NotImplementedError


class GetFashionMnist(GetData):
    """
    image shape: (1,28,28)
    """
    def get_training_data(self, batch_size:int, resize: Union[None,Annotated[Tuple[int], 3]]=None):
        if resize is None:
            return DataLoader(
                datasets.FashionMNIST(
                    root="data", train=True, download=True, transform=ToTensor()
                ),
                batch_size=batch_size,
            )
        else:
            return DataLoader(
                datasets.FashionMNIST(
                    root="data", train=True, download=True, transform=Compose([Resize(resize),ToTensor()
                ])),
                batch_size=batch_size,
            )


    def get_testing_data(self, batch_size):

        return DataLoader(
            datasets.FashionMNIST(
                root="data", train=False, download=True, transform=ToTensor()
            ),
            batch_size=batch_size,
        )

class GetCifar10(GetData):
    """
    image shape: (1,28,28)
    """
    def get_training_data(self, batch_size):

        return DataLoader(
            datasets.CIFAR10(
                root="data", train=True, download=True, transform=ToTensor()
            ),
            batch_size=batch_size,
        )

    def get_testing_data(self, batch_size):

        return DataLoader(
            datasets.CIFAR10(
                root="data", train=False, download=True, transform=ToTensor()
            ),
            batch_size=batch_size,
        )