import torch
import torch.nn as nn
from dataloader.dataloader import GetFashionMnist
from models.cnn import LeNet, AlexNet, VGG, NiN
from training.training import SimpleTraining

model_name = "NiN"


def main():
    epochs = 3
    learning_rate = 0.05
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_loader = GetFashionMnist()
    training_data_loader = data_loader.get_training_data(batch_size=batch_size, resize=(224, 224))
    testing_data_loader = data_loader.get_testing_data(batch_size=batch_size, resize=(224, 224))
    training_method = SimpleTraining()
    if model_name == "LeNet":
        model = LeNet(input_shape=(1, 28, 28), num_label=10).to(device)
    elif model_name == "AlexNet":
        model = AlexNet(input_shape=(1, 224, 224), num_label=10).to(device)
    elif model_name == "VGG":
        model = VGG(input_shape=(1, 224, 224), num_label=10).to(device)
    elif model_name == "NiN":
        model = NiN(input_shape=(1, 224, 224), num_label=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        training_method.train(
            training_data_loader=training_data_loader,
            testing_data_loader=testing_data_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )


if __name__ == "__main__":
    main()
