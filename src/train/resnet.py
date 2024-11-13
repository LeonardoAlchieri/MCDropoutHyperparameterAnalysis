import torchvision

# import dataloader
import torch
from torch.utils.data import DataLoader
from src.train import calculate_outlier_info, calculate_anonmaly_info
from uncertainty_baselines.models import resnet50_dropout_torch


def train(
    num_epochs: int,
    dropout_rate: float,
    learning_rate: float,
    batch_size: int,
    print_info_schedule: int = 1,
    dataset_name: str = "CIFAR100",
):

    if dataset_name == "CIFAR100":
        train_set = torchvision.datasets.CIFAR100(
            root="./cifar100.nosync", train=True, download=True
        )
        test_set = torchvision.datasets.CIFAR100(
            root="./cifar100.nosync", train=False, download=True
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented. Please use CIFAR100.")

    # load resnet 18 pre-trained on imagenet
    model = resnet50_dropout_torch(
        pretrained=True, num_classes=100, dropout_rate=dropout_rate
    )

    # Finetune the model on the train_set
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % print_info_schedule == print_info_schedule-1:
                print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss / 200}")
                running_loss = 0.0

    # Perform validation on the test_set
    # model.eval()
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation accuracy: {accuracy}%")

    ...
