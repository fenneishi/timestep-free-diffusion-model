import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

channels = 1
image_size = 28

training_data = datasets.FashionMNIST(
    root=".data",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
)

test_data = datasets.FashionMNIST(
    root=".data",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
)


def build_data(batch_size, train=True):
    if train:
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(),
                                      pin_memory=True, drop_last=True)
        return train_dataloader
    else:
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(),
                                     pin_memory=True, drop_last=True)
        return test_dataloader
