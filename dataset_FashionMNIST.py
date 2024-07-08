import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import platform

channels = 1
image_size = 28


def normalize(img):
    return (img * 2.) - 1.


def denormalize(img):
    return (img + 1.) / 2.


training_data = datasets.FashionMNIST(
    root=".data",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(normalize)
    ])
)

test_data = datasets.FashionMNIST(
    root=".data",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(normalize)
    ])
)


def build_data(batch_size, train=True):
    num_workers = 0 if platform.system() == 'Windows' else os.cpu_count()
    print(f"dataloader num_workers: {num_workers}")
    if train:
        train_dataloader = DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        return train_dataloader
    else:
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        return test_dataloader
