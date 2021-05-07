#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import models, datasets, transforms
from tqdm import tqdm

from safe_regions.region import MinMaxRegion
from safe_regions.layers import track_safe_region


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_loader(batch_size=128, num_workers=8):
    cifar = datasets.CIFAR10(
        '~/.pytorch/cifar10/',
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    )

    loader = data.DataLoader(
        cifar,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return loader

def get_model():
    resnet = models.resnet18()
    return resnet.to(DEVICE)

def train(
        lr=3e-4,
        num_epochs=10,
        track_safe_regions=False
):
    loader = get_loader()
    model = get_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    if track_safe_regions:
        track_safe_region(model, MinMaxRegion)

    for _ in range(num_epochs):
        train_loss = 0
        progress = tqdm(loader)
        for idx, (input_tensor, target) in enumerate(progress):
            input_tensor, target = input_tensor.to(DEVICE), target.to(DEVICE)
            output = model(input_tensor)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            progress.set_description(f'Loss {train_loss / (idx + 1):.3f}')
            progress.refresh()

        scheduler.step()

    return model

    
if __name__ == '__main__':
    train()
