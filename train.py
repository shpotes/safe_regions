#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import models, datasets, transforms
from tqdm import tqdm

from safe_regions import hook, region


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


def train():
    loader = get_loader()
    model = get_model()
    
    hook.track_safe_region(model, region.MinMaxRegion)

    for input_tensor, _ in tqdm(loader):
        model(input_tensor.to(DEVICE))
    

    for layer in model.children():
        if isinstance(layer, nn.ReLU):
            reg = layer.__region
            
    print(reg._min)

    
if __name__ == '__main__':
    train()
