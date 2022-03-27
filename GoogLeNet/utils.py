import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def imagenet_loader(path, mode, batch_size):
    # mode = ['train', 'val', 'test']
    path = os.path.join(path, mode)
        
    if mode == 'train': shuffle = True
    else: shuffle = False
    
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(
        path,
        transform
    )

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    

    return loader