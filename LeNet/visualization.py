import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
from LeNet import LeNet

def imshow(img, prediction, gt):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    npimg = img.numpy().squeeze()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f'Ground True:{classes[gt]}\nLeNet Pred:{classes[prediction]}')
    plt.savefig('test.png')
    plt.close()

# cifar dataset
cifar = CIFAR10(root='../dataset', train=True,download=False, transform=transforms.ToTensor())
cifar_loader = DataLoader(cifar, batch_size=1, shuffle=True)


# get data
image, gt = next(iter(cifar_loader))

# LeNet prediction
checkpoint = torch.load('checkpoint/Lenet.pt')
model = LeNet(3, 10)
model.load_state_dict(checkpoint)
scores = model(image)
_, prediction = scores.max(1)


# Show Image using Lenet
prediction = prediction.item()
gt = gt.item()
imshow(image, prediction, gt)