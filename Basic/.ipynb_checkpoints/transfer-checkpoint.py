# Import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5

import sys


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# Load pretrain model & modify it
model = torchvision.models.vgg16(pretrained=True)
for param in model.parameters():
    param.reqires_grad = False

model.avgpool = Identity()
model.classifier = nn.Sequential(
    nn.Linear(512,100),
    nn.ReLU(),
    nn.Linear(100,10)
)
model.to(device)

# Load data
train_dataset = datasets.CIFAR10(root='dataset/', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
for epoch in range(num_epochs):
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        # forward
        scores = model(data)
        loss = criterion(scores, target)
        losses.append(loss)
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()
    print(f"Cost at epoch {epoch+1} is {sum(losses)/len(losses):.5f}")


# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on train data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)  # [64, 10]
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        acc = float(num_correct) / float(num_samples) * 100
        print(f'Got {num_correct}/{num_samples} with accuracy {acc:.2f}')

    model.train()
    return acc



check_accuracy(train_loader, model)