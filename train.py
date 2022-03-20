# Import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from customDataset import CatsAndDogsDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter
in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 1

# Load Data
dataset = CatsAndDogsDataset(csv_file='dataset/cats_dogs.csv', root_dir='dataset/cats_dogs_resized', transform=transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [8, 2])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Initialize network
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
for epoch in range(num_epochs):
    losses = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss)
        
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    print(f'Cost at epoch {epoch+1} is {sum(losses)/len(losses):.4f}')

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    # if loader.dataset.train:
    #     print("Checking accuracy on train data")
    # else:
    #     print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x) # [64, 10]
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        acc = float(num_correct)/float(num_samples)*100
        print(f'Got {num_correct}/{num_samples} with accuracy {acc:.2f}')

    model.train()
    return acc
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)