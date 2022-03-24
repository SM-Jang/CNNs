import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from LeNet import LeNet
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameter
num_epochs    = 5
batch_size    = 64
learning_rate = 1e-3


# dataset & loader
train_dataset = CIFAR10('../dataset/', train=True, download=False, transform=transforms.ToTensor())
test_dataset  = CIFAR10('../dataset/', train=False, download=False, transform=transforms.ToTensor())
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# model, loss, optimizer
in_channel  = 3
num_classes = 10
model     = LeNet(in_channel, num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def accuracy(scores, label):
    _, prediction = scores.max(1)
    num_correct = (prediction == label).sum()
    acc = float(num_correct)/batch_size*100
    return acc, num_correct


def train():
    model.train()
    for epoch in range(num_epochs):
        # loss & accuracy
        losses = []
        num_corrects = []
        
        # training
        for idx,(image, label) in enumerate(train_loader):
            # from cpu to gpu
            image = image.to(device)
            label = label.to(device)
            
            # forward
            scores = model(image)
            loss = criterion(scores, label)
            losses.append(loss.item())
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # accuracy
            acc, num_correct = accuracy(scores, label)
            num_corrects.append(num_correct.item())
            
            if (idx+1) % 100 == 0:    
                print("Training process [{}/{}] | Loss: {:.4f} | Accuracy: {:.2f}".format(idx+1, train_loader.__len__(), loss.item(), acc))
            
        print('Epoch {}/{}\t Loss {:.4f}\t Accuracy {:.2f}'.format(epoch+1,num_epochs,sum(losses)/len(losses),sum(num_corrects)/len(num_corrects)), end='\n\n')


        
def test():
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            
            scores = model(image)
            
            _, predictions = scores.max(1)
            num_correct += (predictions == label).sum()
            num_samples += predictions.size(0)
        acc = float(num_correct)/float(num_samples)*100
        print(f'Test Accuracy is {acc:.2f}')
        
    return acc
        
if __name__ == '__main__':
    # train
    train()
    test()