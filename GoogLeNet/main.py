# 패키지
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from GoogLeNet import GoogLeNet
from dataset import 
from utils import imagenet_loader

# GPU
GPU = 0
device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')


# hyperparameter
batch_size = 32
num_epochs = 100
learning_rate = 1e-3


# VGG16
model = GoogLeNet().to(device)


# loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# train VGG16
def train():
    mode = 'train'
    train_loader = imagenet_loader('../dataset/imagenet/', mode, batch_size)
    model.train()
    for epoch in range(num_epochs):
        # metrics
        losses = []
        num_corrects = []
        
        # training
        for idx, (images, labels) in enumerate(train_loader):
            # from cpu to gpu
            images = image.to(device)
            labels = image.to(device)
            
            # forward
            scores = model(images)
            loss = criterion(scores, labels)
            losses.append(loss.item())
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # accuracy
            _, predictions = scores.max(1)
            num_corrct = (predictions == label).sum()
            num_corrects.append(num_correct)
            running_acc = float(num_correct)/batch_size*100
            
            # running states
            if (idx+1)%100==0:
                print("Training process [{}/{}] | loss: {:4f} | accuracy: {:.2f}".format(idx+1, train_loader.__len__(), loss.item(), running_acc))
                
        print('Epoch [{}/{}]\t Loss {:.4f}\t Accuracy {:.2f}'.format(epoch+1, num_epochs, sum(losses)/len(losses), sum(num_corrects)/len(num_corrects)))

        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoint/GoogLeNet{epoch}.pth')
              
def test():
    num_correct=0
    num_samples=0
    mode = 'test'
    test_loader = imagenet_loader('../dataset/imagenet/', mode, batch_size)
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            scores = model(images)
            
            _, predictions = scores.max(1)
            
            num_corrects = (predictions == labels).sum()
            num_samples += predictions.size(0)
            
        acc = float(num_correct)/num_samples*100
        print(f'Test accuracy is {acc:.2f}')
        
    return acc

if __name__ == '__main__':
    train()
    test()
                