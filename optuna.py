from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import optuna
import joblib


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_mnist_loaders(train_batch_size, test_batch_size):
    """Get MNIST data loaders"""
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=train_batch_size, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader

# model.py

class Net(nn.Module):
    def __init__(self, activation):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.activation = activation 

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
# Trainer.py
def train(log_interval, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        # forward
        output = model(data)
        loss = F.nll_loss(output, target)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))  
            
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
        
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_accuracy
            
    
def train_mnist(trial): # optuna의 trial을 통해 space searching!!
      
  cfg = { 'device' : "cuda" if torch.cuda.is_available() else "cpu",
          'train_batch_size' : 64,
          'test_batch_size' : 1000,
          'seed' : 0,
          'log_interval' : 100,
          'save_model' : False,
          'activation': F.relu,
          ############### hyperparameter tuning ##################
          'n_epochs' : trial.suggest_int('n_epochs', 3, 5, 1),
          'lr' : trial.suggest_loguniform('lr', 1e-3, 1e-2),
          'momentum': trial.suggest_uniform('momentun', 0.4, 0.99),
          'optimizer': trial.suggest_categorical('optimizer', [optim.SGD, optim.RMSprop])
          #########################################################
          }

  torch.manual_seed(cfg['seed'])
  train_loader, test_loader = get_mnist_loaders(cfg['train_batch_size'], cfg['test_batch_size'])
  model = Net(cfg['activation']).to(device)
  optimizer = cfg['optimizer'](model.parameters(), lr=cfg['lr'])
  for epoch in range(1, cfg['n_epochs'] + 1):
      train(cfg['log_interval'], model, train_loader, optimizer, epoch)
      test_accuracy = test(model, test_loader)

  if cfg['save_model']:
      torch.save(model.state_dict(), "mnist_cnn.pt")
      
  return test_accuracy # monitor할 성능을 return 해야함!


if __name__ == '__main__':
    sampler = optuna.samplers.TPESampler() # 샘플러에 따라 (예시에서는 TPESampler) 정의 된 매개 변수 공간을 샘플링한다.
      
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(func=train_mnist, n_trials=20)
    joblib.dump(study, 'mnist_optuna.pkl')