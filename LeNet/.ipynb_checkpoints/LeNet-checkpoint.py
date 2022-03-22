# import 
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
input_size: 32x32
C1: 6 feature maps, kernel: 5x5
S2: kernel_size: 2x2, stride: 2x2
C3: 16 feature maps, kernel: 5x5
S4: kernel_size: 5x5
C5: 120 feature maps, kernel_size: 5x5
"""

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=0)
        self.s2 = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=0)
        self.s4 = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5), stride=(1,1), padding=0)
        
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        
    
    def forward(self, x):
        x = self.sigmoid(self.c1(x))
        x = self.s2(x)
        x = self.sigmoid(self.c3(x))
        x = self.s4(x)
        x = self.sigmoid(self.c5(x)) # [B, 120, 1, 1]
        x = x.reshape(x.shape[0], -1)
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
x = torch.randn(64, 1, 32, 32)
model = LeNet()
print(model(x).shape)