import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        """
        layers: 5 * Conv + 3 * FC
        output: number of classes
        """
        super(AlexNet, self).__init__()

        h1, h2, h3, h4, h5 = 96, 256, 384, 384, 256
        k1, k2, k3 = 11, 5, 3
        
        k, n, alpha, beta = 2, 5, 0.0001, 0.75
        z, s = 3, 2
        self.LRN = nn.LocalResponseNorm(size=n, alpha=alpha, beta=beta, k=k)
        self.maxpool = nn.MaxPool2d(kernel_size=z, stride=s)
        
        self.dropout = nn.Dropout(0.5, inplace=True)
        
        # input 3x227x227
        self.conv = nn.Sequential(
            # First 
            nn.Conv2d(3, h1, kernel_size=k1, stride=4), # 96x55x55
            nn.ReLU(),
            self.LRN,
            self.maxpool, # 96x27x27
            
            # Second 
            nn.Conv2d(h1, h2, kernel_size=k2, stride=1, padding=2), # 256x27x27
            nn.ReLU(),
            self.LRN,
            self.maxpool, # 256x13x13
            
            # Third 
            nn.Conv2d(h2, h3, kernel_size=k3, stride=1,padding=1), # 384x13x13
            nn.ReLU(),
            
            # Forth 
            nn.Conv2d(h3, h4, kernel_size=k3, stride=1,padding=1), # 384x13x13
            nn.ReLU(),
            
            # Fifth
            nn.Conv2d(h4, h5, kernel_size=k3, stride=1,padding=1), # 256x13x13
            nn.ReLU(),
            self.maxpool # 256x6x6
        )
        self.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(h5*6*6, 4096),
            nn.ReLU(),
            
            self.dropout,
            nn.Linear(4096, 4096),
            nn.ReLU(),
            
            nn.Linear(4096, num_classes),
        )
        
        self.init_weight()
        
    def init_weight(self):
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
                
        nn.init.constant_(self.conv[4].bias, 1) # conv2
        nn.init.constant_(self.conv[10].bias, 1) # conv4
        nn.init.constant_(self.conv[12].bias, 1) # conv5
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 256*6*6)
        x = self.classifier(x)
        return x
