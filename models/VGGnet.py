import torch.nn as nn
import torch
from torchsummary import summary

# VGG type dict
# int: output channels after conv layer
# 'M': maxpooling layer
VGG_types = {
    'VGG11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'VGG13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M'],
    'VGG16':[64,64,'M',128,128,'M',256,256,'M',512,512,512,'M',512,512,512,'M'],
    'VGG19':[64,64,'M',128,1228,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
}

class VGGnet(nn.Module):
    def __init__(self, model, in_channels=3, num_classes=10, init_weights=True):
        super(VGGnet, self).__init__()
        
        self.in_channels = in_channels
        # model = 'VGG11', 'VGG13', 'VGG16', 'VGG19'
        self.conv_layers = self.create_conv_layers(VGG_types[model])
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(4096, num_classes)
        )
        
        if init_weights:
            self._initialize_weight()
            
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 512*7*7)
        x = self.classifier(x)
        return x
    
    def _initialize_weight(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1) # standard normal
                nn.init.constant_(m.bias, 0)
                
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                # conv layer: output
                out_channels = x
                layers+=[nn.Conv2d(in_channels,out_channels,
                                  kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(),]
                in_channels=x
                
            elif x == 'M':
                # Max pooling
                layers+=[nn.MaxPool2d(kernel_size=2, stride=2)]
                
        return nn.Sequential(*layers)
    
if __name__=='__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VGGnet('VGG16').to(device)
    summary(model, input_size=(3, 224, 224))
    
    