# 패키지
import torch
import torch.nn as nn

# model 구성(https://arxiv.org/abs/1409.1556)

architecture = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']
# flatten, 4096x4096x1000 linear layers
class VGG16(nn.Module):
    def __init__(self, in_channel=3, num_classes=1000):
        super(VGG16, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.conv_layers = self.create_conv_layers(architecture)
        self.fc_layers = nn.Sequential(
            nn.Linear(512*7*7, 4096), # 224/(2**5)=7(MaxPool 5개)
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, num_classes)
        )
        
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        return x
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channel = self.in_channel
        
        for x in architecture:
            if type(x) == int:
                out_channel = x
                layers += [
                    nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU()
                ]
                in_channel = out_channel
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        return nn.Sequential(*layers)
    
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG16().to(device)
    print(model)
    x = torch.randn((10,3,224,224)).to(device)
    print('output shape is',model(x).shape)