from models.alexnet import AlexNet
import torch
import numpy as np


def train(model, data)


if __name__=='__main__':
    
    # GPU
    GPU = 0 # 0~3
    device = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")
    
    # sample data
    imagenet_samples,imagenet_c,imagenet_x,imagenet_y=100,3,227,227
    x = np.random.rand(imagenet_samples,imagenet_c,imagenet_x,imagenet_y)
    x = torch.tensor(x).float().to(device)
    print(f'sample data shape is {x.shape}')
    num_classes = 1000
    
    model = AlexNet(num_classes).to(device)
    output = model(x)
    print(f'output shape is {output.shape}')