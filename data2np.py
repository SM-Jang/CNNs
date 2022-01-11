import pickle
import numpy as np
    
import torchvision.transforms as transforms
from torchvision import datasets
import os

name = 'cifar'

def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

# cifar
if name == 'cifar':
    train_path = ['../dataset/cifar-10-batches-py/data_batch_{}'.format(i) for i in range(1,6)]
    test_path = '../dataset/cifar-10-batches-py/test_batch'

    train_x = []
    train_y = []
    for i in range(5):
        data = unpickle(train_path[i])['data']
        labels = unpickle(train_path[i])['labels']
        train_x.append(data)
        train_y.append(labels)


    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    test_x = unpickle(test_path)['data']
    test_y = np.array(unpickle(test_path)['labels'])

    np.save('../dataset/cifar-10-batches-py/train_data.npy', train_x)
    np.save('../dataset/cifar-10-batches-py/train_labels.npy', train_y)
    np.save('../dataset/cifar-10-batches-py/test_data.npy', test_x)
    np.save('../dataset/cifar-10-batches-py/test_labels.npy', test_y)

if name == 'stl':


    # specify a data path
    path2data = './git/dataset/'

    # if not exists the path, make the path
    if not os.path.exists(path2data):
        os.mkdir(path2data)

    # load STL10 train dataset, and check
    data_transformer = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.STL10(path2data, split='train', download=True, transform=data_transformer)
    print(train_ds.data.shape)
    
    # load STL10 test dataset
    test0_ds = datasets.STL10(path2data, split='test', download=True, transform=data_transformer)
    print(test0_ds.data.shape)
