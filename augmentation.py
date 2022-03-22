# Imports
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from customDataset import CatsAndDogsDataset

# Load data
my_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
])
dataset = CatsAndDogsDataset(csv_file = 'dataset/cats_dogs.csv',
                            root_dir = 'dataset/cats_dogs_resized',
                            transform = my_transform)
img_nums = 10
for img_num in range(img_nums):
    for img, label in dataset:
        print(label) 
        save_image(img, 'dataset/cats_dogs_aug/img'+str(img_num)+'.png')
        

