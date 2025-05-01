import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from cnn import Net

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    # TensorBoard writer to keep tracking of learning curves
    writer = SummaryWriter('bim/dogs-vs-cats-logs')

    # Image transformations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    batch_size = 10
    data_dir = 'archive'

    # Load datasets
    train_dataset = torchvision.datasets.ImageFolder(data_dir + '/Train', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataset = torchvision.datasets.ImageFolder(data_dir + '/Val', transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    classes = {0: 'Meningioma', 1: 'No Tumor'}

    device = torch.device('cpu')
    print(device)

    # Show some images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))
    print(' '.join(f'{classes[labels[j].item()]}' for j in range(batch_size)))

if __name__ == '__main__':
    main()
