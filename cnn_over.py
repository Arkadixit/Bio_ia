import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.batch_norm = nn.BatchNorm2d(3)
        self.pool = nn.MaxPool2d(2,2)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding=1) 
        self.conv4 = nn.Conv2d(128,256,kernel_size=3,padding=1)

        self.fc1 = nn.Linear(256*14*14, 512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128, 4)
        self.lr = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.batch_norm(x)
        x = self.pool(self.lr(self.conv1(x)))
        x = self.pool(self.lr(self.conv2(x)))
        x = self.pool(self.lr(self.conv3(x)))
        x = self.pool(self.lr(self.conv4(x)))
        x = self.dropout(x)
        x = torch.flatten(x,1)
        x = self.lr(self.fc1(x))
        x = self.lr(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x