import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 64 output channels/feature maps, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool1= nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32,  64, 3)
        self.pool2= nn.MaxPool2d(2,2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,  128, 3)
        self.pool3 = nn.MaxPool2d(2,2)
        self.bn3 = nn.BatchNorm2d(128)

        self.drop3 = nn.Dropout2d(0.2)

        self.conv4 = nn.Conv2d(128,  256, 3)
        self.pool4 = nn.MaxPool2d(2,2)
        self.bn4 = nn.BatchNorm2d(256)

        self.drop4 = nn.Dropout2d(0.1)
        self.fc1= nn.Linear(36864,1024)
        self.drop5=nn.Dropout(0.2)
        self.fc2= nn.Linear(1024,512)
        self.fc3= nn.Linear(512,136)

        
    def forward(self, x):
        x= F.selu(self.bn1(self.pool1(self.conv1(x))))
        x= F.selu(self.bn2(self.pool2(self.conv2(x)))) 
        x= F.selu(self.bn3(self.pool3(self.conv3(x))))
        x= self.drop3(x)
        x= F.selu(self.bn4(self.pool4(self.conv4(x))))
        x= self.drop4(x)
        x= x.view(x.size(0), -1)
        x= F.selu(self.fc1(x))
        x= self.drop5(x)
        x= self.fc2(x)
        x= self.fc3(x)
        return x