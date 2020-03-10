#!/usr/bin/env python3

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy


# plt.ion()   # interactive mode


class Network(nn.Module):
    def __init__(self,num_classes):
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.fc1 = nn.Linear(3072,num_classes)
        self.pool = nn.MaxPool2d((3,1))
        self.relu = nn.ReLU()

    def forward(self,x):
        #convlution 1 to 32 feature maps, 3x3 kernel
        x = self.conv1(x)
        # print('conv2d '+str(x.shape))
        #relu
        x = self.relu(x)
        # print(x.shape)
        #max pooling 3x1
        x = self.pool(x)
        x = torch.flatten(x,1)
        #fully connected layer
        x = self.fc1(x)
        # print(x)
        #softmax
        return F.softmax(x,dim=1) #dim=1 refers to along the rows

class Network_enhanced(nn.Module):
    def __init__(self,num_classes):
        super(Network_enhanced,self).__init__()
        self.conv1_1 = nn.Conv2d(1,32,(5,3))#,padding=(4,2),stride=2)
        self.conv2_1 = nn.Conv2d(32,64,(5,3))#,padding=(4,2),stride=2) #changed kernel from 5,3
        # self.conv3 = nn.Conv2d(64,128,(5,3),padding=(4,2),stride=2)
        self.fc1 = nn.Linear(1024,512) #changed 768 from 1024
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,num_classes)
        self.pool = nn.MaxPool2d((3,1))
        self.BN1 = nn.BatchNorm2d(32)
        self.BN2 = nn.BatchNorm2d(64)
        self.BN3 = nn.BatchNorm2d(500)
        self.prelu = nn.PReLU()
        self.drop = nn.Dropout2d()

    def forward(self,x):
        #convlution 1 to 32 feature maps, 3x5 kernel
        x = self.conv1_1(x)
        #batch normalization
        x = self.BN1(x)
        #prelu
        x = self.prelu(x)
        #max pooling 3x1
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = self.conv2_1(x)
        # print(x.shape)

        #batch normalization
        x = self.BN2(x)
        #prelu
        x = self.prelu(x)
        x = self.drop(x)
        #max pooling 3x1
        x = self.pool(x)
        # x = self.conv3(x)
        # #batch normalization
        # x = self.BN3(x)
        # #prelu
        # x = self.prelu(x)
        # # x = self.drop(x)
        #
        # x = self.pool(x)

        x = torch.flatten(x,1)
        #fully connected layers
        x = self.fc1(x)
        # # #prelu
        x = self.prelu(x)
        x = self.fc2(x)
        x = self.prelu(x)
        x = self.fc3(x)
        x = self.prelu(x)

        #softmax
        return F.softmax(x,dim=1)


class Network_XL(nn.Module):
    def __init__(self,num_classes):
        super(Network_XL,self).__init__()
        self.conv1_1 = nn.Conv2d(1,16,(5,3),padding=(4,2))#,stride=2)
        self.conv2_1 = nn.Conv2d(16,32,(5,3),padding=(4,2))#,stride=2) #changed kernel from 5,3
        self.conv3 = nn.Conv2d(32,64,(5,3),padding=(4,2))#,stride=2)
        self.fc1 = nn.Linear(2688,512) #changed 768 from 1024
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,num_classes)
        self.pool = nn.MaxPool2d((3,1))
        self.BN1 = nn.BatchNorm2d(16)
        self.BN2 = nn.BatchNorm2d(32)
        self.BN3 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.drop = nn.Dropout2d()

    def forward(self,x):
        #convlution 1 to 32 feature maps, 3x5 kernel
        x = self.prelu(self.BN1(self.conv1_1(x)))
        #batch normalization
        #prelu
        #max pooling 3x1
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = self.prelu(self.BN2(self.conv2_1(x)))
        # print(x.shape)
        #batch normalization
        #prelu
        x = self.drop(x)
        #max pooling 3x1
        x = self.pool(x)

        x = self.prelu(self.BN3(self.conv3(x)))
        # #batch normalization
        # x = self.BN3(x)
        # #prelu
        # x = self.prelu(x)
        x = self.drop(x)
        #
        x = self.pool(x)

        x = torch.flatten(x,1)
        #fully connected layers
        x = self.fc1(x)
        # # #prelu
        x = self.prelu(x)
        x = self.fc2(x)
        x = self.prelu(x)
        x = self.fc3(x)
        x = self.prelu(x)

        #softmax
        return F.softmax(x,dim=1)
