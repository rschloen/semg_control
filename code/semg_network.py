#!/usr/bin/env python3

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode


class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.fc1 = nn.Linear(3072,7)
        self.pool = nn.MaxPool2d((1,3))
        self.relu = nn.Relu()

    def forward(self,x):
        #convlution 1 to 32 feature maps, 3x3 kernel
        x = self.conv1(x)
        #relu
        x = self.relu(x)
        #max pooling 3x1
        x = self.pool(x)
        #fully connected layer
        x = self.fc1(x)
        #softmax
        return F.softmax(x)

class Network_enhanced(nn.Module):
    def __init__(self):
        super(Network_enhanced,self).__init__()
        self.conv1 = nn.Conv2d(1,32,(3,5))
        self.conv2 = nn.Conv2d(32,64,(3,5))
        self.fc1 = nn.Linear(500,7)
        self.pool = nn.MaxPool2d((1,3))
        self.BN1 = nn.BatchNorm1d(32)
        self.BN2 = nn.BatchNorm1d(64)
        self.BN3 = nn.BatchNorm1d(500)
        self.prelu = nn.PRelu()
        self.drop = nn.Dropout2d()

    def forward(self,x):
        #convlution 1 to 32 feature maps, 3x5 kernel
        x = self.conv1(x)
        #batch normalization
        x = self.BN1(x)
        #prelu
        x = self.prelu(x)
        #dropout
        x = self.drop(x)
        #max pooling 3x1
        x = self.pool(x)
        #convlution 2 to 64 feature maps, 3x5 kernel
        x = self.conv2(x)
        #batch normalization
        x = self.BN2(x)
        #prelu
        x = self.prelu(x)
        #dropout
        x = self.drop(x)
        #max pooling 3x1
        x = self.pool(x)
        #fully connected layer
        x = self.fc1(x)
        #batch normalization
        x = self.BN3(x)
        #prelu
        x = self.prelu(x)
        #dropout
        x = self.drop(x)
        #softmax
        return F.softmax(x)
