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
        self.fc1 = nn.Linear(3072,12)
        self.pool = nn.MaxPool2d((1,3))
        self.relu = nn.ReLU()

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
        self.prelu = nn.PReLU()
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


def train_model(model,criterion,optimizer,scheduler,num_epochs=10):
    """Modified example from pytorch tutorials"""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
        running_loss = 0.0
        running_correct = 0

        for input, labels in data[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, prediction = torch.max(outputs,1)
                loss = criterion(outputs,labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(prediction = labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss/len(data[phase])
            epoch_acc = running_correct.double()/len(data[phase])
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    total_time = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(total_time//60,total_time%60))
    model.load_state_dict(best_model_wts)
    return model


def main():
    model = Network()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    criterion = torch.nn.MSELoss(reduction='sum')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")






if __name__ == '__main__':
    main()
