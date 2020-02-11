#!/usr/bin/env python3

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from scipy.io import loadmat
from statistics import mode
# import scipy
# import scipy.stats
# print(scipy.__version__)

plt.ion()   # interactive mode


class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.fc1 = nn.Linear(3072,6)
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
    def __init__(self):
        super(Network_enhanced,self).__init__()
        self.conv1 = nn.Conv2d(1,32,(5,3))
        self.conv2 = nn.Conv2d(32,64,(5,3))
        self.fc1 = nn.Linear(1024,6)
        self.fc2 = nn.Linear(500,6)
        self.pool = nn.MaxPool2d((3,1))
        self.BN1 = nn.BatchNorm2d(32)
        self.BN2 = nn.BatchNorm2d(64)
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
        # print(x.shape)
        #convlution 2 to 64 feature maps, 3x5 kernel
        x = self.conv2(x)
        # print("CONV2")
        #batch normalization
        x = self.BN2(x)
        #prelu
        x = self.prelu(x)
        #dropout
        x = self.drop(x)
        #max pooling 3x1
        x = self.pool(x)
        # print(x.shape)
        x = torch.flatten(x,1)
        #fully connected layer
        x = self.fc1(x)
        # print('FC')
        # print(x)
        #batch normalization
        # x = self.BN3(x)
        # # # #prelu
        # x = self.prelu(x)
        # # # #dropout
        # x = self.drop(x)
        # #
        # x = self.fc2(x)
        #softmax
        return F.softmax(x,dim=1)


def train_model(model,criterion,optimizer,scheduler,data,num_epochs=10):
    """Modified example from pytorch tutorials, Author: Sasank Chilamkurthy"""
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)
        for phase in ['train','eval']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_correct = 0
            print(scheduler.get_lr())
            for inputs, labels in data[phase]:
                inputs = torch.from_numpy(inputs)
                labels = torch.from_numpy(np.array([labels]))
                inputs = inputs.view(1,1,inputs.shape[0],inputs.shape[1])
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(outputs[0])
                    prediction = torch.argmax(outputs,dim=1)
                    # print(prediction)
                    loss = criterion(outputs,labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                # print(torch.argmax(labels))
                # print(prediction == torch.argmax(labels))
                running_correct +=  torch.sum(prediction == torch.argmax(labels))  #torch.sum mainly acts to put value in correct type/format
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss/len(data[phase])
            epoch_acc = (running_correct.double()/len(data[phase]))*100 #percent
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))

            if phase == 'eval' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,best_acc))
    total_time = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(total_time//60,total_time%60))
    model.load_state_dict(best_model_wts)
    return model

def shuffle(data,size):
    #shuffle input data randomly
    temp_in = []
    temp_out = []
    for _ in range(size/2):
        index = np.random.randint(0,size,2)
        temp_in[:] = input[index[0]]
        temp_out[:] = output[index[0]]
        input[index[0]] = input[index[1]]
        input[index[1]] = temp_in
        output[index[0]] = output[index[1]]
        output[index[1]] = temp_out
    return input,output

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = Network()
    model = Network_enhanced()

    learning_rate = 0.02
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    criterion = nn.MSELoss(reduction='sum')
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.1)
    # emg_data = np.array([])
    x = loadmat('/home/rschloen/WinterProj/ninapro_data/s1/S1_E1_A1.mat') #Exercise 1. 12 hand gestures included gestures of interest
    emg_data = x['emg'] #first 8 columns are from myo closest to elbow, next 8 are from second myo rotated 22 degs
    restim = x['restimulus'] #restimulus and rerepetition are the corrected indexes for the movements
    rep = x['rerepetition'] #6 repetitions per gesture
    for i in range(2,11):
        x = loadmat('/home/rschloen/WinterProj/ninapro_data/s'+str(i)+'/S'+str(i)+'_E1_A1.mat') #Exercise 1. 12 hand gestures included gestures of interest
        emg_data = np.vstack((emg_data,x['emg']))
        restim = np.vstack((restim,x['restimulus']))
        rep = np.vstack((rep,x['rerepetition']))
    # print(emg_data.shape)
    #Want a 260ms window (52 samples) of all eight channels as inputs with 235ms overlap (from paper)
    window_size = 52 #samples, 260ms
    overlap = 47 #sample, 235ms
    step = window_size - overlap
    i = 0
    data = {'train':[],'eval':[]}
    map = [(1,np.array([0,1,0,0,0,0])),(3,np.array([0,0,1,0,0,0])),(5,np.array([0,0,0,1,0,0])),(7,np.array([0,0,0,0,1,0])),(12,np.array([0,0,0,0,0,1]))]

    while i < len(emg_data)-window_size:
        if np.random.randint(1,11) < 8:
            set = 'train'
        else:
            set = 'eval'
        label = mode(list(restim[i:i+window_size][0]))
        if label == 0:
            if np.random.randint(10) == 1: #only save about half of the rest states
                data[set].append([emg_data[i:i+window_size,:8],np.array([1,0,0,0,0,0])])
        else:
            for act, new in map:
                if label == act:
                    data[set].append([emg_data[i:i+window_size,:8],new])
        i += step

    # count = 0
    # test = torch.from_numpy(np.array(10))
    # print(test)
    # print(test.double())
    # acc = test.double()/2
    # print("acc: {}".format(acc))
    # for input, label in data['train']:
    #     label = torch.from_numpy(label)
    #     print(torch.argmax(label))
    #     count += torch.argmax(label) == test
    #     break
    #
    # print(count)

    # loss = nn.CrossEntropyLoss()
    # input = torch.randn(3, 5, requires_grad=True)
    # print(input)
    # target = torch.empty(3, dtype=torch.long).random_(5)
    # print(target)
    # output = loss(input, target)
    # print(output)
    # print(len(data['train']))
    # print(len(data['eval']))
    # print((len(data['train'])+len(data['eval'])))
    # print((len(data['train'])+len(data['eval']))*.8)
    # print((len(data['train'])+len(data['eval']))*.2)


    best_model = train_model(model,criterion,optimizer,exp_lr_scheduler,data)





if __name__ == '__main__':
    main()
