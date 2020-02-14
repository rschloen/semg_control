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
from scipy.io import loadmat
from statistics import mode


plt.ion()   # interactive mode


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
        self.conv1 = nn.Conv2d(1,32,(5,3))
        self.conv2 = nn.Conv2d(32,64,(5,3))
        self.fc1 = nn.Linear(1024,num_classes)
        self.fc2 = nn.Linear(500,num_classes)
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
        # # #prelu
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
    prev_loss = 0
    loss_cnt = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)
        print(scheduler.get_lr())
        for phase in ['train','eval']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_correct = 0
            counter = 0
            for input, label in data[phase]:
                input = torch.from_numpy(input)
                label = torch.from_numpy(np.array([label]))
                input = input.view(1,1,input.shape[0],input.shape[1])
                input = input.to(device)
                label = label.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(input)
                    prediction = torch.argmax(output,dim=1)
                    # print(output.size())
                    # loss = F.nll_loss(output,torch.tensor([torch.argmax(label)]))
                    loss = criterion(output,label.float())


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # print(list(model.parameters())[0].grad)
                running_loss += loss.item() * input.size(0)
                '''Add early stopping: if change in loss less than ... x times, stop.
                Useful check if updating properly as well'''
                if (running_loss - prev_loss) < 1e-6: loss_cnt += 1
                if loss_cnt > 5: break
                prev_loss = running_loss
                # print(torch.argmax(labels))
                # print(prediction == torch.argmax(labels))
                running_correct +=  torch.sum(prediction == torch.argmax(label))  #torch.sum mainly acts to put value in correct type/format
                if counter % 2000 == 1999:
                    print('{} Loss: {:.4f}'.format(phase,running_loss/counter))
                    print(list(model.parameters())[0].grad)
                    counter = 0
                    running_loss = 0.0
                counter += 1
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss/len(data[phase])
            epoch_acc = (running_correct.double()/len(data[phase]))*100 #percent

            if phase == 'eval' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))
    total_time = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(total_time//60,total_time%60))
    model.load_state_dict(best_model_wts)
    return model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## Initialize model
    # model = Network(6)
    model = Network_enhanced(6)
    # print(list(model.parameters()))

    # Initialize hyperparameters and supporting functions
    learning_rate = 0.02
    optimizer = optim.SGD(model.parameters(),lr=learning_rate)
    criterion = nn.MSELoss(reduction='mean')
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

    ## Load and combine data from all subjects
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

    ## Initialize sampling parameters
    ## Want a 260ms window (52 samples) of all eight channels as inputs with 235ms overlap (from paper)
    window_size = 52 #samples, 260ms
    overlap = 47 #sample, 235ms
    step = window_size - overlap
    i = 0
    data = {'train':[],'eval':[]}
    map = [(1,np.array([0,1,0,0,0,0])),(3,np.array([0,0,1,0,0,0])),(5,np.array([0,0,0,1,0,0])),(7,np.array([0,0,0,0,1,0])),(12,np.array([0,0,0,0,0,1]))]

    ## Sort and label it for training/validation
    while i < len(emg_data)-window_size:
        if np.random.randint(1,11) < 8:
            set = 'train'
        else:
            set = 'eval'
        label = mode(list(restim[i:i+window_size][0]))
        ## Normalize Data??
        emg_win = emg_data[i:i+window_size,:8]
        norm_emg = (emg_win - np.mean(emg_win))/(np.std(emg_win))
        if label == 0:
            if np.random.randint(6) == 1: #only save about fifth of the rest states
                data[set].append([norm_emg,np.array([1,0,0,0,0,0])])
        else:
            for act, new in map:
                if label == act:
                    data[set].append([norm_emg,new])
        i += step


    # pkl_emg_data = json.dumps(data)
    np.save("nina_data/all_6C_data_1.npy",data)
    # pickle.write(data,f)
    # f.close()

    # print(type(data))

    # count = 0
    # test = torch.from_numpy(np.array(10))
    # for input, label in data['train']:
    #     label = torch.from_numpy(label)
    #     print(torch.argmax(label))
    #     count += torch.argmax(label) == test
    #     break
    # print(count)

    # print(len(data['train']))
    # print(len(data['eval']))
    # print((len(data['train'])+len(data['eval'])))
    # print((len(data['train'])+len(data['eval']))*.8)
    # print((len(data['train'])+len(data['eval']))*.2)

    ## Train model
    # best_model = train_model(model,criterion,optimizer,exp_lr_scheduler,data)





if __name__ == '__main__':
    main()
