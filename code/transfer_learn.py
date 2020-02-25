#!/usr/bin/env python3
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
import torch.nn.functional as F
import numpy as np
import os
from scipy.io import loadmat
from semg_network import Network, Network_enhanced
from data_loader import SEMG_Dataset,window_data
from train_net import Trainer





if __name__ == '__main__':
    gen_data = False
    if gen_data:
        path = 'JRS_7C_'
        emg_array = np.zeros((1,8))
        gesture_array = np.array([])
        count = 0
        for file in os.listdir("myo_rec_data/"):
            if file.endswith(".csv"):
                count += .5
        # print(int(count))
        for i in range(int(count)):
            emg_array = np.vstack((emg_array,np.loadtxt('myo_rec_data/raw_emg_'+path+'{}.csv'.format(i+1),delimiter=',')))
            gesture_array = np.append(gesture_array,np.loadtxt('myo_rec_data/gesture_'+path+'{}.csv'.format(i+1),delimiter=','))
        emg_array = np.delete(emg_array,0,0)
        path = 'myo_rec_data/win_JRS_7C_1'
        label_map = [(1,np.array([0,1,0,0,0,0,0])),(2,np.array([0,0,1,0,0,0,0])),(3,np.array([0,0,0,1,0,0,0])), (4,np.array([0,0,0,0,1,0,0])),(5,np.array([0,0,0,0,0,1,0])),(6,np.array([0,0,0,0,0,0,1]))]
        window_data(emg_array,gesture_array.T.astype(int),path,label_map)

    else:
        PATH = 'nina_data/all_7C_data_1'
        new_PATH = 'myo_rec_data/win_JRS_7C_1'
        model = Network_enhanced(7)
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        optimizer = optim.SGD(model.parameters(),lr=0.02)
        criterion = nn.MSELoss(reduction='mean')
        params = {'batch_size': 1, 'shuffle': True,'num_workers': 4}


        # nt = Trainer(model,optimizer,criterion,device,PATH,params,epochs=1)
        # nt.stats['train']['model_wt'] = torch.load(PATH+'.pt')
        # tl,ta = nt.test(use_best_wt=True,epoch=1)
        # num = model.fc3.in_features
        # model.fc3 = nn.Linear(num,7)
        ft_nt = Trainer(model,optimizer,criterion,device,new_PATH,params,epochs=100)
        ft_nt.train(val_train=True)
        tl, ta = ft_nt.test(use_best_wt=True, epoch=1)
