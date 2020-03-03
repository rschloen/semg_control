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
import argparse, sys, csv





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_data','-g',help='((True or true or t)/False) Generate data or train model with transfer learning')
    args=parser.parse_args()
    if args.gen_data == 'True' or args.gen_data == 'true' or args.gen_data == 't':
        gen_data = True
    else:
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
        count -= 1
        for i in range(3):
            emg_array = np.vstack((emg_array,np.loadtxt('myo_rec_data/raw_emg_'+path+'{}.csv'.format(i+1),delimiter=',')))
            gesture_array = np.append(gesture_array,np.loadtxt('myo_rec_data/gesture_'+path+'{}.csv'.format(i+1),delimiter=','))

        emg_array = np.delete(emg_array,0,0)
        # emg_array = np.loadtxt('myo_rec_data/raw_emg_JRS_7C_testing.csv',delimiter=',')
        # gesture_array = np.loadtxt('myo_rec_data/gesture_JRS_7C_testing.csv',delimiter=',')
        new_path = 'myo_rec_data/win_JRS_7C_comb3'
        # with open('myo_rec_data/win_JRS_7C_testing_stats.csv','w') as real_time:#, open('real_time_gest.csv','w') as real_gest:
        #     stat_writer = csv.writer(real_time, delimiter=',')
        #     mean_emg = np.mean(emg_array)
        #     std_emg = np.std(emg_array)
        #     stat_writer.writerow((mean_emg,std_emg))
        label_map = [(1,np.array([0,1,0,0,0,0,0])),(2,np.array([0,0,1,0,0,0,0])),(3,np.array([0,0,0,1,0,0,0])), (4,np.array([0,0,0,0,1,0,0])),(5,np.array([0,0,0,0,0,1,0])),(6,np.array([0,0,0,0,0,0,1]))]
        window_data(emg_array,gesture_array.T.astype(int),new_path,label_map)

    else:
        WT_PATH = 'nina_data/all_7C_data_comb_drop'
        PATH = 'nina_data/all_7C_data_comb'
        new_PATH = 'myo_rec_data/win_JRS_7C_comb3'
        model = Network_enhanced(7)
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        rates = np.logspace(-2.0,-4.0,20)
        decay = np.logspace(-1,-4,20) #best was
        lr = rates[10] # for adamw
        dec = decay[2] # for adamw
        optimizer = optim.AdamW(model.parameters(),lr=lr,weight_decay=dec)
        criterion = nn.MSELoss(reduction='mean')
        params = {'batch_size': 100, 'shuffle': True,'num_workers': 4}
        nt = Trainer(model,optimizer,criterion,device,PATH,params,epochs=1,early_stop=False)
        nt.stats['val']['model_wt'] = torch.load(WT_PATH+'.pt',map_location='cpu')
        tl,ta = nt.test(use_best_wt=True,epoch=1)


        num = model.fc3.in_features
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc1.parameters():
            param.requires_grad = True
        for param in model.fc2.parameters():
            param.requires_grad = True
        for param in model.fc3.parameters():
            param.requires_grad = True
        ft_nt = Trainer(model,optimizer,criterion,device,new_PATH,params,epochs=1000,early_stop=False)
        ft_nt.stats['val']['model_wt'] = torch.load(WT_PATH+'.pt',map_location='cpu')
        ft_nt.model.fc3 = nn.Linear(num,7)
        ft_nt.model = ft_nt.model.to(device)

        ft_nt.train(val_train=True)
        tl, ta = ft_nt.test(use_best_wt=True, epoch=1)
        torch.save(ft_nt.stats['val']['model_wt'],new_PATH+'_adamw_best_tran2.pt')
        plot_path = 'win_JRS_7C_2_adamw_best_tran2'
        ft_nt.plot_loss(plot_path,True)
