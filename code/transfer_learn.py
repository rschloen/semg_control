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
from semg_network import Network_XL, Network_enhanced
from data_loader import SEMG_Dataset,window_data
from train_net import Trainer, best_model_params
import argparse, sys, csv
# import pnn_train as pnn



def most_active(array):
    '''Calculates IEMG (integrated EMG) for each channel of rectified emg signal in array and returns most active channel.
    ARGS: array: Can be either (str) path to csv file containing emg data or numpy array of data.
    RETURNS: (int) most active channel'''
    if type(array) == str:
        return np.argmax(np.trapz(np.abs(np.loadtxt(array,delimiter=',')),axis=0))
    else:
        return np.argmax(np.trapz(np.abs(array),axis=0))

def circle_shift(array,start,goal):
    '''Circle shifts array from start to goal. Used for calibrating channels from different subjects/sessions
    ARGS: array: emg data array
          start: Most active channel for dataset
          goal: Channel to line the most active channel up to
    RETURNS: shifted: emg data array with channels shifted'''
    shifted = np.zeros(array.shape)
    s = goal - start
    for i in range(8):
        if i+s > 7:
            j = i+s-8
        else:
            j = i+s
        shifted[:,j] = array[:,i]
    return shifted


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
        for file in os.listdir("myo_rec_data/"): #count number of emg data files in folder
            if file.endswith(".csv"):
                count += .5
        count -= 1
        first_activity =  most_active('myo_rec_data/raw_emg_'+path+'1.csv')
        print('Target channel {}'.format(first_activity))

        for i in range(int(count)): # loop through data files in folder to build dataset containing all of them
            emg_array = np.zeros((1,8))
            gesture_array = np.array([])
            temp_arr = np.loadtxt('myo_rec_data/raw_emg_'+path+'{}.csv'.format(i+1),delimiter=',')
            active_chan = most_active(temp_arr)
            print('File {} most active channel {}'.format(i,active_chan))
            shifted_array = circle_shift(temp_arr,active_chan,first_activity)
            emg_array = np.vstack((emg_array,shifted_array))
            gesture_array = np.append(gesture_array,np.loadtxt('myo_rec_data/gesture_'+path+'{}.csv'.format(i+1),delimiter=','))

        emg_array = np.delete(emg_array,0,0) #first row was initialized with zeros, needs to be removed
        new_path = 'myo_rec_data/win_JRS_7C_comb7_shifted' #file to store new combined dataset
        with open('myo_rec_data/win_JRS_7C_comb7_shifted_stats.txt','w') as real_time:
            stat_writer = csv.writer(real_time, delimiter=',')
            mean_emg = np.mean(emg_array)
            std_emg = np.std(emg_array)
            stat_writer.writerow((mean_emg,std_emg))
        label_map = [(1,np.array([0,1,0,0,0,0,0])),(2,np.array([0,0,1,0,0,0,0])),(3,np.array([0,0,0,1,0,0,0])), (4,np.array([0,0,0,0,1,0,0])),(5,np.array([0,0,0,0,0,1,0])),(6,np.array([0,0,0,0,0,0,1]))]
        window_data(emg_array,gesture_array.T.astype(int),new_path,label_map)

    else:
        WT_PATH = 'nina_data/all_7C_data_comb_XL_cross' # file with model state dictionary
        PATH = 'nina_data/all_7C_data_comb' # file with pretraining data
        new_PATH = 'myo_rec_data/win_JRS_7C_comb7_shifted' # file with new data for transfer learning
        model = Network_XL(7)
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

        # Test pretrained network
        nt = best_model_params(model,PATH)
        nt.stats['val']['model_wt'] = torch.load(WT_PATH+'.pt',map_location='cpu')
        tl,ta = nt.test(use_best_wt=True,epoch=1)

        model.load_state_dict(torch.load(WT_PATH+'.pt',map_location='cpu'))

        ## Examples of selectively freezing or unfreezeing layer's weights for transfer learning
        # num2 = model.fc2.in_features
        # model.fc2 = nn.Linear(num2, num3)
        # for param in model.parameters():
        #     param.requires_grad = False
        # for param in model.conv3.parameters():
        #     param.requires_grad = True
        # for param in model.fc1.parameters():
        # #     param.requires_grad = True
        # # for param in model.fc2.parameters():
        # #     param.requires_grad = True
        # for param in model.fc3.parameters():
        #     param.requires_grad = True
        # num3 = model.fc3.in_features
        # model.fc3 = nn.Linear(num3, 7)
        # model = ft_nt.model.to(device)

        # Begin transfer learning
        ft_nt = best_model_params(model,new_PATH)
        ft_nt.max_epochs = 500
        ft_nt.early_stop = True
        ft_nt.train(val_train=True)
        tl, ta = ft_nt.test(use_best_wt=True, epoch=1)
        torch.save(ft_nt.wt_hist['val'][np.argmin(ft_nt.loss_hist['val'])],new_PATH+'_XL_cross_tran_final2.pt')
        plot_path = 'win_JRS_7C_comb7_shifted_XL_cross_tran_final'
        ft_nt.plot_loss(plot_path,True)

        '''For PNN (incomplete)'''
        # dir = 'myo_rec_data/win_JRS_7C_'
        # new_PATH = [dir+'1_shifted',dir+'2_shifted',dir+'3_shifted',dir+'4_shifted',dir+'5_shifted',dir+'6_shifted']
        # ft_nt = pnn.best_model_params(model,new_PATH)
        # print(ft_nt.new_model.state_dict())
        # ft_nt.train(val_train=True)
        # tl, ta = ft_nt.test(use_best_wt=True, epoch=1)
        # torch.save(ft_nt.wt_hist['val'][np.argmin(ft_nt.loss_hist['val'])],new_PATH+'_XL_cross_tran2.pt')
        # plot_path = 'win_JRS_7C_comb7_shifted_XL_cross_tran2'
