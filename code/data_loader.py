#!/usr/bin/env python3
from __future__ import print_function

import torch
from torch.utils import data
import numpy as np
from scipy.io import loadmat
from statistics import mode


class SEMG_Dataset(data.Dataset):
    def __init__(self,path,mode,fold):
        all_data = np.load(path+'.npy',allow_pickle='TRUE').item()
        num_folds = 10.
        rng = int(len(all_data['train'])/num_folds)
        # print(rng)
        # print(rng*fold)
        # print(rng*(fold+1))
        self.val_data = all_data['train'][rng*fold:rng*(fold+1)]
        self.train_data = all_data['train']
        # print(len(self.train_data))
        del self.train_data[rng*fold:rng*(fold+1)]
        # print(len(self.train_data))
        self.test_data = all_data['test']

        if mode == 'train':
            self.data = self.train_data[:]
        elif mode == 'test':
            self.data = self.test_data[:]
        elif mode == 'val':
            self.data = self.val_data[:]




    def __len__(self):
        return len(self.data)


    def __getitem__(self,index):
        samp = torch.from_numpy(self.data[index][0])
        label = torch.from_numpy(self.data[index][1])
        samp = samp.view(1,samp.shape[0],samp.shape[1])

        return samp, label



if __name__ == '__main__':
    gen_data = False
    if gen_data:
        ## Load and combine data from all subjects
        x = loadmat('../../ninapro_data/s1/S1_E1_A1.mat') #Exercise 1. 12 hand gestures included gestures of interest
        emg_data = x['emg'] #first 8 columns are from myo closest to elbow, next 8 are from second myo rotated 22 degs
        restim = x['restimulus'] #restimulus and rerepetition are the corrected indexes for the movements
        rep = x['rerepetition'] #6 repetitions per gesture
        for i in range(2,11):
            x = loadmat('../../ninapro_data/s'+str(i)+'/S'+str(i)+'_E1_A1.mat') #Exercise 1. 12 hand gestures included gestures of interest
            emg_data = np.vstack((emg_data,x['emg']))
            restim = np.vstack((restim,x['restimulus']))
            rep = np.vstack((rep,x['rerepetition']))
        # print(restim[-100:])
            y = loadmat('../../ninapro_data/s'+str(i)+'/S'+str(i)+'_E2_A1.mat')
            e2 = y['emg']
            e2_res = y['restimulus']
            for n in range(len(e2)):
                if e2_res[n] == 6:
                    emg_data = np.vstack((emg_data,e2[n]))
                    restim = np.vstack((restim,26))

        ## Initialize sampling parameters
        ## Want a 260ms window (52 samples) of all eight channels as inputs with 235ms overlap (from paper)
        window_size = 52 #samples, 260ms
        overlap = 47 #sample, 235ms
        step = window_size - overlap
        i = 0
        data = {'train':[],'val':[],'test':[]}
        map = [(1,np.array([0,1,0,0,0,0,0])),(3,np.array([0,0,1,0,0,0,0])),(5,np.array([0,0,0,1,0,0,0])), (7,np.array([0,0,0,0,1,0,0])),(12,np.array([0,0,0,0,0,1,0])),(26,np.array([0,0,0,0,0,0,1]))]

        ## Sort and label it for training/validation
        while i < len(emg_data)-window_size:
            # Split all data into either training(80%), validation(10%), or testing(10%)
            if np.random.randint(1,11) < 9:
                # if np.random.randint(1,11) < 9:
                set = 'train'
                # else:
                    # set = 'val'
            else:
                set = 'test'


            label = mode(list(restim[i:i+window_size][0]))
            ## Normalize Data??
            emg_win = emg_data[i:i+window_size,:8]
            normalize = True
            if normalize:
                emg_win = (emg_win - np.mean(emg_win))/(np.std(emg_win))
                # emg_win = emg_win/128. #not helpful

            if label == 0:
                if np.random.randint(6) == 1: #only save about fifth of the rest states
                    data[set].append([emg_win,np.array([1,0,0,0,0,0,0])])
            else:
                for act, new in map:
                    if label == act:
                        data[set].append([emg_win,new])
            i += step

        np.save("nina_data/all_7C_data_2.npy",data)

    else:
        ## Test distribution of
        path = "nina_data/all_7C_data_2"
        f = open(path+'.txt','w')
        modes = ['train','val','test']
        tot = []
        data_len = 0
        # prev_dataset = SEMG_Dataset(path,'train',0)
        for phase in modes:
            dataset = SEMG_Dataset(path,phase,0)
            data_len += len(dataset)
            params = {'batch_size': 10, 'shuffle': True,'num_workers': 4}
            train_loader = data.DataLoader(dataset, **params)
            test = [0,1,2,3,4,5,6]
            # for c in test:
            c1 = c2 = c3 = c4 = c5 = c6 = c7 = 0
            for input, label in train_loader:
                # print(torch.argmax(label))
                c1 += (torch.argmax(label,dim=1) == test[0]).sum().item()
                c2 += (torch.argmax(label,dim=1) == test[1]).sum().item()
                c3 += (torch.argmax(label,dim=1) == test[2]).sum().item()
                c4 += (torch.argmax(label,dim=1) == test[3]).sum().item()
                c5 += (torch.argmax(label,dim=1) == test[4]).sum().item()
                c6 += (torch.argmax(label,dim=1) == test[5]).sum().item()
                c7 += (torch.argmax(label,dim=1) == test[6]).sum().item()
            res = [c1,c2,c3,c4,c5,c6,c7]
            for c in test:
                print("Class {} has {} samples".format(c+1,res[c]))
                f.write("Class {} has {} samples\n".format(c+1,res[c]))
            tot.append(c1+c2+c3+c4+c5+c6+c7)

        print('Total training samples: {}, {:.4f}%'.format(tot[0],(tot[0]/data_len)*100))
        f.write('Total training samples: {}, {:.4f}%'.format(tot[0],(tot[0]/data_len)*100))
        print('Total testing samples: {}, {:.4f}%'.format(tot[2],(tot[2]/data_len)*100))
        f.write('Total testing samples: {}, {:.4f}%'.format(tot[2],(tot[2]/data_len)*100))
        f.close()
