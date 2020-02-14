#!/usr/bin/env python3
from __future__ import print_function

import torch
from torch.utils import data
import numpy as np


class SEMG_Dataset(data.Dataset):
    def __init__(self,path,mode):
        all_data = np.load(path,allow_pickle='TRUE').item()
        self.data = all_data[mode]

    def __len__(self):
        return len(self.data)


    def __getitem__(self,index):
        samp = torch.from_numpy(self.data[index][0])
        label = torch.from_numpy(self.data[index][1])
        samp = samp.view(1,samp.shape[0],samp.shape[1])

        return samp, label



if __name__ == '__main__':
    path = "nina_data/all_6C_data_1.npy"
    dataset = SEMG_Dataset(path,'eval')
    params = {'batch_size': 4, 'shuffle': True,'num_workers': 4}
    train_loader = data.DataLoader(dataset, **params)
    count = 0
    for input, output in train_loader:
        print(input.size())
        print(output.size())
        count += 1

    print(count)


    # data = np.load("nina_data/all_6C_data_1.npy",allow_pickle='TRUE').item()
    # training_data = data['train']
    # print(torch.from_numpy(training_data[0][0]))
    # print(torch.from_numpy(training_data[0][1]))
