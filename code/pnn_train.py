#!/usr/bin/env python3
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import argparse
from semg_network import Network, Network_enhanced, Network_XL
from data_loader import SEMG_Dataset



class PNN_Trainer():
    '''This class is used to train and test progressive neural networks, plot the models performance throughout training and save useful statistics'''
    def __init__(self,model,optimizer,criterion,device,data_path,loader_params,file=None,scheduler=None,epochs=100,early_stop=False):
        '''ARGS: model: class object of the pytorch model to train
                 optimizer: pytorch optim object for the eoptimization algorithm to use (Ex: SDG, AdamW)
                 criterion: pytorch loss function object such as MSE or CrossEntropyLoss
                 device: str specifying cuda or cpu
                 data_path: dir/file of data set to use in pytorch Dataset class
                 loader_params: parameters for pytorch dataloaders
                 file: (default=None) dir/file to store stats in
                 scheduler: (default=None) pytorch learning rate scheduler object
                 epochs: (default=100) Max epochs of training to perform
                 early_stop: (default=False) Allows early stopping, such as when loss plateaus during training  '''
        self.source_model = model.to(device)
        self.new_model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.f = file
        self.max_epochs = epochs
        self.early_stop = early_stop
        self.og_wt = copy.deepcopy(self.source_model.state_dict())
        self.path = data_path
        self.params = loader_params
        self.data_loaders = {}
        self.data_lens = {}
        self.models = {}
        self.history = {}
        # Generates series of 'subjects' that are used for progrssivly training the network
        for i in range(len(data_path)):
            self.models[i] = Network_XL(7)
            self.models[i].load_state_dict(self.og_wt)
            train_dataset = SEMG_Dataset(data_path[i],'train',0)
            val_dataset = SEMG_Dataset(data_path[i],'val',0)
            test_dataset = SEMG_Dataset(data_path[i],'test',0)
            self.data_loaders[i] = {'train':data.DataLoader(train_dataset,**loader_params),
                              'val':data.DataLoader(val_dataset,batch_size=loader_params['batch_size'], shuffle=False,num_workers=loader_params['num_workers']),
                              'test':data.DataLoader(test_dataset,batch_size=loader_params['batch_size'], shuffle=False,num_workers=loader_params['num_workers'])}
            self.data_lens[i] = {'train':len(train_dataset),'val':len(val_dataset),'test':len(test_dataset)}
            self.history[i] = {'loss':{'train':[],'val':[]},'acc':{'train':[],'val':[]},'wt':{'train':[],'val':[]}}
        self.data_loaders['new'] = self.data_loaders[0]
        self.data_lens['new'] = self.data_lens[0]
        self.history['new'] = {'loss':{'train':[],'val':[]},'acc':{'train':[],'val':[]},'wt':{'train':[],'val':[]}}

        self.stats = {'train':{'loss': float('+inf'),
                           'model_wt': copy.deepcopy(self.source_model.state_dict()),
                           'acc': 0,
                           'epoch': 0,'fold':0},
                      'val':{'loss': float('+inf'),
                           'model_wt': copy.deepcopy(self.source_model.state_dict()),
                           'acc': 0,
                           'epoch': 0,'fold':0}}



    def one_epoch(self,model,num,phase):
        '''Iterates through dataset to complete ONE epoch. Steps: pull sample(s) from data loader,
        pass data through model, calculate loss, update weights, take a step for optimizer. Keeps track of
        running loss and running correct classifications.
        ARGS: model: which subjects model to train
              num: subject number
              phase: ('train','val','test') sets whether or not to use training specific
        steps (enabling grad,loss.backward(),etc.)
        RETURNS: running_loss: total loss over all data points
                 cor_classify: number of correctly classified samples'''
        running_loss = 0.0
        cor_classify = 0.0
        i = 0
        for input,label in self.data_loaders[num][phase]:
            if phase == 'train':
                model.train()
                torch.set_grad_enabled(True)
                self.optimizer.zero_grad()
            elif phase == 'test' or phase == 'val':
                model.eval()
                torch.set_grad_enabled(False)
            input = input.to(self.device)
            label = label.to(self.device)
            output = model(input)
            loss = self.criterion(output,torch.argmax(label,dim=1))
            running_loss += loss.item()
            if phase == 'train':
                loss.backward()
                self.optimizer.step()

            #Does prediction == actual class?
            cor_classify += (torch.argmax(output,dim=1) == torch.argmax(label,dim=1)).sum().item()
            i+=1

        return running_loss, cor_classify


    def train(self,val_train=True):
        '''Controls Training loop: runs set number of epochs and calculates/prints/saves stats for training/validation
        phases. Checks change in epoch loss for early stopping due to plateau.
        ARGS: val_train: True/False (defualt:True) denotes whether to run validation phase or not '''
        since = time.time()
        prev_loss = float('+inf')
        self.prev_t_loss = float('+inf')
        print('Training...')
        self.loss_cnt = 0
        for epoch in range(1,self.max_epochs+1):
            for i in range(len(self.path)):
                e_loss, e_classify = self.one_epoch(self.models[i],i,'train')
                e_loss /= self.data_lens[i]['train']
                e_acc = (e_classify/self.data_lens[i]['train'])*100
                temp = copy.deepcopy(self.new_model.state_dict())
                print(temp['conv1_1.weight'] == self.new_model.state_dict()['conv1_1.weight'])
                temp += self.models[i].state_dict()
                print(temp['conv1_1.weight'] == self.new_model.state_dict()['conv1_1.weight'])

                # for param_new, param_sub in zip(self.new_model.parameters(),self.models[i].parameters()):
                    # param_new = param_new + param_sub
                # print("Epoch: {}/{}\nPhase: Train  Loss: {:.8f}    Accuracy: {:.4f}".format(epoch,self.max_epochs,e_loss,e_acc))
                self.history[i]['loss']['train'].append(e_loss)
                self.history[i]['acc']['train'].append(e_acc)
                if val_train:
                    t_loss, t_acc = self.test(self.models[i],i,False,epoch)
                    # print("Phase: Validation    Loss: {:.8f}    Accuracy: {:.4f}".format(t_loss,t_acc))

                if self.early_stop:
                    '''Add early stopping: if change in loss less than ... x times, stop.
                    Useful check if updating properly as well'''
                    if abs(e_loss - prev_loss) < 1e-8:
                        self.loss_cnt += 1 #or e_loss > prev_loss
                    else:
                        self.loss_cnt = 0
                    if self.loss_cnt > 10: break
                    prev_loss = e_loss

                self.history[i]['wt']['train'].append(copy.deepcopy(self.models[i].state_dict()))

                self.progress_bar(epoch)
            new_loss, new_classify = test(self.new_model,'new',False)

        print('\n')
        tl_ind,vl_ind = np.argmin(self.history['new']['loss']['train']),np.argmin(self.history['new']['loss']['val'])
        if self.f != None:
            self.f.write("Training Summary:\n")
            self.f.write("Best Training epoch was {} of {} in fold {} with Loss: {:.8f}    Accuracy: {:.4f}\n".format(tl_ind,epoch,0,self.history['new']['loss']['train'][tl_ind],self.history['new']['acc']['train'][tl_ind]))
            self.f.write("Best Validation epoch was {} of {} in fold {} with Loss: {:.8f}    Accuracy: {:.4f}\n".format(vl_ind,epoch,0,self.history['new']['loss']['val'][vl_ind],self.history['new']['acc']['val'][vl_ind]))
            total_time = time.time() - since
            self.f.write('Training completed in {:.0f}m {:.0f}s\n'.format(total_time//60,total_time%60))
        total_time = time.time() - since
        print("Training Summary:")
        print("\tBest Training epoch was {} of {} in fold {} with Loss: {:.8f}    Accuracy: {:.4f}".format(tl_ind,epoch,0,self.history['loss']['train'][tl_ind],self.history['acc']['train'][tl_ind]))
        print("\tBest Validation epoch was {} of {} in fold {} with Loss: {:.8f}    Accuracy: {:.4f}".format(vl_ind,epoch,0,self.history['loss']['val'][vl_ind],self.history['acc']['val'][vl_ind]))
        print('Training completed in {:.0f}m {:.0f}s\n'.format(total_time//60,total_time%60))


    def test(self,model,num,use_best_wt,epoch=1):
        '''Can be used for either validation phase during training or testing on trained model. When one_epoch
        function is called, passes argument to put model in eval mode and disable grad.
        ARGS: model: which subjects model to train
              num: subject number
              use_best_wt: True/False denotes whether its a validation or testing phase.
              epoch: current epoch, used in printing and saving stats for validation Phase
        RETURNS: test_loss: validation or testing epoch loss
                 test_acc: validation or testing epoch accuracy'''
        set = 'val'
        if use_best_wt:
            print("Testing with best weights...")
            if len(self.history[num]['wt']['val']) != 0:
                self.source_model.load_state_dict(self.history[num]['wt']['val'][np.argmin(self.history[num]['loss']['val'])])
                print('From history')
            else:
                self.source_model.load_state_dict(self.stats['val']['model_wt'])
            set = 'test'

        test_loss, test_correct = self.one_epoch(model,num,set)
        test_loss /= self.data_lens[num][set]
        test_acc = (test_correct/self.data_lens[num][set])*100
        if not use_best_wt:
            self.history[num]['loss']['val'].append(test_loss)
            self.history[num]['acc']['val'].append(test_acc)
            self.history[num]['wt']['val'].append(copy.deepcopy(model.state_dict()))

        if not use_best_wt and self.early_stop:
            if abs(test_loss - self.prev_t_loss) < 1e-6:
                self.loss_cnt += 1
            else:
                self.loss_cnt = 0
            self.prev_t_loss = test_loss
        if use_best_wt:
            print("Test Summary:")
            print("    Loss = {:.8f}".format(test_loss))
            print("    Correct: {}/{}".format(test_correct,self.data_lens[num][set]))
            print("    Accuracy: {:.4f}".format(test_acc))
            if self.f != None:
                self.f.write("Test Summary:\n")
                self.f.write("    Loss = {:.8f}\n".format(test_loss))
                self.f.write("    Correct: {}/{}\n".format(test_correct,self.data_lens[num][set]))
                self.f.write("    Accuracy: {:.4f}\n".format(test_acc))
        return test_loss, test_acc


    def plot_loss(self,path,save_plt):
        '''Plot training and validation loss and accuracy for each epoch
        ARGS: path: directory and name of file to save plot to
              save_plt: Boolean. Whether to save the plots or not'''
        path_loss = 'img/'+path+'_loss.png'
        path_acc = 'img/'+path+'_acc.png'
        fig = plt.figure()
        ax = fig.gca()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over {} Epochs'.format(len(self.history['loss']['train'])))
        x = np.arange(1,len(self.history['loss']['train'])+1)
        train, = plt.plot(x,self.history['loss']['train'],'k')
        val, = plt.plot(x,self.history['loss']['val'],'r')
        plt.legend((train,val),('Training','Validation'))
        if save_plt:
            plt.savefig(path_loss,format='png')
        fig.clf()
        fig = plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over {} Epochs'.format(len(self.history['acc']['train'])))
        train, = plt.plot(x,self.history['acc']['train'],'k')
        val, = plt.plot(x,self.history['acc']['val'],'r')
        plt.legend((train,val),('Training','Validation'))
        if save_plt:
            plt.savefig(path_acc,format='png')
        fig.clf()
        plt.close('all')

    def progress_bar(self,cur):
         '''Displays progress bar
        ARGS: cur: number of items (epochs in this case) completed'''
        bar_len = 25
        percent = cur/self.max_epochs
        fill = round(percent*bar_len)
        bar = '['+'#'*fill+'-'*(bar_len-fill)+']  [{:.2f}%]'.format(percent*100)

        print('\r'+bar,end='')
