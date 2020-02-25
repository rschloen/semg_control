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
from semg_network import Network, Network_enhanced
from data_loader import SEMG_Dataset



class Trainer():
    def __init__(self,model,optimizer,criterion,device,data_path,loader_params,scheduler=None,epochs=25):
        self.model = model.to(device)#.float()
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = epochs
        self.og_wt = copy.deepcopy(self.model.state_dict())
        self.path = data_path
        self.params = loader_params
        train_dataset = SEMG_Dataset(data_path,'train',0)
        val_dataset = SEMG_Dataset(data_path,'val',0)
        test_dataset = SEMG_Dataset(data_path,'test',0)
        self.data_loaders = {'train':data.DataLoader(train_dataset,**loader_params),
                              'val':data.DataLoader(val_dataset,batch_size=loader_params['batch_size'], shuffle=False,num_workers=loader_params['num_workers']),
                              'test':data.DataLoader(test_dataset,batch_size=loader_params['batch_size'], shuffle=False,num_workers=loader_params['num_workers'])}
        self.data_lens = {'train':len(train_dataset),'val':len(val_dataset),'test':len(test_dataset)}

        self.stats = {'train':{'loss': float('+inf'),
                           'model_wt': copy.deepcopy(self.model.state_dict()),
                           'acc': 0,
                           'epoch': 0,'fold':0},
                      'val':{'loss': float('+inf'),
                           'model_wt': copy.deepcopy(self.model.state_dict()),
                           'acc': 0,
                           'epoch': 0,'fold':0}}
        self.loss_hist = {'train':[] ,'val':[]}
        self.acc_hist = {'train':[] ,'val':[]}



    def one_epoch(self,phase):
        '''Iterates through dataset to complete ONE epoch. Steps: pull sample(s) from data loader,
        pass data through model, calculate loss, update weights, take a step for optimizer. Keeps track of
        running loss and running correct classifications.
        ARGS: phase: ('train','val','test') sets whether or not to use training specific
        steps (enabling grad,loss.backward(),etc.)
        RETURNS: running_loss: total loss over all data points
                 cor_classify: number of correctly classified samples'''
        running_loss = 0.0
        cor_classify = 0.0
        i = 0
        for input,label in self.data_loaders[phase]:
            if phase == 'train':
                self.model.train()
                torch.set_grad_enabled(True)
                self.optimizer.zero_grad()
            elif phase == 'val' or phase == 'test':
                self.model.eval()
                torch.set_grad_enabled(False)
            # input.double()
            input = input.to(self.device)
            label = label.to(self.device)
            # print(input.dtype)
            # print(label.dtype)
            output = self.model(input)
            loss = self.criterion(output,label.float()) # for MSE
            # loss = self.criterion(output,torch.argmax(label,dim=1)) # for CrossEntropyLoss
            running_loss += loss.item()
            # return
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
        loss_cnt = 0
        # prev_lr = self.scheduler.get_lr()
        print('Training...\n')
        folds = 1
        for f in range(folds):
            self.data_loaders = {'train':data.DataLoader(SEMG_Dataset(self.path,'train',f),**self.params),
                                  'val':data.DataLoader(SEMG_Dataset(self.path,'val',f),batch_size=self.params['batch_size'], shuffle=False,num_workers=self.params['num_workers']),
                                  'test':data.DataLoader(SEMG_Dataset(self.path,'test',f),batch_size=self.params['batch_size'], shuffle=False,num_workers=self.params['num_workers'])}
            print("Fold {} of {}".format(f+1,folds))
            self.loss_cnt = 0
            self.model.load_state_dict(self.og_wt)
            for epoch in range(1,self.max_epochs+1):
                e_loss, e_classify = self.one_epoch('train')
                e_loss /= self.data_lens['train']
                e_acc = (e_classify/self.data_lens['train'])*100
                print("Epoch: {}/{}\nPhase: Train  Loss: {:.8f}    Accuracy: {:.4f}".format(epoch,self.max_epochs,e_loss,e_acc))
                self.loss_hist['train'].append(e_loss)
                self.acc_hist['train'].append(e_acc)
                if val_train:
                    t_loss, t_acc = self.test(False,epoch)
                    print("Phase: Validation    Loss: {:.8f}    Accuracy: {:.4f}".format(t_loss,t_acc))
                if e_loss < self.stats['train']['loss']:
                    self.stats['train'] = {'loss': e_loss,
                                       'model_wt': copy.deepcopy(self.model.state_dict()),
                                       'acc': e_acc,
                                       'epoch': epoch,
                                       'fold':f}

                '''Add early stopping: if change in loss less than ... x times, stop.
                Useful check if updating properly as well'''
                if abs(e_loss - prev_loss) < 1e-6 or e_loss > prev_loss: self.loss_cnt += 1
                if self.loss_cnt > 20: break
                prev_loss = e_loss

            torch.save(self.stats['train']['model_wt'],self.path+'_{}_{}.pt'.format(f,epoch))

            # if self.scheduler != None:
            #     self.scheduler.step()
            #     if self.scheduler.get_lr() != prev_lr:
            #         torch.save(self.stats['train']['model_wt'],self.path+'_SGD_MSE_{}.pt'.format(epoch))
            #         self.model.load_state_dict(self.og_wt)

        print("Training Summary:")
        print("Best Training epoch was {} of {} in fold {} with Loss: {:.8f}    Accuracy: {:.4f}".format(self.stats['train']['epoch'],epoch,self.stats['train']['fold'],self.stats['train']['loss'],self.stats['train']['acc']))
        print("Best Validation epoch was {} of {} in fold {} with Loss: {:.8f}    Accuracy: {:.4f}".format(self.stats['val']['epoch'],epoch,self.stats['val']['fold'],self.stats['val']['loss'],self.stats['val']['acc']))
        total_time = time.time() - since
        print('Training completed in {:.0f}m {:.0f}s'.format(total_time//60,total_time%60))

    def test(self,use_best_wt,epoch=1,f=0):
        '''Can be used for either validation phase during training or testing on trained model. When one_epoch
        function is called, passes argument to put model in eval mode and disable grad.
        ARGS: use_best_wt: True/False denotes whether its a validation or testing phase.
              epoch: current epoch, used in printing and saving stats for validation Phase
        RETURNS: test_loss: validation or testing epoch loss
                 test_acc: validation or testing epoch accuracy'''
        set = 'val'
        if use_best_wt:
            print("Testing with best weights...")
            self.model.load_state_dict(self.stats['train']['model_wt'])
            set = 'test'

        test_loss, test_correct = self.one_epoch(set)
        test_loss /= self.data_lens[set]
        test_acc = (test_correct/self.data_lens[set])*100
        if not use_best_wt:
            self.loss_hist['val'].append(test_loss)
            self.acc_hist['val'].append(test_acc)

        if test_loss < self.stats['val']['loss'] and not use_best_wt:
            self.stats['val'] = {'loss': test_loss,
                               'model_wt': copy.deepcopy(self.model.state_dict()),
                               'acc': test_acc,
                               'epoch': epoch,
                               'fold':f}
        if not use_best_wt:
            if abs(test_loss - self.prev_t_loss) < 1e-6 or test_loss > self.prev_t_loss: self.loss_cnt += 1
            # if loss_cnt > 5: self.stop == True
            self.prev_t_loss = test_loss
        if use_best_wt:
            print("Test Summary:")
            print("    Loss = {:.8f}".format(test_loss))
            print("    Correct: {}/{}".format(test_correct,self.data_lens[set]))
            print("    Accuracy: {:.4f}".format(test_acc))
        return test_loss, test_acc


    def plot_loss(self,path,save_plt):
        '''Plot training and validation loss for each epoch'''
        path_loss = 'img/'+path+'_loss.png'
        path_acc = 'img/'+path+'_acc.png'
        fig = plt.figure()
        ax = fig.gca()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over {} Epochs'.format(len(self.loss_hist['train'])))
        x = np.arange(1,len(self.loss_hist['train'])+1)
        train, = plt.plot(x,self.loss_hist['train'],'k')
        val, = plt.plot(x,self.loss_hist['val'],'r')
        plt.legend((train,val),('Training','Validation'))
        if save_plt:
            plt.savefig(path_loss,format='png')

        fig = plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over {} Epochs'.format(len(self.acc_hist['train'])))
        train, = plt.plot(x,self.acc_hist['train'],'k')
        val, = plt.plot(x,self.acc_hist['val'],'r')
        plt.legend((train,val),('Training','Validation'))
        if save_plt:
            plt.savefig(path_acc,format='png')


    # def save_model(self,path):
    #     torch.save(nt.stats['train']['model_wt'],path+'_SGD_MSE_reduced.pt')





def main():
    test_only = False
    save_md = True
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))
    ## Initialize model
    # model = Network(6)
    model = Network_enhanced(7)

    ## Initialize hyperparameters and supporting functions
    learning_rate = 0.2
    optimizer = optim.SGD(model.parameters(),lr=learning_rate)
    # optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    criterion = nn.MSELoss(reduction='mean')
    # criterion = nn.CrossEntropyLoss(reduction='sum')
    scheduler = lr_scheduler.StepLR(optimizer,step_size=25,gamma=0.1)

    ## Initialize datasets path and dataloader parameters
    dir = 'nina_data/'
    file = 'all_7C_data_1'
    path = dir+file
    params = {'batch_size': 10, 'shuffle': True,'num_workers': 4}

    ## Initialize network trainer class
    if not test_only:
        og_wts = copy.deepcopy(model.state_dict())
        # count = 0
        # rates = np.logspace(-1.0,-2.0,20)
        # # rates = [0.02]
        # for lr in rates:
        #     model.load_state_dict(og_wts)
        #     optimizer = optim.SGD(model.parameters(),lr=lr)
        #     nt = Trainer(model,optimizer,criterion,device,path,params,epochs=100)
        #     print('\nTrain model with learning rate: {}'.format(lr))
        #     ## Train and test network
        #     nt.train(val_train=True)
        #     tl, ta = nt.test(use_best_wt=True, epoch=1)
        #     if save_md:
        #         torch.save(nt.stats['train']['model_wt'],path+'_{}.pt'.format(count))
        #     # nt.test(use_best_wt=True, epoch=1)
        #     plot_path = file+'_{}'.format(count)
        #     nt.plot_loss(plot_path,save_md)
        #     count += 1
        for red in ['sum','mean']:
            for bts in [2,4,6,8,10]:
                lr = 0.02
                model.load_state_dict(og_wts)
                optimizer = optim.SGD(model.parameters(),lr=lr)
                criterion = nn.MSELoss(reduction=red)
                params = {'batch_size': bts, 'shuffle': True,'num_workers': 4}
                nt = Trainer(model,optimizer,criterion,device,path,params,epochs=100)
                print('\nTrain model with reduction: {}; and batch_size: {}'.format(red,bts))
                ## Train and test network
                nt.train(val_train=True)
                tl, ta = nt.test(use_best_wt=True, epoch=1)
                if save_md:
                    torch.save(nt.stats['train']['model_wt'],path+'_'+red+'_{}.pt'.format(bts))
                # nt.test(use_best_wt=True, epoch=1)
                plot_path = file+'_'+red+'_{}'.format(bts)
                nt.plot_loss(plot_path,save_md)


        # plt.show(block=True)

    elif test_only:
        # model = Network_enhanced(7)
        nt = Trainer(model,optimizer,criterion,device,path,params,epochs=1)
        nt.stats['train']['model_wt'] = torch.load(path+'.pt')
        tl,ta = nt.test(use_best_wt=True,epoch=1)



if __name__ == '__main__':
    main()
