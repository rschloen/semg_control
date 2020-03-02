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
from semg_network import Network, Network_enhanced
from data_loader import SEMG_Dataset



class Trainer():
    def __init__(self,model,optimizer,criterion,device,data_path,loader_params,file=None,scheduler=None,epochs=25,early_stop=True):
        self.model = model.to(device)#.float()
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.f = file
        self.max_epochs = epochs
        self.early_stop = early_stop
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
        self.wt_hist = {'train':[] ,'val':[]}




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
            elif phase == 'test' or phase == 'val':
                self.model.eval()
                torch.set_grad_enabled(False)
            # input.double()
            input = input.to(self.device)
            label = label.to(self.device)
            output = self.model(input)
            loss = self.criterion(output,label.float()) # for MSE
            # loss = self.criterion(output,torch.argmax(label,dim=1)) # for CrossEntropyLoss
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
        # loss_cnt = 0
        # prev_lr = self.scheduler.get_lr()
        print('Training...')
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
                # print("Epoch: {}/{}\nPhase: Train  Loss: {:.8f}    Accuracy: {:.4f}".format(epoch,self.max_epochs,e_loss,e_acc))
                self.loss_hist['train'].append(e_loss)
                self.acc_hist['train'].append(e_acc)
                if val_train:
                    t_loss, t_acc = self.test(False,epoch)
                    # print("Phase: Validation    Loss: {:.8f}    Accuracy: {:.4f}".format(t_loss,t_acc))
                # if e_loss < self.stats['train']['loss']:
                #     self.stats['train'] = {'loss': e_loss,
                #                        'model_wt': copy.deepcopy(self.model.state_dict()),
                #                        'acc': e_acc,
                #                        'epoch': epoch,
                #                        'fold':f}

                if self.early_stop:
                    '''Add early stopping: if change in loss less than ... x times, stop.
                    Useful check if updating properly as well'''
                    if abs(e_loss - prev_loss) < 1e-8: self.loss_cnt += 1 #or e_loss > prev_loss
                    if self.loss_cnt > 20: break
                    prev_loss = e_loss
                self.wt_hist['train'].append(copy.deepcopy(self.model.state_dict()))

                self.progress_bar(epoch)
            # torch.save(self.stats['train']['model_wt'],self.path+'_{}_{}.pt'.format(f,epoch))

            # if self.scheduler != None:
            #     self.scheduler.step()
            #     if self.scheduler.get_lr() != prev_lr:
            #         torch.save(self.stats['train']['model_wt'],self.path+'_SGD_MSE_{}.pt'.format(epoch))
            #         self.model.load_state_dict(self.og_wt)

        print('\n')
        tl_ind,vl_ind = np.argmin(self.loss_hist['train']),np.argmin(self.loss_hist['val'])
        # print("Best Training epoch was {} of {} in fold {} with Loss: {:.8f}    Accuracy: {:.4f}".format(tl_ind,epoch,0,self.loss_hist['train'][tl_ind],self.acc_hist['train'][tl_ind]))
        # print("Best Validation epoch was {} of {} in fold {} with Loss: {:.8f}    Accuracy: {:.4f}".format(vl_ind,epoch,0,self.loss_hist['val'][vl_ind],self.acc_hist['val'][vl_ind]))
        if self.f != None:
            self.f.write("Training Summary:\n")
            self.f.write("Best Training epoch was {} of {} in fold {} with Loss: {:.8f}    Accuracy: {:.4f}\n".format(tl_ind,epoch,0,self.loss_hist['train'][tl_ind],self.acc_hist['train'][tl_ind]))
            self.f.write("Best Validation epoch was {} of {} in fold {} with Loss: {:.8f}    Accuracy: {:.4f}\n".format(vl_ind,epoch,0,self.loss_hist['val'][vl_ind],self.acc_hist['val'][vl_ind]))
            total_time = time.time() - since
            self.f.write('Training completed in {:.0f}m {:.0f}s\n'.format(total_time//60,total_time%60))
        print('Training completed in {:.0f}m {:.0f}s\n'.format(total_time//60,total_time%60))


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
            if len(self.wt_hist['val']) != 0:
                self.model.load_state_dict(self.wt_hist['val'][np.argmin(self.loss_hist['val'])])
                # print('history')
            else:
                self.model.load_state_dict(self.stats['val']['model_wt'])
            set = 'test'

        test_loss, test_correct = self.one_epoch(set)
        test_loss /= self.data_lens[set]
        test_acc = (test_correct/self.data_lens[set])*100
        if not use_best_wt:
            self.loss_hist['val'].append(test_loss)
            self.acc_hist['val'].append(test_acc)
            self.wt_hist['val'].append(copy.deepcopy(self.model.state_dict()))

        # if test_loss < self.stats['val']['loss'] and not use_best_wt:
        #     self.stats['val'] = {'loss': test_loss,
        #                        'model_wt': copy.deepcopy(self.model.state_dict()),
        #                        'acc': test_acc,
        #                        'epoch': epoch,
        #                        'fold':f}

        if not use_best_wt and self.early_stop:
            if abs(test_loss - self.prev_t_loss) < 1e-6: self.loss_cnt += 1
            # if loss_cnt > 5: self.stop == True
            self.prev_t_loss = test_loss
        if use_best_wt:
            print("Test Summary:")
            print("    Loss = {:.8f}".format(test_loss))
            print("    Correct: {}/{}".format(test_correct,self.data_lens[set]))
            print("    Accuracy: {:.4f}".format(test_acc))
            if self.f != None:
                self.f.write("Test Summary:\n")
                self.f.write("    Loss = {:.8f}\n".format(test_loss))
                self.f.write("    Correct: {}/{}\n".format(test_correct,self.data_lens[set]))
                self.f.write("    Accuracy: {:.4f}\n".format(test_acc))
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
        fig.clf()
        fig = plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over {} Epochs'.format(len(self.acc_hist['train'])))
        train, = plt.plot(x,self.acc_hist['train'],'k')
        val, = plt.plot(x,self.acc_hist['val'],'r')
        plt.legend((train,val),('Training','Validation'))
        if save_plt:
            plt.savefig(path_acc,format='png')
        fig.clf()
        plt.close('all')

    def progress_bar(self,cur):
        bar_len = 25
        percent = cur/self.max_epochs
        fill = round(percent*bar_len)
        bar = '['+'#'*fill+'-'*(bar_len-fill)+']  [{:.2f}%]'.format(percent*100)

        print('\r'+bar,end='')


def hyperparam_selection(test_only,save_md):

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
    file = 'all_7C_data_3'
    path = dir+file
    params = {'batch_size': 100, 'shuffle': True,'num_workers': 4}

    # Initialize network trainer class
    if not test_only:
        since = time.time()
        # f = open(path+'_stats.txt','w')
        f = open(path+'_stats_adamw_best.txt','w')
        og_wts = copy.deepcopy(model.state_dict())
        rates = np.logspace(-2.0,-4.0,20)
        mom = np.linspace(0.9,0.99,10) #best value was .092
        decay = np.logspace(-1,-4,20) #best was
        m = mom[1]
        # lr = rates[2] # for sdg
        lr = rates[10] # for adamw
        dec = decay[2] # for adamw
        print('\nTrain model with AdamW')
        f.write('\nTrain model with AdamW\n')
        # f.write('Momentum: {}\n'.format(mom))
        count = 0
        all_tl = []
        f.write('Learning Rate: {}\n'.format(lr))
        # for dec in decay:
            # model.load_state_dict(og_wts)
            # if op == 0:
            #     optimizer = optim.SGD(model.parameters(),lr=lr,momentum=m)
            #     print('\nTrain model with SGD')
            #     f.write('\nTrain model with SGD\n')
            #
            # elif op == 1:
            #     optimizer = optim.Adam(model.parameters())
            #     print('\nTrain model with Adam')
            #     f.write('\nTrain model with Adam\n')
        optimizer = optim.AdamW(model.parameters(),lr=lr,weight_decay=dec)
        nt = Trainer(model,optimizer,criterion,device,path,params,f,epochs=500)
        print('\nTrain model with decay: {}'.format(dec))
        f.write('Weight Decay: {}\n'.format(dec))


        ## Train and test network
        nt.train(val_train=True)
        tl, ta = nt.test(use_best_wt=True, epoch=1)
        # all_tl.append(tl)
        # f.write('For momentum {}: \nBest val loss: {:.8f}; Best val accuracy: {:.4f}\n\n'.format(m,np.min(nt.loss_hist['val']),np.max(nt.acc_hist['val'])))
        if save_md:
            torch.save(nt.wt_hist['val'][np.argmin(nt.loss_hist['val'])],path+'_adamw_best.pt')

        plot_path = file+'_adamw_best'
        nt.plot_loss(plot_path,save_md)
        count += 1
        # print("Best test: {}, with loss: {:.8f}. Therefore best weight_decay is {}".format(np.argmin(all_tl),np.min(all_tl),decay[np.argmin(all_tl)]))
        # f.write("Best test: {}, with loss: {:.8f}. Therefore best weight_decay is {}".format(np.argmin(all_tl),np.min(all_tl),decay[np.argmin(all_tl)]))


        # f.write('Training with SGD:\n')
        # print('Training with SGD:')
        # for m in [0.91]:
        #     for dec in [0,1e-5]:
        #         for nest in [True,False]:
        #             # lr = 0.02
        #             print('Momentum: {}, Weight Decay: {}, Nesterov? {}\n'.format(m,dec,nest))
        #             f.write('Momentum: {}, Weight Decay: {}, Nesterov? {}'.format(m,dec,nest))
        #             model.load_state_dict(og_wts)
        #             optimizer = optim.SGD(model.parameters(),lr=lr,momentum=m,weight_decay=dec,)
        #             # criterion = nn.MSELoss(reduction=red)
        #             # params = {'batch_size': bts, 'shuffle': True,'num_workers': 4}
        #             nt = Trainer(model,optimizer,criterion,device,path,params,f,epochs=100)
        #
        #             ## Train and test network
        #             nt.train(val_train=True)
        #             tl, ta = nt.test(use_best_wt=True, epoch=1)
        #             if save_md:
        #                 torch.save(nt.stats['train']['model_wt'],path+'_'+red+'_{}.pt'.format(bts))
        #             # nt.test(use_best_wt=True, epoch=1)
        #             plot_path = file+'_SGD_m{}_d{}_{}'.format(m,dec,nest)
        #             nt.plot_loss(plot_path,True)


        # plt.show(block=True)

    elif test_only:
        path = 'myo_rec_data/win_JRS_7C_2'
        model_path = 'myo_rec_data/win_JRS_7C_2_tran3'
        model = Network_enhanced(7)
        nt = Trainer(model,optimizer,criterion,device,path,params,epochs=1)
        nt.stats['val']['model_wt'] = torch.load(model_path+'.pt',map_location=torch.device('cpu'))
        tl,ta = nt.test(use_best_wt=True,epoch=1)

    total_time = time.time() - since
    print('Time to complete: {:.0f}m {:.0f}s'.format(total_time//60,total_time%60))
    f.close()


def best_model_params(model):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))
    dir = 'nina_data/'
    file = 'all_7C_data_3'
    path = dir+file
    f = open(path+'_stats_adamw_best.txt','w')
    params = {'batch_size': 100, 'shuffle': True,'num_workers': 4}
    criterion = nn.MSELoss(reduction='mean')
    rates = np.logspace(-2.0,-4.0,20)
    # mom = np.linspace(0.9,0.99,10) #best value was .092
    decay = np.logspace(-1,-4,20) #best was
    # m = mom[1]
    # lr = rates[2] # for sdg
    lr = rates[10] # for adamw
    dec = decay[2] # for adamw
    f.write('AdamW:\nLearning Rate: {}, Momentum: Defualt, Weight Decay:{}\n'.format(lr,dec))
    # f.write('Repeat layer {} times\n'.format(model.repeat))
    optimizer = optim.AdamW(model.parameters(),lr=lr,weight_decay=dec)
    nt = Trainer(model,optimizer,criterion,device,path,params,f,epochs=100)
    return nt



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test','-t',help='((True or true or t)/False) Only test model using best weights')
    parser.add_argument('--save','-s',help='((True or true or t)/False) Save the best model weights and generated plots')
    args=parser.parse_args()

    if args.test == 'True' or args.test == 'true' or args.test == 't':
        test_only = True
    else:
        test_only = False

    if args.save == 'True' or args.save == 'true' or args.save == 't':
        save_md = True
    else:
        save_md = False
    if save_md:
        print("Saving model")

    all_tl = []
    # for i in range(5):
    model = Network_enhanced(7)
    net = best_model_params(model)
    # print('Repeat layer {} times\n'.format(model.repeat))
    ## Train and test network
    net.train(val_train=True)
    tl, ta = net.test(use_best_wt=True, epoch=1)
    if save_md:
        torch.save(nt.wt_hist['val'][np.argmin(nt.loss_hist['val'])],path+'_adamw_best.pt')

    plot_path = file+'_adamw_best'
    nt.plot_loss(plot_path,save_md)
    # all_tl.append(tl)
    # print("Best test: {}, with loss: {:.8f}. Therefore best number of repeatitions is {}".format(np.argmin(all_tl),np.min(all_tl),np.argmin(all_tl)))
    # net.f.write("Best test: {}, with loss: {:.8f}. Therefore best number of repeatitions is {}".format(np.argmin(all_tl),np.min(all_tl),np.argmin(all_tl)))




if __name__ == '__main__':
    main()
