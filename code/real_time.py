#!/usr/bin/env python3

from myo_raw import MyoRaw
import torch
import numpy as np
from semg_network import Network_XL, Network_enhanced
from train_net import Trainer, best_model_params
import time, sys, copy, csv, argparse
from statistics import mode
from transfer_learn import circle_shift


class RealTime():
    """docstring for ."""
    def __init__(self,model,m,start=0,goal=0):
        super(RealTime, self).__init__()
        self.mean_emg,self.std_emg = np.loadtxt('myo_rec_data/win_JRS_7C_comb7_shifted_stats.txt',delimiter=',')
        self.model = model
        self.emg_array = []
        self.pred_array = []
        self.myo = m
        self.start_ch = start
        self.goal_ch = goal

    def start_pred(self):
        self.myo.add_emg_handler(self.proc_emg)
        self.myo.connect()

        # self.myo.add_arm_handler(lambda arm, xdir: print('arm', arm, 'xdir', xdir))
        # self.m.add_pose_handler(lambda p: print('pose', p))

        try:
            while True:
                self.myo.run(1)
        except KeyboardInterrupt:
             self.myo.disconnect()

    def proc_emg(self,emg, moving, times=[]):
        # global emg_array, cor, prev_array
        start_ch = 3
        goal_ch = 3
        since = time.time()
        self.emg_array.append(list((emg - self.mean_emg)/(self.std_emg)))
        self.emg_array = circle_shift(np.array(self.emg_array),self.start_ch,self.goal_ch).tolist()
        if len(self.emg_array) == 52:

            input = torch.Tensor(self.emg_array)
            input = input.view(1,1,input.shape[0],input.shape[1])

            pred = torch.argmax(self.model(input)).item()
            self.pred_array.append(pred)
            if len(self.pred_array) == 10:
                try:
                    print('Predicted Gesture: {}'.format(mode(self.pred_array)))
                    self.pred_array.clear()
                except:
                    self.pred_array.clear()

            self.emg_array.pop(0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_channel','-s',help='(int) Calibrated start channel')
    parser.add_argument('--goal_channel','-g',help='(int) Calibrated target channel')
    args=parser.parse_args()

    # with open('nina_data/all_7C_data_comb_abs_stats.txt','w') as real_time:#, open('real_time_gest.csv','w') as real_gest:
    #     stat_writer = csv.writer(real_time, delimiter=',')
    #     emg_data = np.loadtxt('myo_rec_data/raw_emg_JRS_7C_testing.csv',delimiter=',')
    #     mean_emg = np.mean(emg_data)
    #     std_emg = np.std(emg_data)
    #     stat_writer.writerow((mean_emg,std_emg))
    mean_emg,std_emg = np.loadtxt('myo_rec_data/win_JRS_7C_comb7_shifted_stats.txt',delimiter=',')
    m = MyoRaw(None)
    # model = Network_enhanced(7)
    model = Network_XL(7)
    model_path = 'myo_rec_data/win_JRS_7C_comb6_shifted_XL_tran1.pt'
    # model_path = 'myo_rec_data/win_JRS_7C_comb3_adamw_best_tran4.pt'
    # path = 'myo_rec_data/win_JRS_7C_testing'
    path = 'myo_rec_data/win_JRS_7C_comb7_shifted'

    nt = best_model_params(model,path)
    # nt.max_epochs = 1;
    nt.stats['val']['model_wt'] = torch.load(model_path,map_location=torch.device('cpu'))
    tl,ta = nt.test(use_best_wt=True,epoch=1)


    model.load_state_dict(torch.load(model_path,map_location='cpu'))
    model.eval()
    rt = RealTime(model,m,int(args.start_channel),int(args.goal_channel))

    start = input('Begin real time prediction? ')
    if start == 'y':
        rt.start_pred()
