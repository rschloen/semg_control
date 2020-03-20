#!/usr/bin/env python3

from myo_raw import MyoRaw
import torch
import numpy as np
from semg_network import Network_XL, Network_enhanced
from train_net import Trainer, best_model_params
import time, sys, copy, csv, argparse
from statistics import mode
from transfer_learn import circle_shift, most_active
from lstm_emg import LSTM_Net



class RealTime():
    """This class is used for realtime prediction of gestures from emg signals detected by the MYO armband.
    Inputs: model: neural network model used to predict gestures from inputted emg signals
            m: MyoRaw class object used for comminication with the Myo armband
            start: Most active channel of Myo armband at current placement, passed if known, otherwise found using calibrate function
            goal: Most active channel of first recording of dataset. Channel new data is calibrated with respect to."""
    def __init__(self,model,m,start=0,goal=0):
        super(RealTime, self).__init__()
        self.mean_emg,self.std_emg = np.loadtxt('myo_rec_data/win_JRS_7C_comb7_shifted_stats.txt',delimiter=',')
        self.model = model
        self.emg_array = []
        self.pred_array = []
        self.last_pred = 0
        self.pred_cnt = 0
        self.myo = m
        self.start_ch = start
        self.goal_ch = goal
        self.cal_array = []

    def start_pred(self):
        '''Initialize and connect to MYO armband, then disconnect when interrupted'''
        self.myo.add_emg_handler(self.proc_emg)
        self.myo.connect()
        try:
            while True:
                self.myo.run(1)
        except KeyboardInterrupt:
            print('\n')
            self.myo.disconnect()

    def proc_emg(self,emg, moving, times=[]):
        '''Takes sampled emg, shifts the channels per the calibration, and builds a 260ms (52 samples)
        input array that is passed through the trained model to produce the predicted gesture. Prediciton
        is only reported when it differs from the previously reported prediction a sufficent number of times.
        ARGS: emg: list containing sample from each of the 8 channels
              moving and times may not be needed'''
        since = time.time()
        self.emg_array.append(list((emg - self.mean_emg)/(self.std_emg))) #normalization
        self.emg_array = circle_shift(np.array(self.emg_array),self.start_ch,self.goal_ch).tolist() #calibration
        if len(self.emg_array) == 52:
            input = torch.Tensor(self.emg_array)
            input = input.view(1,1,input.shape[0],input.shape[1])
            pred = torch.argmax(self.model(input)).item() #prediction
            self.pred_array.append(pred)
            if len(self.pred_array) == 10:
                try:
                    if mode(self.pred_array) != self.last_pred:
                        self.pred_cnt += 1
                    if self.pred_cnt > 15:
                        # print('Predicted Gesture: {}'.format(mode(self.pred_array)))
                        self.display_gest(mode(self.pred_array))
                        self.last_pred = mode(self.pred_array)
                        self.pred_cnt = 0
                    self.pred_array.clear()
                except:
                    self.pred_array.clear()
            self.emg_array.pop(0)

    def display_gest(self,pred):
        '''Prints text corresponding to passed prediction (pred)'''
        if pred == 0:
            print('\rPredicted Gesture: OPEN HAND     ',end='')
        elif pred == 1:
            print('\rPredicted Gesture: INDEX FINGER  ',end='')
        elif pred == 2:
            print('\rPredicted Gesture: MIDDLE FINGER ',end='')
        elif pred == 3:
            print('\rPredicted Gesture: RING FINGER   ',end='')
        elif pred == 4:
            print('\rPredicted Gesture: PINKY FINGER  ',end='')
        elif pred == 5:
            print('\rPredicted Gesture: THUMB         ',end='')
        elif pred == 6:
            print('\rPredicted Gesture: FIST          ',end='')


    def calibrate_emg(self,emg,moving,times=[]):
        '''Saves emg to be used in calibration'''
        self.cal_array.append(emg)

    def calibrate(self):
        '''Records emg for set duration, or until interrupted, that is then used to calulate the most active
        channel for the purposes of calibrating the emg'''
        print('Calibrating....')
        self.myo.add_emg_handler(self.calibrate_emg)
        self.myo.connect()

        i = 0
        try:
            while i < 10000:
                self.myo.run(1)
                i += 1
        except KeyboardInterrupt:
             self.myo.disconnect()

        self.myo.disconnect()
        # print(self.cal_array)
        self.start_ch = most_active(self.cal_array)
        print('Most active channel is {}'.format(self.start_ch))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_channel','-s',help='(int) Calibrated start channel')
    parser.add_argument('--goal_channel','-g',help='(int) Calibrated target channel')
    args=parser.parse_args()

    m = MyoRaw(None)
    model = Network_XL(7)
    model_path = 'myo_rec_data/win_JRS_7C_comb7_shifted_XL_cross_tran_final2.pt'
    path = 'myo_rec_data/win_JRS_7C_comb7_shifted'


    model.load_state_dict(torch.load(model_path,map_location='cpu'))
    model.eval()
    first_activity = most_active('myo_rec_data/raw_emg_JRS_7C_1.csv') #should be 3
    rt = RealTime(model,m,int(args.start_channel),first_activity)
    cal = input('Begin calibration recording? ')
    if cal == 'y':
        rt.calibrate()

    start = input('Begin real time prediction? ')
    if start == 'y':
        rt.start_pred()
