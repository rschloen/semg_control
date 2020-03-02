#!/usr/bin/env python3

from myo_raw import MyoRaw
import torch
import numpy as np
from semg_network import Network, Network_enhanced
from train_net import Trainer
import time
import sys
import copy
import csv






if __name__ == '__main__':
    # with open('myo_rec_data/win_JRS_7C_2_stats.csv','w') as real_time:#, open('real_time_gest.csv','w') as real_gest:
    #     stat_writer = csv.writer(real_time, delimiter=',')
    #     emg_data = np.genfromtxt('myo_rec_data/raw_emg_JRS_7C_2.csv',delimiter=',')
    #     mean_emg = np.mean(emg_data)
    #     std_emg = np.std(emg_data)
    #     stat_writer.writerow((mean_emg,std_emg))
    mean_emg,std_emg = np.loadtxt('myo_rec_data/win_JRS_7C_2_stats.csv',delimiter=',')
    # m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)
    model = Network_enhanced(7)
    path = 'myo_rec_data/win_JRS_7C_2_tran3.pt'
    model.load_state_dict(torch.load(path,map_location='cpu'))
    model.eval()
    global emg_array, cor
    cor = 0
    emg_array = []
    # with open('real_time_rec.csv','w') as real_time, open('real_time_gest.csv','w') as real_gest:
    #     emg_writer = csv.writer(real_time, delimiter=',')
    #     gest_writer = csv.writer(real_gest,delimiter=',')


    def proc_emg(emg, moving, times=[]):
        global emg_array, cor
        since = time.time()
        # print(len(emg))
        # print(emg)
        # print((emg - mean_emg)/(std_emg))
        # emg_writer.writerow(emg)
        emg_array.append(list((emg - mean_emg)/(std_emg)))
        # gest_writer.writerow((0,))
        # print(emg_array)
        # print(len(emg_array[0]))
        # print(len(emg_array[1]))
        if len(emg_array) == 10:
            # emg_array = emg_array.tolist()
            input = torch.Tensor(emg_array)
            input = input.view(1,1,input.shape[0],input.shape[1])
            print(input)
            # output = model(input)
            # pred = torch.argmax(output).item()
            # if pred == moving:
            #     cor += 1
            # print('Predicted Gesture: {}'.format(pred))
            emg_array.pop(0)
            # print(len(emg_array))
            # print('Time for one prediction: {}'.format(time.time()-since))

    data = np.loadtxt('myo_rec_data/raw_emg_JRS_7C_2.csv',delimiter=',')
    label = np.loadtxt('myo_rec_data/gesture_JRS_7C_2.csv',delimiter=',')
    # print(label)
    # win_data = np.load('myo_rec_data/win_JRS_7C_2.npy',allow_pickle='True').item()
    # first_train = win_data['train'][0][0]
    # first_val = win_data['val'][0][0]
    # first_test = win_data['test'][0][0]
    # print('Train')
    # print(first_train)
    # print('val')
    # print(first_val)
    # print('Test')
    # print(first_test)
    print((data[:10] - mean_emg)/(std_emg))
    print((data[1:11] - mean_emg)/(std_emg))
    for i in range(11):
        # print(data[i])
        proc_emg(data[i],label[i])
    print('Accuracy: {}%'.format(cor/len(data)*100))

    # m.add_emg_handler(proc_emg)
    # m.connect()
    #
    # m.add_arm_handler(lambda arm, xdir: print('arm', arm, 'xdir', xdir))
    # m.add_pose_handler(lambda p: print('pose', p))
    #
    # try:
    #     while True:
    #         m.run(1)
    # except KeyboardInterrupt:
    #      m.disconnect()
