#!/usr/bin/env python3

from myo_raw import MyoRaw
import torch
import numpy as np
from semg_network import Network, Network_enhanced
from train_net import Trainer
import time







if __name__ == '__main__':
    m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)
    model = Network_enhanced(7)
    path = 'myo_rec_data/win_JRS_7C_2_tran3.pt'
    model.load_state_dict(path)
    model.eval()
    emg_array = []

    def proc_emg(emg, moving, times=[]):
        since = time.time()
        emg_array.append([emg])
        if len(emg_array) == 52:
            input = torch.Tensor(emg_array)
            input = input.view(1,1,input.shape[0],input.shape[1])
            output = model(input)
            pred = torch.argmax(output).item()
            print('Predicted Gesture: {}'.format(pred))
            emg_array.pop(0)
            print('Time for one prediction: {}'.format(time.time()-since))





    m.add_emg_handler(proc_emg)
    m.connect()

    m.add_arm_handler(lambda arm, xdir: print('arm', arm, 'xdir', xdir))
    m.add_pose_handler(lambda p: print('pose', p))

    try:
        while True:
            m.run(1)
    except KeyboardInterrupt:
         m.disconnect()
