#!/usr/bin/env python3

import myo_raw
import csv
import matplotlib.pyplot as plt
import numpy as np


def plot_emg(data,gesture,path):
    fs = 200. #200Hz
    time = len(data)/fs #samples/(sample/second)
    x = np.linspace(0,time,len(data))
    fig = plt.figure()
    fig.text(0.5,0.04,'Time (seconds)',ha='center')
    fig.text(0.04,0.5,'au(-128,128)',va='center',rotation='vertical')
    fig.text(0.5,0.95,'File set: '+path,ha='center')
    plt.xlim([0,time])

    ax1 = plt.subplot(9,1,1,title='Channel 1')
    # if len(data[0]) == 9 and data
    plt.plot(x,data[:,0])
    for i in range(1,8):
        plt.subplot(9,1,i+1,sharex=ax1,title='Channel '+str(i+1))
        plt.plot(x,data[:,i])
    plt.subplot(9,1,9,sharex=ax1,title='Gesture')
    plt.plot(x,gesture,'k')
    plt.show()

def MAV(data):
    array = np.zeros(8)
    for i in range(8):
        array[i] = np.sum(np.abs(data[:,i]))/len(data)
    return array

def RMS(data):
    array = np.zeros(8)
    for i in range(8):
        array[i] = np.sqrt(np.sum(data[:,i]**2)/len(data))
    return array


if __name__ == '__main__':
    # emg_array = np.array([])
    fs = 200. #200Hz
    # test_array = np.array([[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8]])
    path = 'real_time'
    emg_array = np.loadtxt('real_time_rec.csv',delimiter=',')
    gesture_array = np.loadtxt('real_time_gest.csv',delimiter=',')
    # print(RMS(emg_array[:100]))
    # print(MAV(emg_array[:100]))
    time = len(emg_array)/fs #samples/(sample/second)
    # print(time)
    plot_emg(emg_array,gesture_array,path=path)
