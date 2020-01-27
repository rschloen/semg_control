#!/usr/bin/env python

import myo_raw
import csv
import matplotlib.pyplot as plt
import numpy as np



def ReadData(file):
    with open(file) as csv_file:
        reader = csv.reader(csv_file,delimiter=',')
        count = 0
        for row in reader:
            for i in range(len(row)):
                row[i] = float(row[i])
            if count == 0:
                array = np.array(row)
            else:
                array = np.vstack((array,row))
            count += 1
    return array

def plot_emg(data):
    fs = 200. #200Hz
    time = len(data)/fs #samples/(sample/second)
    x = np.linspace(0,time,len(data))
    fig = plt.figure()
    fig.text(0.5,0.04,'Time (seconds)',ha='center')
    fig.text(0.04,0.5,'au(-128,128)',va='center',rotation='vertical')
    plt.xlim([0,time])

    ax1 = plt.subplot(8,1,1,title='Channel 1')
    plt.plot(x,data[:,0],'k')
    for i in range(1,len(data[0])):
        plt.subplot(8,1,i+1,sharex=ax1,title='Channel '+str(i+1))
        plt.plot(x,data[:,i],'k')
    plt.show()


if __name__ == '__main__':
    # emg_array = np.array([])
    fs = 200. #200Hz
    emg_array = ReadData('raw_emg.csv')
    # print('rows:'+str(emg_array.shape[0])+' cols:'+str(emg_array.shape[1]))
    # print(emg_array[:10,0])
    time = len(emg_array)/fs #samples/(sample/second)
    # print(time)
    plot_emg(emg_array)
