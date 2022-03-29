# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:54:55 2022

@author: David
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio

'''
Loading EMG data
'''

#loads matlab data into an array
mat = spio.loadmat('S11_E1_A1.mat', squeeze_me=True)
emg = np.array(mat['emg'])
stimulus = np.array(mat['stimulus'])
index = np.where(stimulus==1) #finds time index of pose 1
emg_pose_1 = emg[index]
#for more electrodes
emg_labelled = [emg[np.where(stimulus==i)] for i in range(2)]
no_electrodes = np.shape(emg_pose_1)[1] #number of electrodes

sampling_rate = 2000 #Hz
time_pose_1 = len(emg_pose_1)/sampling_rate #data collection time interval in seconds


def plot_EMG(emg_data, no_electrodes, pose='x'):
    '''
    Input:
        -emg data (for a single stimulus, e.g. hand pose)
        -pose/hand movement id
    Output:
        -plots EMG response for all electrodes
    Comment:
        -now only works for more than 1 electrode (otherwise indexing breaks down)
    '''
    fig, axs = plt.subplots(nrows=no_electrodes, ncols=1, figsize=(10,20), sharex=True)
    fig.suptitle('EMG for pose {}'.format(pose), size='x-large', weight = 'bold')
    fig.tight_layout(rect=[0.05, 0.03, 1, 0.97]) #[left, bottom, right, top] in normalized (0, 1) figure coordinates 
    for electrode in range(no_electrodes):#for all electrodes
        axs[electrode].plot(np.linspace(0,time_pose_1,len(emg_data)), emg_data[:,electrode]*1000000, color='black') #in microVolts
        axs[electrode].set_title('Electrode no. {}'.format(electrode))
        
    fig.text(0.01, 0.5, 'Voltage (\u03BCV)', va='center', rotation='vertical')
    plt.xlabel('Time (s)')
    # fig.savefig('EMG_example.svg', format='svg', dpi=1200)

# plot_EMG(emg_pose_1, no_electrodes=no_electrodes, pose='1')


'''
Spike Encoding
-temporal contrast methods (TBR, SF, MW)
'''
def temporal_contrast_encod(data_sample, sampling_rate, f=1, refractory=5):
    '''
    Input:
        - EMG data sample, sampling rate of EMG electrodes, 'f' factor for threshold tuning, refractory period in ms
    Output:
        - arrays of UP and DOWN spike times (size=dependt. on no. of spikes)
        - arrays of spike indeces w.r.t. to the data_sample array (i.e., at what sample point does spike occur)
        - spike data (size=len(data_sample, values ={-1, 0, 1})
        - threshold (for signal reconstruction)
    Comment:
        - converts EMG data into spike trains using the temporal contrast method (also called threshold-based representation)
    '''
    refractory_samples = round((refractory/1000)*sampling_rate) #number of samples corresponding to a refractory period
    # plot_EMG(data_sample, no_electrodes=2, pose='1')
    #save delta values between consecutive samples
    diff = np.zeros(len(data_sample))
    #determining threshold (V_th) based on mean and std of data
    for t in range(len(data_sample)-1):
        diff[t] = data_sample[t+1]-data_sample[t]
    V_th = np.mean(diff)+f*np.std(diff) #negative threshold [uV]

    UP_spikes = []
    DOWN_spikes = []
    spikes = np.zeros(len(data_sample)) #both UP and DOWN
    it = iter(range(len(data_sample)-1)) #set up an iterator
    #extracting spikes based on threshold
    for t in it:
        delta = data_sample[t+1]-data_sample[t]
        if delta > V_th:
            UP_spikes.append(t)
            spikes[t] = 1
            #enforce refractory period by skipping over samples
            for _ in range(refractory_samples):
                try:
                    t = next(it)
                except: #to prevent code from stopping when t reaches end of iterator
                    break
        if delta < -V_th:
            DOWN_spikes.append(t)
            spikes[t] = -1
            #enforce refractory period by skipping over samples
            for _ in range(refractory_samples):
                try:
                    t = next(it)
                except: #to prevent code from stopping when t reaches end of iterator
                    break
    
    #converting from sample indeces to spike times
    UP_spike_times = np.array(UP_spikes)/sampling_rate
    DOWN_spike_times = np.array(DOWN_spikes)/sampling_rate
    
    
    return UP_spike_times, DOWN_spike_times, UP_spikes, DOWN_spikes, spikes, V_th

def SF_encod(data_sample, sampling_rate, f=1, refractory=5):
    '''
    Input:
        - EMG data sample, sampling rate of EMG electrodes, 'f' factor for threshold tuning, refractory period in ms
    Output:
        - arrays of UP and DOWN spike times (size=dependt. on no. of spikes)
        - arrays of spike indeces w.r.t. to the data_sample array (i.e., at what sample point does spike occur)
        - spike data (size=len(data_sample, values ={-1, 0, 1})
        - threshold (for signal reconstruction)
    Comment:
        - converts EMG data into spike trains using the Step-Forward (SF) Encoding
    '''
    refractory_samples = round((refractory/1000)*sampling_rate) #number of samples corresponding to a refractory period
    # plot_EMG(data_sample, no_electrodes=2, pose='1')
    #save delta values between consecutive samples
    diff = np.zeros(len(data_sample))
    #determining threshold (V_th) based on mean and std of data
    for t in range(len(data_sample)-1):
        diff[t] = data_sample[t+1]-data_sample[t]
    V_th = np.mean(diff)+f*np.std(diff) #negative threshold [uV]
    
    initial = data_sample[0]
    spikes = np.zeros(len(data_sample))
    baseline = initial
    UP_spikes = []
    DOWN_spikes = []
    it = iter(range(1,len(data_sample)-1)) #set up an iterator
    #extracting spikes based on threshold
    for t in it:
        if data_sample[t] > baseline + V_th:
            spikes[t] = 1
            UP_spikes.append(t)
            baseline += V_th
            #enforce refractory period by skipping over samples
            for _ in range(refractory_samples):
                try:
                    t = next(it)
                except: #to prevent code from stopping when t reaches end of iterator
                    break
        elif data_sample[t] < baseline - V_th:
            spikes[t] = -1
            DOWN_spikes.append(t)
            baseline -= V_th
            for _ in range(refractory_samples):
                try:
                    t = next(it)
                except: #to prevent code from stopping when t reaches end of iterator
                    break
                
    #converting from sample indeces to spike times
    UP_spike_times = np.array(UP_spikes)/sampling_rate
    DOWN_spike_times = np.array(DOWN_spikes)/sampling_rate
    
    return UP_spike_times, DOWN_spike_times, UP_spikes, DOWN_spikes, spikes, V_th


def signal_reconstruction(data_sample, spikes, V_th):
    '''
    Input:
        - array of data, spike data (size=len(data_sample, values ={-1, 0, 1}), threshold
    Output:
        - array of reconstructed data from spikes using inverse process of decoding, applicable to TBR, SF & MW
    '''
    #temporal contrast signal reconstruction
    reconstr_data = np.zeros(len(spikes))
    reconstr_data[0]= data_sample[0]
    for t in range(1,len(spikes)):
        if spikes[t]==1:
            reconstr_data[t] = reconstr_data[t-1] + V_th
        elif spikes[t]==-1:
            reconstr_data[t] = reconstr_data[t-1] - V_th
        else:
            reconstr_data[t] = reconstr_data[t-1]
    return reconstr_data
    
def plot_encoding_bi(data_sample, sampling_rate, UP_spike_times, DOWN_spike_times):
    '''
    Input:
        -EMG data sample, sampling rate of EMG electrodes, UP and DOWN spike times after encoding
    Output:
        -plots EMG response and corresponding bipolar spike encoding 
    '''
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,7), sharex=True)
    # fig.tight_layout()
    
    ax1.plot(np.linspace(0, len(data_sample)/sampling_rate, len(data_sample)), data_sample*1000000, color='black') #in microVolts
    ax1.set_title('raw EMG data')
    ax1.set_ylabel('Voltage (\u03BCV)')
    #spike raster plot
    ax2.eventplot([UP_spike_times, DOWN_spike_times], color= 'black', linelengths = 0.5)
    ax2.set_title('Spike encoding raster plot')
    ax2.set_yticks([0,1])
    ax2.set_yticklabels(['DOWN', 'UP'])
    ax2.set_ylabel('Neuron activation')
    # ax2.set_xlabel('Spike time (s)')
    plt.xlabel('Time (s)')
    # fig.savefig('EMG_spike_encoding_example2.svg', format='svg', dpi=1200)
    plt.show()

def plot_encoding_bi_recon(data_sample, sampling_rate, UP_spike_times, DOWN_spike_times, reconstr_data):
    '''
    Input:
        -EMG data sample, sampling rate of EMG electrodes, UP and DOWN spike times after encoding
    Output:
        -plots EMG response and corresponding bipolar spike encoding and reconstructed signal
    '''
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(10,7), sharex=True)
    # fig.tight_layout()
    #original signal
    ax1.plot(np.linspace(0, len(data_sample)/sampling_rate, len(data_sample)), data_sample*1000000, color='black') #in microVolts
    ax1.set_title('raw EMG data')
    ax1.set_ylabel('Voltage (\u03BCV)')
    #spike raster plot
    ax2.eventplot([UP_spike_times, DOWN_spike_times], color= 'black', linelengths = 0.5)
    ax2.set_title('Spike Encoding Raster Plot')
    ax2.set_yticks([0,1])
    ax2.set_yticklabels(['DOWN', 'UP'])
    ax2.set_ylabel('Neuron activation')
    # ax2.set_xlabel('Spike time (s)')
    #reconstructed signal
    ax3.plot(np.linspace(0, len(data_sample)/sampling_rate, len(data_sample)), reconstr_data*1000000, color='black') #in microVolts
    ax3.set_title('Reconstructed Signal')
    ax3.set_ylabel('Voltage (\u03BCV)')
    ax3.set_ylim(ax1.get_ylim())
    plt.xlabel('Time (s)')
    # fig.savefig('EMG_spike_encoding_recon.svg', format='svg', dpi=1200)
    plt.show()
    
    fig2 = plt.figure(figsize=(14,7))
    plt.plot(np.linspace(0, len(data_sample)/sampling_rate, len(data_sample)), data_sample*1000000, color='black', label='original') #in microVolts
    plt.plot(np.linspace(0, len(data_sample)/sampling_rate, len(data_sample)), reconstr_data*1000000, color='red', linestyle='--', label='reconstructed') #in microVolts
    plt.title('Signal Reconstruction')
    plt.ylabel('Voltage (\u03BCV)')
    plt.xlabel('Time (s)')
    plt.legend()
    # fig2.savefig('EMG_origin_recon.svg', format='svg', dpi=1200)
    plt.show()
 

'''
Testing hub
'''
data_sample = emg_pose_1[:500,0]
# UP_spike_times, DOWN_spike_times, UP_spikes, DOWN_spikes, spikes, V_th = temporal_contrast_encod(data_sample, sampling_rate, f=1, refractory=3)
UP_spike_times, DOWN_spike_times, UP_spikes, DOWN_spikes, spikes, V_th = SF_encod(data_sample, sampling_rate, f=1, refractory=0)



# plot_encoding_bi(data_sample, sampling_rate, UP_spike_times, DOWN_spike_times)

reconstr_data = signal_reconstruction(data_sample, spikes, V_th)

plot_encoding_bi_recon(data_sample, sampling_rate, UP_spike_times, DOWN_spike_times, reconstr_data)




