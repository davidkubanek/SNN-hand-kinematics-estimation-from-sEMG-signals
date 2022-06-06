"""
Created on April 03 14:54:55 2022

@author: David

Can load data collected using the custom BioWolf EMG device (our data).

all subjects for all classes.
Can also load data for a single subject in a single class.
Can do data extraction straight on after data load.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import pandas as pd
import os

#Fast Fourier Transform package
from scipy.fft import rfft, rfftfreq, irfft


def time_to_freq_domain(data, time_pose, sampling_rate, classes, no_electrodes, sample=False):
    '''
    Input:
        - data in time-domain in shape (samples, 12)
    Output:
        -  signal coordinates in frequency-domain: xf frequency bins and corresponding yf powers
    '''
    if sample is True: #for single-class data
        # Number of samples in EMG sample
        N = int(sampling_rate * time_pose)
        #matrix of fourier transform for each electrode
        yf = [rfft(data[:,e]) for e in range(no_electrodes)]
        xf = rfftfreq(N, 1 / sampling_rate)
       
        #frequency plot
        # fig = plt.figure(figsize=(10,7))
        # plt.plot(xf, np.abs(yf[0]), color='black') #absolute since yf outputs as complex numbers
        # plt.title('Single motion, Single electrode Power Spectrum')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Power')
        # plt.show()
        
    else: #for multi-class data
        # Number of samples in EMG sample
        N = [int(sampling_rate * time_pose[c]) for c in range(len(classes))]


        #matrix of fourier transform for each electrode and class
        yf = [[rfft(data[c][:,e]) for e in range(no_electrodes)] for c in range(len(classes))]
        xf = [rfftfreq(N[c], 1 / sampling_rate) for c in range(len(classes))]

    
    return xf, yf

# %%     
'''
Large scale data
'''
def load_data(subjects, classes, sampling_rate, no_electrodes=16, extract_features=False):
    '''
    Input:
        - list of subject IDs and classes (hand poses)
    Output
        - emg_labelled: list of lists corresponding to raw EMG for each class repetition (if more than one subject is input, only data of last subject is stored)
                        shape=(no_classes*no_repetitions, samples, no_electrodes)       
        - y holds the classes (hand pose ID) of samples
        - if extract_features=True: extract features from raw EMG and saves features for each electrode over all samples (i.e., subjects and classes) 
          into global_el. if extract_features=False: global_el is empty

    '''
    #alternative paths to file when file is nearby relative to current repository (here we dont have to explicitly define the data_path)
    # script_path = os.path.abspath('EMG_data_SVM_classifier.ipynb') #absolute path to script
    #script_dir = os.path.split(script_path)[0] #absolute path into current directory
    # script_dir = script_dir[:script_dir.rfind('\\')+1] #absolute path into folder one level above the current directory
    # rel_path = 'Data/'+'S'+str(subject)+'_E1_A1.mat' #relative path from folder one level above the current directory to data file
    # abs_file_path = os.path.join(script_dir, rel_path) #absolute path to data file

    #explicit path to data folder
    #  windows
    # data_path = 'C:\\Users\\David\\Projects\\Data'
    #  mac
    data_path = '/Users/david/Documents/Code/Data/EMG_data_BioWolf'

    subjects_labels = [str(s) for s in subjects]
    #target variable stores class ID for each EMG data stream
    y = []
    #initialize empty lists for storing electrode data across samples and subjects
    global_el = [[] for i in range(no_electrodes)]
    for idx, subject in enumerate(subjects):
        #extract data for each subject
        #  windows
        # file_name = '\\S'+str(subject)+'_E1_A1.mat' #relative path from current script directory to data file
        #  mac
        file_name = '/BioWolf_S'+str(subject)+'_E1.mat' #relative path from current script directory to data file
        abs_file_path = data_path + file_name #absolute path to data file
        #load data from MATLAB file
        mat = (spio.loadmat(abs_file_path, squeeze_me=False))
        #extract values from matlab dictionary
        #extracts raw emg data
        emg = mat['Data']
        Test_Name = mat['Test_Name']
        # Subject_Name = mat['Subject_Name']
        # Subject_Age = mat['Subject_Age']
        # Remarks = mat['Remarks']
        sampling_rate = mat['Sample_Rate']
        # Signal_Gain = mat['Signal_Gain']
        Trigger = mat['Trigger']
        timestamp = mat['timestamp']
        # Imu_Data = mat['Imu_Data']

        #converting to shape (reps, samples, 12) compatible with rest of code (mirroring structure of NinaPro VII dataset used in Load_Data)
        emg_labelled = np.array([emg])
        '''
        - add extractions of repetitions
        - what does trigger meean?
        - is timestamp in seconds?
        '''
    

    return emg_labelled

#m = mat['ExGData']
#data = np.array([m['Data']])
# %%     
if __name__ == "__main__":

    no_electrodes = 12 #number of electrodes
    sampling_rate = 2000 #Hz
    classes = [3, 5, 6] #how many movement to classify
    subjects = [1,2]#,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] #subjects to extract
    emg_labelled, y, time_pose, global_el = load_data(subjects, classes, sampling_rate, no_electrodes, extract_features=True)
    data = {'El: '+str(e):global_el[e] for e in range(12)}
    data['Class'] = y
    data_X = pd.DataFrame(data)    
    data_X.head()
    data_X.tail()
    #data_X.info()
    #data_X.shape
    #display(data_X)

#%%
'''
Small scale data
'''   
def sample_data(file_name = 'S11_E1_A1.mat', class_ID=1, extract_features=False, tot_power=False):
    '''
    Input:
        - single subject data
    Output:
        - can be used to process single subject data
        - can be used to plot EMG channels data
        - if tot_power=True: extract a representative long sample spanning multiple hand poses and rest periods 
          from all electrodes and show its power spectrum. This shows what is the general frequency spectrum of our EMG signals
          for use in filters, etc.
        
    '''
    #explicit path to data folder
    data_path = 'C:\\Users\\David\\Projects\\Data'
    file_name = '\\'+file_name #relative path from current script directory to data file
    abs_file_path = data_path + file_name #absolute path to data file
    #load data from MATLAB file
    mat = (spio.loadmat(abs_file_path, squeeze_me=True))
    emg = np.array(mat['emg']) #raw emg data stream containing all poses and rest
    stimulus = np.array(mat['stimulus']) #each time point labelled with corresponding pose
    emg_pose_1 = emg[np.where(stimulus==class_ID)] #emg signal for pose 1
    no_electrodes = np.shape(emg_pose_1)[1] #number of electrodes
    sampling_rate = 2000 #Hz
    time_pose_1 = len(emg_pose_1)/sampling_rate #data collection time interval in seconds

    if extract_features is True:
        #convert to frequency domain
        xf, yf = time_to_freq_domain(emg_pose_1, time_pose_1, sampling_rate, [], no_electrodes, sample=True)
        #Feature Extraction
        MP = [np.sum(np.abs(yf[e])/len(xf)) for e in range(no_electrodes)]

    if tot_power is True: #extract a representative long sample from all electrodes and show its power spectrum
        emg_stream = []
        for e in range(no_electrodes):#concatonate first x samples of each electrode
            emg_stream += emg[:500000,e].tolist()
        time_pose_stream = len(emg_stream)/sampling_rate #data collection time interval in seconds
        #convert to frequency domain
        # Number of samples in EMG sample
        N = int(sampling_rate * time_pose_stream)
        #matrix of fourier transform for each electrode
        yf = rfft(emg_stream)
        xf = rfftfreq(N, 1 / sampling_rate)

        #frequency plot
        fig = plt.figure(figsize=(10,7))
        plt.title('Total Power Spectrum')
        plt.plot(xf, np.abs(yf), color='black') #absolute since yf outputs as complex numbers
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.xticks([0,50,100,150,200,250,300,400,500,600,700,800,900,1000])
        plt.show()
        # fig.savefig('EMG_total_freq_spectrum.svg', format='svg', dpi=1200)

    return emg_pose_1, time_pose_1
#%%
if __name__ == "__main__":
    sample_data(file_name = 'S11_E1_A1.mat', extract_features=False, tot_power=False)

# %%
