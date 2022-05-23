

#%%
import numpy as np
from pyrsistent import b
from scipy.signal import butter, lfilter, resample
import matplotlib.pyplot as plt
import importlib
import Load_Data
#this method of import ensures that when support scripts are updated, the changes are imported in this script
importlib.reload(Load_Data)
from Load_Data import *
import Support_Functions
importlib.reload(Support_Functions)
from Support_Functions import *
#import PCA
#importlib.reload(PCA)
#from PCA import *
import Brian_Input
importlib.reload(Brian_Input)
from Brian_Input import *

import cProfile

#Instantiate the plotter class
p = Plots()


#%%
'''
Extracting and transformting data
'''

no_electrodes = 12
sampling_rate = 2000 #Hz
classes = [5] #which movements to classify based on ID
subjects = [1] #subjects to extract
reps = 6

emg_labelled, y, time_pose, _, _, restimulus, _ = load_data(subjects, classes, sampling_rate, no_electrodes)

#functions to extract data from emg_labelled
def Index_To_Tag(index, classes=classes, reps=reps):
    '''
    Converts index of emg_labelled to a meaningful tag
    '''
    c = str(classes[int(np.floor(index/6))])
    rep = str(index%reps)
    tag = 'Class: ' + c + ', Rep: ' + rep
    return tag
def Tag_To_Index(c=0, rep=0, classes=classes, reps=reps):
    '''
    Converts tag to an index of emg_labelled
    c: class of movement
    rep: no. of repetition of movement
    '''
    assert c in classes, 'Class not in classes'
    assert rep in range(reps), 'Repetition not in range'

    index = (classes.index(c)*reps)+rep
    return int(index)

#single subject, single class repetition: data structure with 12 channels (electrodes)
#shape (12, samples)
#convert to microVolts
index = Tag_To_Index(c=5, rep=2)
tag = Index_To_Tag(index)
emg_data = emg_labelled[index]*1000000
emg_data = np.swapaxes(emg_data, 0, 1)

'''
PCA
'''
#sample = emg_data[:no_electrodes+1,:200]
#pc_electrodes = PCA_reduction(sample, no_electrodes, sampling_rate, ex_var=0.9, visual=0)
#emg_data = emg_data[pc_electrodes,:]

#most influential electrodas from global offline PCA analysis
pc_electrodes = np.array([0, 1, 6, 7, 8, 9])
'''
Front-end
'''
front_end_data = Front_End(emg_data[pc_electrodes], time_pose, no_electrodes=len(pc_electrodes))
print('No. of front-end channels:', front_end_data.shape[0])

#%%
'''
LIF Input Layer Spike Encoding
'''
#Extracting input spike trains
channels = 24
sim_run_time = np.max(time_pose[:channels])*1000 #ms #defined such that simulation time is always longer or equal to the gesture time
inp_spike_times, inp_indeces = Input_Spikes(front_end_data[:channels], sim_run_time, sampling_rate, R=1, scale=1000000, visual=False, Plots_object=p)

