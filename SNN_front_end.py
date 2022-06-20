

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
Extracting and transformting data:
    - type='class' or 'subject' depending on whether the data contains samples from:
            - a single subject with many classes OR
            -  a single class with many subjects
'''
type = 'subject'
if type=='class':
    '''
    suitable for analysing all samples (classes and reps) of a single subject
    '''

    no_electrodes = 12
    sampling_rate = 2000 #Hz
    classes = [c+1 for c in range(17)] #which movements to classify based on ID
    subjects = [1] #subject to extract
    reps = 6

    #extract data
    emg_labelled, y, time_pose, _, _, restimulus, _, hand_kin_labelled = load_data(subjects, classes, sampling_rate, no_electrodes)
    print(f'Data loaded: {len(classes)} movements ({reps} reps each) for Subject {subjects[0]}')

elif type=='subject':
    '''
    ssuitable for analysing all samples of single condition from all subjects
    '''

    no_electrodes = 12
    sampling_rate = 2000 #Hz
    classes = [6] #which movement to classify based on ID
    subjects = [s+1 for s in range(4)] #subjects to extract
    reps = 6

    #extract data
    #stores all samples of a class from all subjects
    #shape= (subjects,reps)
    emg_labelled_s = []
    hand_kin_labelled_s = []
    time_pose_s = []
    for s in subjects:
        emg_labelled_one_subject, y, time_pose_one_subject, _, _, restimulus, _, hand_kin_labelled_one_subject = load_data([s], classes, sampling_rate, no_electrodes)
        emg_labelled_s.append(emg_labelled_one_subject)
        hand_kin_labelled_s.append(hand_kin_labelled_one_subject)
        time_pose_s.append(time_pose_one_subject)
    del(emg_labelled_one_subject)
    del(hand_kin_labelled_one_subject)
    del(time_pose_one_subject)
    #reshapes all samples of a class from all subjects
    #shape= (subjects*reps,) to be consistent
    emg_labelled = []
    hand_kin_labelled = []
    time_pose = []
    for s in range(len(subjects)):
        for rep in range(reps):
            emg_labelled.append(emg_labelled_s[s][rep])
            hand_kin_labelled.append(hand_kin_labelled_s[s][rep])
            time_pose.append(time_pose_s[s][rep])
    del(emg_labelled_s)
    del(hand_kin_labelled_s)
    del(time_pose_s)
    
    print(f'Data loaded: {len(emg_labelled)} samples of movement id={classes[0]} from {len(subjects)} subjects')

#functions to extract data from emg_labelled
def Index_To_Tag(index, type=type, classes=classes, subjects=subjects, reps=reps):
    '''
    Converts index of emg_labelled or hand_kin_labelled to a meaningful tag
    - type='class' or 'subject' depending on whether the data contains samples from:
            - a single subject with many classes OR
            -  a single class with many subjects
    '''
    if type=='subject':
        s = str(subjects[int(np.floor(index/reps))])
        rep = str(index%reps)
        tag = 'Subject: ' + s + ', Rep: ' + rep + ' (Class: ' + str(classes[0]) + ')'

    elif type=='class':
        c = str(classes[int(np.floor(index/reps))])
        rep = str(index%reps)
        tag = 'Class: ' + c + ', Rep: ' + rep + ' (Subject: ' + str(subjects[0]) + ')'
    
    return tag
def Tag_To_Index(s=-1, c=0, rep=0, type=type, classes=classes, subjects=subjects, reps=reps):
    '''
    Converts tag to an index of emg_labelled or hand_kin_labelled
    - s: subject id
    - c: class of movement
    - rep: no. of repetition of movement
    - type='class' or 'subject' depending on whether the data contains samples from:
            - a single subject with many classes (type='classes') OR
            -  a single class with many subjects (type='subjects')
    '''
    assert rep in range(reps), 'Repetition not in range use indeces from 0 to ' + str(reps-1)
    
    if type=='subject':
        assert s!=(-1), 'Wrong type: use type=class to get from class tag to index OR change input arguments to match current type: ' + type + '(i.e., specify subject id)'
        assert s in subjects, f'Subject {s} not in subjects:' + str(subjects)
        index = (subjects.index(s)*reps)+rep
    elif type=='class':
        assert c!=0, 'Wrong type: use type=subject to get from subject tag to index OR change input arguments to match current type: ' + type + '(i.e., specify class id)'
        assert c in classes, f'Class {c} not in classes:' + str(classes)
        index = (classes.index(c)*reps)+rep
    return int(index)

def SNN_Full_Input(emg_labelled, hand_kin_labelled, time_pose, c=5, rep=2, subjects=subjects, classes=classes, reps=reps, no_electrodes=no_electrodes, sampling_rate=sampling_rate):
    '''
    Returns the input channels spike trains for a given class and repetition id
    Input:
        - c and rep: class and repetition id of movement for which to get input channels spike trains
    '''
    #single subject, single class repetition: data structure with 12 channels (electrodes)
    #shape (12, samples)
    #convert to microVolts
    index = Tag_To_Index(c=c, rep=rep)
    tag = Index_To_Tag(index)
    emg_data = emg_labelled[index]*1000000
    emg_data = np.swapaxes(emg_data, 0, 1)
    #convert hand kinematics data to the same format
    hand_kin_data = hand_kin_labelled[index]
    hand_kin_data = np.swapaxes(hand_kin_data, 0, 1)
    #get indices of 5 dominant (i.e., most varying) nodes
    hk_std = np.std(hand_kin_data, axis=1)
    dom_nodes = np.where(hk_std>=np.average(hk_std)+np.std(hk_std))[0] #higher than (average + std) variability
    # dom_nodes = (-hk_std).argsort()[:5]
    hand_kin_data = NormalizeData(hand_kin_data)

    '''
    PCA
    '''
    #sample = emg_data[:no_electrodes+1,:200]
    #pc_electrodes = PCA_reduction(sample, no_electrodes, sampling_rate, ex_var=0.9, visual=0)
    #emg_data = emg_data[pc_electrodes,:]

    #most influential electrodas from global offline PCA analysis
    pc_electrodes = np.array([0, 1, 6, 7, 8, 9])
    '''
    Front-end data pre-processing
    '''
    front_end_data = Front_End(emg_data[pc_electrodes], time_pose, no_electrodes=len(pc_electrodes))
    print('No. of front-end channels:', front_end_data.shape[0])

    '''
    LIF Input Layer Spike Encoding
    '''
    #Extracting input spike trains
    channels = front_end_data.shape[0]
    sim_run_time = time_pose[index]*1000 #sim time directly related to length of injected current
    #alternatively, but slower
    #sim_run_time = np.max(time_pose[:channels])*1000 #ms #defined such that simulation time is always longer or equal to the gesture time
    inp_spike_times, inp_indeces = Input_Spikes(front_end_data[:channels], sim_run_time, sampling_rate, R=1, scale=1000000, visual=False, Plots_object=p)

    return inp_spike_times, inp_indeces, index, tag, hand_kin_data, dom_nodes



#%%

def SNN_Full_Input(emg_labelled, hand_kin_labelled, time_pose, s=-1, c=0, rep=2, type=type, subjects=subjects, classes=classes, reps=reps, no_electrodes=no_electrodes, sampling_rate=sampling_rate):
    '''
    Returns the input channels spike trains for a given class and repetition id
    Input:
        - s, c and rep: subject, class and repetition id of movement for which to get input channels spike trains
        - type='class' or 'subject' depending on whether the data contains samples from:
            - a single subject with many classes OR
            - a single class with many subjects
    '''
    #single subject, single class repetition: data structure with 12 channels (electrodes)
    #shape (12, samples)
    #convert to microVolts
    index = Tag_To_Index(s=s, c=c, rep=rep, type=type)
    tag = Index_To_Tag(index, type=type)
    emg_data = emg_labelled[index]*1000000
    emg_data = np.swapaxes(emg_data, 0, 1)
    #convert hand kinematics data to the same format
    hand_kin_data = hand_kin_labelled[index]
    hand_kin_data = np.swapaxes(hand_kin_data, 0, 1)
    #get indices of 5 dominant (i.e., most varying) nodes
    hk_std = np.std(hand_kin_data, axis=1)
    dom_nodes = np.where(hk_std>=np.average(hk_std)+np.std(hk_std))[0] #higher than (average + std) variability
    # dom_nodes = (-hk_std).argsort()[:5]
    hand_kin_data = NormalizeData(hand_kin_data)

    '''
    PCA
    '''
    #sample = emg_data[:no_electrodes+1,:200]
    #pc_electrodes = PCA_reduction(sample, no_electrodes, sampling_rate, ex_var=0.9, visual=0)
    #emg_data = emg_data[pc_electrodes,:]

    #most influential electrodas from global offline PCA analysis
    pc_electrodes = np.array([0, 1, 6, 7, 8, 9])
    '''
    Front-end data pre-processing
    '''
    front_end_data = Front_End(emg_data[pc_electrodes], time_pose, no_electrodes=len(pc_electrodes))
    print('No. of front-end channels:', front_end_data.shape[0])

    '''
    LIF Input Layer Spike Encoding
    '''
    #Extracting input spike trains
    channels = front_end_data.shape[0]
    sim_run_time = time_pose[index]*1000 #sim time directly related to length of injected current
    #alternatively, but slower
    #sim_run_time = np.max(time_pose[:channels])*1000 #ms #defined such that simulation time is always longer or equal to the gesture time
    inp_spike_times, inp_indeces = Input_Spikes(front_end_data[:channels], sim_run_time, sampling_rate, R=1, scale=1000000, visual=False, Plots_object=p)

    return inp_spike_times, inp_indeces, index, tag, hand_kin_data, dom_nodes
# %%
'''
- comment functions
- test random_net with new functions
- test implementation type='subject': remove old SNN function
'''

#%%
SNN_Full_Input(emg_labelled, hand_kin_labelled, time_pose, s=2, c=0, rep=2)