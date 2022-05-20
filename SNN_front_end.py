

#%%
import numpy as np
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

if __name__ == '__main__':
    #raster plot
    fig = plt.figure(figsize=(10,7))
    plt.plot(inp_spike_times, inp_indeces, '.k')
    plt.title('Input Spikes', fontname="Cambria", fontsize=12)
    plt.xlabel('Time [ms]', fontname="Cambria", fontsize=12)
    plt.ylabel('Neuron index [dimensionless]', fontname="Cambria", fontsize=12)
    plt.yticks([int(tick)*4 for tick in range(int(max(inp_indeces)/4)+1)]);

# %%
'''
Plot band-pass filter gain
Plot original and filtered data for a single electrode
'''
if __name__ == '__main__':
    # Sample rate and desired cutoff frequencies (in Hz)
    fs = 2000
    lowcut = 100
    highcut = 200
    order=4
    data = emg_data[0]
    #Instantiate the plotter class
    p = Plots()
    p.plot_bandpass(data,  time_pose[0], order, fs, lowcut, highcut)
    data_filtered = butter_bandpass_filter(data, lowcut, highcut, fs, order=4)

# %%
'''
Filter sample data and show power spectra of original unfilitered and filtered signals
'''
if __name__ == '__main__':
    #Band-pass filter all 12 channels
    data_filtered = [butter_bandpass_filter(emg_data[e], lowcut, highcut, fs, order=4) for e in range(no_electrodes)]
    print(np.shape(data_filtered))
    
    #original unfiltered power spectrum (note input axis conversion to make compatible with function)
    xf, yf = time_to_freq_domain(np.swapaxes(emg_data, 0, 1), time_pose[0], sampling_rate, classes, no_electrodes, sample=True)
    p.plot_power_spectrum(xf, yf, 'Single motion, Single electrode Unfiltered Power Spectrum')
    #filtered power spectrum
    xf, yf = time_to_freq_domain(np.swapaxes(data_filtered, 0, 1), time_pose[0], sampling_rate, classes, no_electrodes, sample=True)
    p.plot_power_spectrum(xf, yf, 'Single motion, Single electrode Filtered Power Spectrum (BP: {lowcut}-{highcut}Hz)'.format(lowcut=lowcut,highcut=highcut))

    Filter_spectogram(emg_data, np.array(data_filtered), sampling_rate, electrode = 0)

#%%
'''
Profiler to assess function runtimes
- wrap any code block in 'def main()' function to assess it with the block below
'''
if __name__ == '__main__':
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.print_stats()


# %%
if __name__=='__main__':
    p.plot_EMG(emg_data[0,:], time_pose[0], 'emg')

    # p.plot_EMG(stimulus[6000:25000], len(stimulus[6000:25000])/sampling_rate, 'stimulus')
    # p.plot_EMG(restimulus[6000:110000], len(restimulus[6000:110000])/sampling_rate, 'restimulus')
    # p.plot_EMG(repetition[6000:25000], len(repetition[6000:25000])/sampling_rate, 'repetition')


# %%
'''
Plot EMG data of repetitions of the same movement
'''
if __name__=='__main__':
    for rep in range(6):
        p.plot_EMG(emg_labelled[rep][:,1], time_pose[rep], 'EMG for repetition {}'.format(rep+1))
# %%
if __name__=='__main__':
    '''
    Check if two methods for parsing samples from raw EMG output equivalent samples
    '''
    m = 0
    for i in range(len(classes)*reps):
        t = emg_labelled_1[i] == emg_labelled_2[i]
        if len(np.where(t==0)[0])!=0:
            print('Conflicts:')
            print(np.where(t==0)[0])
            m = 1
    if m == 0:
        print('No clashes, methods are identical')

# %%
