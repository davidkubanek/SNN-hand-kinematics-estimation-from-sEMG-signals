

#%%
import numpy as np
from scipy.signal import butter, lfilter, resample
import matplotlib.pyplot as plt
from sympy import npartitions, total_degree
import Load_Data
import Support_Functions
import PCA
import importlib
#this method of import ensures that when support scripts are updated, the changes are imported in this script
importlib.reload(Load_Data)
importlib.reload(Support_Functions)
importlib.reload(PCA)
from Load_Data import *
from Support_Functions import *
from PCA import *
import cProfile

#Instantiate the plotter class
p = Plots()

#%%
'''
Extracting and transformting data
'''
no_electrodes = 12
sampling_rate = 2000 #Hz
classes = [3] #which movements to classify based on ID
subjects = [1] #subjects to extract
emg_labelled, y, time_pose, _, _, restimulus, _ = load_data(subjects, classes, sampling_rate, no_electrodes)
#single subject, single class repetition: data structure with 12 channels (electrodes)
#shape (12, samples)
#convert to microVolts
emg_data = emg_labelled[0]*1000000
emg_data = np.swapaxes(emg_data, 0, 1)

'''
PCA
'''
#sample = emg_data[:no_electrodes+1,:200]
#pc_electrodes = PCA_reduction(sample, no_electrodes, sampling_rate, ex_var=0.9, visual=0)
#emg_data = emg_data[pc_electrodes,:]

'''
Front-end
'''
front_end_data = Front_End(emg_data, time_pose, no_electrodes=len(pc_electrodes))
print('No. of front-end channels:', front_end_data.shape[0])

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
