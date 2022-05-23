'''
Hub for second priority code & prototyping.
'''
#%%
import importlib
import SNN_front_end
importlib.reload(SNN_front_end)
from SNN_front_end import *
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
    xf, yf = time_to_freq_domain(np.swapaxes(emg_data, 0, 1), time_pose[index], sampling_rate, classes, no_electrodes, sample=True)
    p.plot_power_spectrum(xf, yf, 'Single motion, Single electrode Unfiltered Power Spectrum', electrode=1)
    #filtered power spectrum
    xf, yf = time_to_freq_domain(np.swapaxes(data_filtered, 0, 1), time_pose[index], sampling_rate, classes, no_electrodes, sample=True)
    p.plot_power_spectrum(xf, yf, 'Single motion, Single electrode Filtered Power Spectrum (BP: {lowcut}-{highcut}Hz)'.format(lowcut=lowcut,highcut=highcut), electrode=1)

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

if __name__=='__main__':
    '''
    Adaptive frequency split
        - perform adaptive cochlear frequency split
        - plot resulting pwoer spectra
    '''
    #filtering using adaptive frequency split
    data_filtered = cochlear_freq_split(emg_data, Adaptive_Freq_Split(emg_data, index, time_pose, sampling_rate, classes, no_electrodes), fs=2000, order=4, no_electrodes=no_electrodes)

    #sample unfiltered power spectrum
    xf, yf = time_to_freq_domain(np.swapaxes(emg_data, 0, 1), time_pose[index], sampling_rate, classes, no_electrodes, sample=True)
    p.plot_power_spectrum(xf, yf, 'Single motion, Single electrode Unfiltered Power Spectrum', electrode=1)
    #filtered power spectrum
    xf, yf = time_to_freq_domain(np.swapaxes(data_filtered, 0, 1), time_pose[index], sampling_rate, classes, data_filtered.shape[0], sample=True)
    for ch in range(data_filtered.shape[0]):
        p.plot_power_spectrum(xf, yf, 'Single Motion Filtered Power Spectrum', electrode=ch, power=int(np.sum(np.abs(yf)[ch])))
# %%
