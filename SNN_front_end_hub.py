'''
Hub for second priority code & prototyping.
'''
#%%
import importlib
import SNN_front_end
importlib.reload(SNN_front_end)
from SNN_front_end import *

#%%
if __name__ == '__main__':
    inp_spike_times, inp_indeces, index, tag, hand_kin_data, dom_nodes = SNN_Full_Input(emg_labelled, hand_kin_labelled, time_pose, s=3, c=6, rep=0, type=type, subjects=subjects, classes=classes, reps=reps, no_electrodes=no_electrodes, sampling_rate=sampling_rate)
    #raster plot
    fig = plt.figure(figsize=(10,7))
    plt.plot(inp_spike_times, inp_indeces, '.k')
    plt.title('Input Spikes Raster Plot', fontname="Palatino", fontsize=12)
    plt.xlabel('Time [ms]', fontname="Palatino", fontsize=12)
    plt.ylabel('Neuron index [dimensionless]', fontname="Palatino", fontsize=12)
    plt.yticks([int(tick)*4 for tick in range(int(max(inp_indeces)/4)+1)]);
    plt.xlim([0, 500])
    #fig.savefig('Figures/input_raster.png', format='png', dpi=1200)

# %%
'''
Plot band-pass filter gain
Plot original and filtered data for a single electrode
'''
if __name__ == '__main__':
    # Sample rate and desired cutoff frequencies (in Hz)
    fs = 2000
    lowcut = 50
    highcut = 100
    order=4
    data = emg_data[0]
    #Instantiate the plotter class
    p = Plots()
    p.plot_bandpass(data,  time_pose[0], order, fs, lowcut, highcut)

#%%
    p.plot_EMG(data, time_pose[0], 'Raw EMG Data', ylabel=None)

    #I. First amplifier
    gain_dB = 30
    data = amplifier(data, gain_dB, plot=False, time_pose=time_pose[0], Plots_object=p)
    p.plot_EMG(data, time_pose[0], 'Amplified', ylabel=None)

    #II. Split into frequency bands
    f_bands = [1,50,100,500,999] #frequency intervals
    #data = cochlear_freq_split(data, f_bands, fs=2000, order=4, no_electrodes=no_electrodes)
    
    data = butter_bandpass_filter(data, lowcut, highcut, fs, order=4)
    p.plot_EMG(data, time_pose[0], 'Filtered', ylabel=None)

    #III. Second amplifier
    gain_dB = 30
    data = amplifier(data, gain_dB, plot=False, time_pose=time_pose[0], Plots_object=p)
    p.plot_EMG(data, time_pose[0], 'Amplified', ylabel=None)

    #IV. Half-rectifier
    data = half_rectifier(data, plot=False, time_pose=time_pose[0], Plots_object=p)
    p.plot_EMG(data, time_pose[0], 'Rectified', ylabel=None)

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
'''
Overlay emg and kinematics data
'''
if __name__=='__main__':
    #single subject, single class repetition: data structure with 12 channels (electrodes)
    #shape (12, samples)
    #convert to microVolts
    index = Tag_To_Index(s=13, rep=1)
    tag = Index_To_Tag(index)
    emg_data = emg_labelled[index]*1000000
    emg_data = np.swapaxes(emg_data, 0, 1)
    hand_kin_data = hand_kin_labelled[index]
    hand_kin_data = np.swapaxes(hand_kin_data, 0, 1)

    #get indices of 5 dominant (i.e., most varying) nodes
    hk_std = np.std(hand_kin_data, axis=1)
    dom_nodes = np.where(hk_std>=np.average(hk_std)+np.std(hk_std))[0] #higher than (average + std) variability
    # dom_nodes = (-hk_std).argsort()[:5]

    #p.plot_EMG(emg_data[0,:], time_pose[0], 'emg')

    '''Kinematics data'''
    fig = plt.figure(figsize=(10,7))
    for s in range(no_electrodes):
        plt.plot(np.linspace(0,time_pose[index],len(hand_kin_data[s,:])), hand_kin_data[s,:], label='Node: '+str(s), color='#eb0962')
    plt.title('Hand Kinematics')
    plt.xlabel('Time [seconds]')
    plt.ylabel('Node Angle (degrees)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    '''overlay emg and kinematics data'''
    fig, ax = plt.subplots(figsize=(10,7))
    ax.plot(np.linspace(0,time_pose[index] ,len(emg_data[0,:])), emg_data[0,:], color='#04c8e0', label='EMG')
    ax2=ax.twinx()
    for s in range(no_electrodes): #dom_nodes:
        if s in dom_nodes:
            ax2.plot(np.linspace(0,time_pose[index] ,len(hand_kin_data[s,:])), hand_kin_data[s,:], color='#eb0962', label='DomNode'+str(s))
            continue
        #ax2.plot(np.linspace(0,time_pose[index],len(hand_kin_data[s,:])), hand_kin_data[s,:], label='Node: '+str(s), color='red')

    plt.rc('font',family='Palatino')
    plt.title('EMG and Hand Kinematics',fontname='Palatino', fontsize=14)
    plt.xlabel('Time [seconds]',fontname='Palatino', fontsize=12)
    ax.set_ylabel('Voltage (\u03BCV)',fontname='Palatino', fontsize=12)
    ax2.set_ylabel('Node Angle (degrees)',fontname='Palatino', fontsize=12)
    plt.xlabel('Time [seconds]',fontname='Palatino', fontsize=12)
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.show()

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
    


