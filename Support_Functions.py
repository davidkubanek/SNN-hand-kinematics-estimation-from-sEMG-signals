'''

'''
import numpy as np
from scipy.signal import butter, lfilter, resample
import matplotlib.pyplot as plt
import math

'''
Filtering functions
'''
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    '''
    Input:
        - data, bandpass range, fs sampling frequency, and filter order
    Output:
        - filtered data
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    data_filtered = lfilter(b, a, data)
    return data_filtered

def amplifier(signal, gain_dB, plot=False, time_pose=None, Plots_object=None):
    '''
    Amplifies input signal by input decibel gain
    Input:
        - data: 2D array (single class, single electrode)
        - if plot is True: time_pose should be scalar value for time of corresponding class 
                           plots only one example channel
    '''
    gain_v=10**(gain_dB/20)
    amp_signal = signal*gain_v
    if plot is True:
        Plots_object.plot_EMG(signal[0], time_pose, 'Pre-amp EMG')
        Plots_object.plot_EMG(amp_signal[0], time_pose, 'Amplified EMG (Gain: {}dB)'.format(gain_dB))
    return amp_signal

def half_rectifier(signal, plot=False, time_pose=None, Plots_object=None):
    '''
    Half rectifies input signal (i.e., floors negative values to zero)
    Input:
        - data: 1D array (single class, single electrode)
        - if plot is True: time_pose should be scalar value for time of corresponding class
                           plots only one example channe
    '''
    rect_signal = signal.clip(min=0)
    if plot is True:
        Plots_object.plot_EMG(signal[0], time_pose, 'Pre-rectified signal')
        Plots_object.plot_EMG(rect_signal[0], time_pose, 'Half-rectified signal')
    return rect_signal

def cochlear_freq_split(emg_data, f_bands, fs=2000, order=4, no_electrodes=12):
    '''
    Splits data stream into mutliple freq. channels using band-pass filters.
    Input:
        - emg_data: single subject, single class, shape (12,samples)
        - f_bands: interval boundaries for lowcuts and highcuts of BP-filters
    Output:
        - (12*no_freq_bands, samples) array of data split into freq. channels
    '''
    data_filtered = None
    for i in range(len(f_bands)-1):
        lowcut = f_bands[i]
        highcut = f_bands[i+1]
        if data_filtered is None:
            data_filtered = [butter_bandpass_filter(emg_data[e], lowcut, highcut, fs, order=4) for e in range(no_electrodes)]
            #plot power spectrum for any electrode (default=0)
            # xf, yf = time_to_freq_domain(np.swapaxes(data_filtered, 0, 1), time_pose[0], sampling_rate, classes, no_electrodes, sample=True)   
            # p.plot_power_spectrum(xf, yf, 'Single motion, Single electrode Filtered Power Spectrum (BP: {lowcut}-{highcut}Hz)'.format(lowcut=lowcut,highcut=highcut), electrode=0)
        else:
            data_filtered = np.append(data_filtered, [butter_bandpass_filter(emg_data[e], lowcut, highcut, fs, order=4) for e in range(no_electrodes)], axis=0)
            #plot power spectrum for any electrode (default=0)
            # xf, yf = time_to_freq_domain(np.swapaxes(data_filtered, 0, 1), time_pose[0], sampling_rate, classes, no_electrodes, sample=True)   
            # p.plot_power_spectrum(xf, yf, 'Single motion, Single electrode Filtered Power Spectrum (BP: {lowcut}-{highcut}Hz)'.format(lowcut=lowcut,highcut=highcut), electrode=0)
    return data_filtered



def down_sample(a, R):
    '''
    Input:
        - array, downsampling factor R
    Output:
        - Downsampled array a
    Comments:
        - nanmean method is faster than resample
        - also scipy decimate method exists, but you lose information
    ''' 
    #print(a.shape)
    pad_size = math.ceil(float(a.shape[1])/R)*R - a.shape[1]
    a_padded = np.append(a, np.zeros((a.shape[0],pad_size))*np.NaN, axis=1)
    #print(a_padded.shape)
    down_sampled = np.zeros((a_padded.shape[0],int(a_padded.shape[1]/R)))
    #this approach is faster
    for c in range(a_padded.shape[0]):
        down_sampled[c,:] = np.nanmean(a_padded[c].reshape(-1,R), axis=1, keepdims=True)[:,0]
    #for c in range(a_padded.shape[0]):
    #    down_sampled[c,:] = resample(a_padded[c], int(np.shape(a_padded)[1]/2), t=None, axis=0, window=None, domain='time')
    return down_sampled

def Front_End(data, time_pose, no_electrodes=12):
    '''
    Input
        - data: single subject, single class data structure with 12 channels (electrodes)
    Output
        - pipeline that returns a processed (12*no_of_freq_bands,:) array
        - can be injected as current into SpikeGenerator or into LIF neuronal dynamics to get spike IDs and spike times
    Comments:
        - all functions must work with a 2D arrays


    '''
    #I. First amplifier
    gain_dB = 30
    data = amplifier(data, gain_dB, plot=False, time_pose=time_pose[0], Plots_object=None)

    #II. Split into frequency bands
    f_bands = [1,50,100,500,999] #frequency intervals
    data = cochlear_freq_split(data, f_bands, fs=2000, order=4, no_electrodes=no_electrodes)

    #III. Second amplifier
    gain_dB = 30
    data = amplifier(data, gain_dB, plot=False, time_pose=time_pose[0], Plots_object=None)

    #IV. Half-rectifier
    data = half_rectifier(data, plot=False, time_pose=time_pose[0], Plots_object=None)

    return data

def DS_spectogram(emg_data, electrode = 0):
    '''
    Compares the spectogram frequency components of the original and downsampled data
    '''
    from scipy.signal import spectrogram
    from scipy.fft import fftshift
    data = emg_data[electrode,] #for a single movement repetition (approx.)
    #spectogram of original data
    f, t, Sxx = spectrogram(data, fs=sampling_rate)
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,10), sharex=True)
    ax1.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0), shading='gouraud')
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_title('Original data')
    #spectogram of downsampled data
    R = 2
    down_sampled = down_sample(np.reshape(data, (1,len(data))), 2)
    f, t, Sxx = spectrogram(np.reshape(down_sampled,(np.size(down_sampled),)), fs=sampling_rate/R)
    ax2.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0), shading='gouraud')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_title('Downsampled data')
    plt.xlabel('Time [sec]')
    plt.show()

def Filter_spectogram(original_data, filtered_data, sampling_rate, electrode = 0):
    '''
    Compares the spectogram frequency components of the original and filtered data
    '''
    from scipy.signal import spectrogram
    from scipy.fft import fftshift
    data = original_data[electrode,] #for a single movement repetition (approx.)
    #spectogram of original data
    f, t, Sxx = spectrogram(data, fs=sampling_rate)
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,10), sharex=True)
    ax1.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0), shading='gouraud')
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_title('Original data')
    #spectogram of filtered data
    data = filtered_data[electrode,]
    f, t, Sxx = spectrogram(data, fs=sampling_rate)
    ax2.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0), shading='gouraud')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_title('Filtered data')
    plt.xlabel('Time [sec]')
    plt.show()

def Adaptive_Freq_Split(emg_data, index, time_pose, sampling_rate, classes, no_electrodes, bins=4):
    '''
    Adaptive frequency split based on partition of the power spectrum into bins of equal power
        - computes total power in all electrodes, then splits the power spectrum into bins of equal power cumulative across all electrodes
    Input:
        - emg_data: sample of shape (12,samples)
        - 'index' of sample (movement), use Index_To_Tag(index) to get semantic label
        - time_pose: array of pose times
        - 'sampling_rate' of emg_data
        - classes: list of classes
        - no_electrodes: no. of channels matching shape of emg_data
        - bins: how many frequnecy bands to split the data into
    Output:
        - cut_f: the boundaries of the computed frequency bins
    '''
    #original unfiltered power spectrum (note input axis conversion to make compatible with function)
    xf, yf = time_to_freq_domain(np.swapaxes(emg_data, 0, 1), time_pose[index], sampling_rate, classes, no_electrodes, sample=True)
    #finding the global frequency bins of equal cumulative power
    power_per_bin = np.sum(np.abs(yf))/bins #total power / no. bins
    cut_f = [1] #stores cut-off frequencies
    cut_idx = [] #stores cut-off indices
    b=1
    for i in range(np.shape(np.abs(yf))[1]):
        cum_sum = np.sum(np.abs(yf)[:,:i])
        if cum_sum >= power_per_bin*b:
            cut_f.append(int(xf[i]))
            cut_idx.append(i)
            b += 1
            # print(cum_sum)

    if len(cut_f)<bins+1:
        cut_f.append(int(xf[-1]))

    # print('Freq. bands:\n', cut_f)
    # print('Cut-off indeces:\n', cut_idx)
    return cut_f
    
'''
Plotting functions
'''
class Plots:

    def plot_bandpass(self, data, time_pose, order, fs, lowcut, highcut):
        '''
        Plots filter gain profile and original and filtered signal overlay
        Input:
            - data: 1D array (single class, single electrode)
            - time_pose: scalar value for time of corresponding class 
        '''
        import matplotlib.pyplot as plt
        from scipy.signal import freqz

        # Plot the frequency response
        plt.figure(1)
        plt.clf()
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
                '--', label='sqrt(0.5)') #plot sqrt() threshold
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')

        # Filter a noisy signal.
        plt.figure(2, figsize=(10,7))
        t = np.linspace(0,time_pose,len(data))
        plt.clf()
        plt.plot(t, data, color='black') #in microVolts

        data_filtered = butter_bandpass_filter(data, lowcut, highcut, fs, order=4)
        plt.plot(t, data_filtered, label='Filtered signal')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Voltage (\u03BCV)')
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc='upper left')

        plt.show()

    def plot_EMG(self, data, time_pose, plt_title, ylabel=None):
        '''
        Input:
            - data: 1D array (single class, single electrode)
            - time_pose: scalar value for time of corresponding class 
        '''
        fig = plt.figure(figsize=(10,7))
        plt.plot(np.linspace(0,time_pose,len(data)), data, color='#52AD89')
        plt.title(plt_title)
        plt.xlabel('Time [seconds]')
        plt.ylabel('Voltage (\u03BCV)')
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc='upper left')
        plt.show()

    def plot_power_spectrum(self, xf, yf, plt_title, electrode=0, **kwargs):
        '''
        Displays power spectrum of corresponding electrode
        Input:
            - xf, yf for 12 electrodes, shape (12,samples)
            - can feed in kwargs to customize visualization
        '''
        power = ''
        for arg in kwargs:
            if arg == 'power':
                power = f'\nPower: {kwargs[arg]}'

        fig = plt.figure(figsize=(10,7))
        plt.plot(xf, np.abs(yf[electrode]), color='black', label=f'Channel: {electrode}') #absolute since yf outputs as complex numbers
        plt.title(plt_title + power)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.legend()
        plt.show()


