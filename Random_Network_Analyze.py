#%%
'''
Runs analysis of the Random Network from Random_Network.py
'''
import importlib
import Random_Network
importlib.reload(Random_Network)
from Random_Network import *

#%%
pars, ro_rate, ro_t, hand_kin_data = Net_Simulation(emg_labelled, time_pose, s=3, c=6, rep=0)

# %%
'''Sweep through all samples'''
for c in classes:
    for rep in range(reps):
        Net_Simulation(emg_labelled, time_pose, c=c, rep=rep)

#%%
'''
AVERAGE READOUT ACTIVITY AND HAND KINEMATCS
'''
from scipy.interpolate import interp1d
def standardize_data(x, y, avg, sampling_rate, visual=False):
    '''
    Standardizes data samples with varying lengths to a constant length given by the avearge sample length
    Inputs:
        - x: time axis of data to be standardized
        - y: data to be standardized
        - avg [ms]: average length of samples (i.e., the target length of standardized data)
        - sampling_rate [Hz]: sampling rate of data
        - visual: boolean to determine whether to plot the original and standardized data
    '''
    dt = 1/sampling_rate *1000 #ms
    #the data sample squeezed to fit into the average sample length
    #but the no. of data samples is still the same as in the original (i.e., not standardized)
    x_squeezed = np.linspace(0, avg, len(y))
    #the common time axis of the standardized data 
    x_com = np.linspace(0, avg, int(avg/dt)+1)
    #we interpolate the data to the common time axis such that the squeezed data is resampled with the standard sampling rate
    f = interp1d(x_squeezed, y)
    y_com = f(x_com)
    if visual is True:
        fig = plt.figure(figsize=(10,7))
        plt.title('Standardized Data to Range', fontname="Cambria", fontsize=12)
        plt.plot(x, y, color='blue', label='Original')
        plt.plot(x_squeezed, y, color='red', label='Squeezed')
        plt.plot(x_com, y_com, color='green', label='Standardized')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    return x_com, y_com

if False:#test
    '''Test'''
    #plot y and y_com on same figure
    x = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13])
    y = np.array([0,0.2,0.4,0.6,0.8,0.9,1,0.9,0.85,0.7,0.6,0.5,0.4,0.2])
    x_com, y_com = standardize_data(x, y, avg=8, sampling_rate=1, visual=True)

'''Sweep through all samples of a gesture from one subject'''
def Average_Over_Samples(s, emg_labelled, time_pose, sampling_rate, visual=True):
    '''
    
    '''
    ## Preparing data structures to store standardized data
    #sim net once to get values for defining the following data structures
    pars, ro_rate, ro_t, hand_kin_data = Net_Simulation(emg_labelled, time_pose, s=s, c=6, rep=0)

    avg_time = np.average(time_pose)*1000 #average time of pose in ms

    #sampling rate of activity readout neuron
    activity_sampling_rate = 1/int(ro_t[-1]/ro_rate.shape[0])*1000 #Hz
    #stores the standardized activation readout: shape(sample,standardized activity)
    ro_rate_global = np.zeros((reps,int(avg_time/1000/(1/activity_sampling_rate))+1))

    #hand kinematics: looking only at one sensor (out of the dominant sensors)
    hand_kin_data_one = hand_kin_data[0]
    #stores the standardized hand kinematics: shape(sample,standardized hand kinematics)
    hand_kin_data_global = np.zeros((reps,int(avg_time/(1/sampling_rate*1000))+1))
    #time axis of hand kinematics (and emg data) [ms]
    t_data = np.linspace(0,time_pose[pars['index']],len(hand_kin_data_one))*1000 #ms

    for rep in range(reps): #sweep through samples
        pars, ro_rate, ro_t, hand_kin_data = Net_Simulation(emg_labelled, time_pose, s=s, c=6, rep=rep)

        ro_t_standard, ro_rate_global[rep,:] = standardize_data(ro_t, ro_rate, avg=avg_time, sampling_rate=activity_sampling_rate, visual=False)
        t_data_standard, hand_kin_data_global[rep,:] = standardize_data(t_data, hand_kin_data_one, avg=avg_time, sampling_rate=sampling_rate, visual=False)
    #averaging over all samples
    avg_ro_activity = np.average(ro_rate_global,axis=0)
    avg_kinematics = np.average(hand_kin_data_global,axis=0)

    if visual is True:
        '''overlay AVERAGE activity and AVERAGE norm. kinematics data'''
        fig, ax = plt.subplots(figsize=(10,7))
        ax.plot(ro_t_standard, avg_ro_activity, color='#04c8e0', label='Average Readout Activity')
        ax2=ax.twinx()

        ax2.plot(t_data_standard, ndimage.gaussian_filter1d(avg_kinematics, sigma=75), label='Average Hand Kinematics (smooth)', color='red', alpha=0.8)
        ax2.plot(t_data_standard, avg_kinematics, label='Average Hand Kinematics', color='red', alpha=0.3)

        plt.title('AVERAGE Network Activity and Smoothed-Normed Hand Kinematics '+f'Subject {s}', fontname="Cambria", fontsize=12)
        plt.xlabel('Time [ms]', fontname="Cambria", fontsize=12)
        ax.set_ylabel('Firing Rate [Hz]', fontname="Cambria", fontsize=12)
        ax2.set_ylabel('Fraction of Gesture [dimensionless]', fontname="Cambria", fontsize=12)
        plt.grid(True)
        plt.axis('tight')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        fig.savefig('Figures/Subject_Sweep/'+f'subject_{s}.png')
        plt.show()

# %%
'''
sweep through subjects to see whose activation and kinematics has the best envelope
'''
for s in subjects:
    Average_Over_Samples(s, emg_labelled, time_pose, sampling_rate, visual=True)


# %%
