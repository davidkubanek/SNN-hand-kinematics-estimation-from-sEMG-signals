#%%
'''
Runs analysis of the Random Network from Random_Network.py
'''
import importlib
import Random_Network
importlib.reload(Random_Network)
from Random_Network import *

#%%
pars, ro_rate_200, ro_t_200, gesture_ratio  = Net_Simulation(emg_labelled, time_pose, s=3, c=6, rep=5)


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
        plt.title('Standardized Data to Range', fontname="Palatino", fontsize=14)
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
    x_com, y_com = standardize_data(x, y, avg=8, sampling_rate=1000, visual=True)

'''Sweep through all samples of a gesture from one subject'''
def Average_Over_Samples(s, emg_labelled, time_pose, sampling_rate, reps=reps-1, visual=True):
    '''
    
    '''
    ## Preparing data structures to store standardized data
    #sim net once to get values for defining the following data structures
    pars, ro_rate, ro_t, gesture_ratio  = Net_Simulation(emg_labelled, time_pose, s=s, c=6, rep=0)

    avg_time = np.average(time_pose)*1000 #average time of pose in ms

    #sampling rate of activity readout neuron
    activity_sampling_rate = int(1/ro_t[0]*1000) #Hz
    #stores the standardized activation readout: shape(sample,standardized activity)
    ro_rate_global = np.zeros((reps,int(avg_time/1000/(1/activity_sampling_rate))+1))

    #stores the standardized hand kinematics: shape(sample,standardized hand kinematics)
    gesture_ratio_global = np.zeros((reps,int(avg_time/(1/sampling_rate*1000))+1))
    #time axis of hand kinematics (and emg data) [ms]
    t_data = np.linspace(0,time_pose[pars['index']],len(gesture_ratio))*1000 #ms

    for rep in range(reps): #sweep through samples
        pars, ro_rate, ro_t, hand_kin_data = Net_Simulation(emg_labelled, time_pose, s=s, c=6, rep=rep)

        ro_t_low, ro_rate_global[rep,:] = standardize_data(ro_t, ro_rate, avg=avg_time, sampling_rate=activity_sampling_rate, visual=False)
        t_data_high, gesture_ratio_global[rep,:] = standardize_data(t_data, gesture_ratio, avg=avg_time, sampling_rate=sampling_rate, visual=False)
    #averaging over all samples
    avg_ro_activity_low = np.average(ro_rate_global,axis=0)
    avg_kinematics_high = np.average(gesture_ratio_global,axis=0)

    #standardize hand kinematics further to math sampling rate of activity readout
    t_data_low, avg_kinematics_low = standardize_data(t_data_high, avg_kinematics_high, avg=avg_time, sampling_rate=activity_sampling_rate, visual=False)

    #standardize hand kinematics further to math sampling rate of activity readout
    t_data_high, avg_ro_activity_high = standardize_data(ro_t_low, avg_ro_activity_low, avg=avg_time, sampling_rate=sampling_rate, visual=False)


    if visual is True:
        '''overlay AVERAGE activity and AVERAGE norm. kinematics data'''
        fig, ax = plt.subplots(figsize=(10,7))
        ax.plot(t_data_high, avg_ro_activity_high, color='#04c8e0', label='Average Readout Activity')
        ax2=ax.twinx()

        ax2.plot(t_data_high, ndimage.gaussian_filter1d(avg_kinematics_high, sigma=100), label='Average Hand Kinematics (smooth)', color='#eb0962', alpha=1)
        ax2.plot(t_data_high, avg_kinematics_high, label='Average Hand Kinematics', color='#eb0962', alpha=0.3)

        plt.title('Average Network Activity and Hand Kinematics', fontname="Palatino", fontsize=14)# +f'Subject {s}'
        plt.xlabel('Time [ms]', fontname="Palatino", fontsize=12)
        ax.set_ylabel('Firing Rate [Hz]', fontname="Palatino", fontsize=12)
        ax2.set_ylabel('Gesture Ratio [dimensionless]', fontname="Palatino", fontsize=12)
        plt.grid(True)
        plt.axis('tight')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        fig.savefig('Figures/average_net_5reps_S3_200_100N.png', dpi=800)
        plt.show()

    return avg_ro_activity_high, avg_kinematics_high, t_data_high, avg_ro_activity_low, avg_kinematics_low, t_data_low

# %%
'''
sweep through subjects to see whose activation and kinematics has the best envelope
'''
for s in [3]:#subjects:
    avg_ro_activity_high, avg_kinematics_high, t_data_high, avg_ro_activity_low, avg_kinematics_low, t_data_low = Average_Over_Samples(s, emg_labelled, time_pose, sampling_rate, visual=True)



# %%
'''
smooth out activation curve such that it is an inverse U shape. 
Downsample hand kinematics proportions to match the activation curve.
Rescale kinematics to 0-0.99

hand_kin_data[np.where(hand_kin_data>0.99)] = 0.99

Create a lookup table.

For each incoming rate, assign to closest bin of the lookup table, check index and fetch the correspnding motion ratio value.
output ratio as command to the hand.

Pipeline from raw emg to commands
-make figure: visualize kinematic curves and ASCII commands

classifier that tells us every 200ms in which direction are we going

train on 5, test on remaing. Do for all combinations and give average testing results

install USB drivers
'''

# %%
#the testing repetition is the last one
pars, ro_rate, ro_t, gesture_ratio_ground_truth = Net_Simulation(emg_labelled, time_pose, s=3, c=6, rep=5)

avg_kinematics_high[np.where(avg_kinematics_high>0.99)] = 0.99
max_avg_rate = np.max(avg_ro_activity_high)
max_avg_hand = np.max(avg_kinematics_high)
peak_index = np.where(avg_ro_activity_high==max_avg_rate)[0][0]
ro_close = avg_ro_activity_high[:peak_index+1]
ro_open = avg_ro_activity_high[peak_index:]
hand_close = avg_kinematics_high[:peak_index+1]
hand_open = avg_kinematics_high[peak_index:]
#%%
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



#%%
max_rate = np.max(ro_rate)
test_peak_index = np.where(ro_rate==max_rate)[0][0]
hand_close_test = []
hand_open_test = []
predicted_gesture_ratio = []
for bin in range(len(ro_rate)):
    rate = ro_rate[bin]
    if bin<=test_peak_index:#in the closing phase
        #match ro_rate[bin] firing rate to closest value in avg_ro_activity_high
        idx = find_nearest(ro_close, rate)
        hand_close_test.append(hand_close[idx])
        predicted_gesture_ratio.append(hand_close[idx])
    if bin>test_peak_index:#if in the opening phase
        #match ro_rate[bin] firing rate to closest value in avg_ro_activity_high
        idx = find_nearest(ro_open, rate)
        hand_open_test.append(hand_open[idx])
        predicted_gesture_ratio.append(hand_open[idx])

    
    #print('cmd: ',cmd='@AGPM0'+str(int(avg_kinematics_standard[index]*100))+'45++++++*\r')

#%%
'''overlay test and ground truth gesture ratios'''
fig, ax = plt.subplots(figsize=(10,7))
#ax.plot(np.linspace(bin, run_length, int(run_length/bin)), p_rate, color='#04c8e0', label='Input Population', linestyle='--')
ax.plot(np.linspace(0,time_pose[pars['index']],len(gesture_ratio_ground_truth))*1000, ndimage.gaussian_filter1d(gesture_ratio_test, sigma=75), label='ground truth', color='#eb0962')
ax.plot(ro_t[:test_peak_index+1], ndimage.gaussian_filter1d(hand_close_test, sigma=1), label='closing phase predicted', color='#850537')
ax.plot(ro_t[test_peak_index:], ndimage.gaussian_filter1d(hand_open_test, sigma=1), linestyle='dashed', label='opening phase predicted', color='#850537')
plt.title('Gesture Ratio Test Predictions', fontname="Palatino", fontsize=14)#+'('+tag+')'
plt.xlabel('Time [ms]', fontname="Palatino", fontsize=12)
ax.set_ylabel('Gesture Ratio [dimensionless]', fontname="Palatino", fontsize=12)
plt.grid(True)
plt.axis('tight')
plt.legend()
#fig.savefig('Figures/test_truth_100N.png', format='png', dpi=800)
plt.show()

#%%
f = interp1d(np.linspace(0,time_pose[pars['index']],len(gesture_ratio_ground_truth))*1000, gesture_ratio_ground_truth)
ground_truth = f(ro_t)

from sklearn.metrics import mean_squared_error
mse = sklearn.metrics.mean_squared_error(ground_truth, predicted_gesture_ratio)
rmse = math.sqrt(mse)
print('RMS error:',rmse)

#%%
fig = plt.figure(figsize=(10,7))
plt.plot(ro_t_50, ro_rate_50, color='#04c8e0', linestyle='dashed', label='bin=50', alpha=0.6)
plt.plot(ro_t_100, ro_rate_100, color='#329ba8', linestyle='dashed', label='bin=100', alpha=0.6)
plt.plot(ro_t_150, ro_rate_150, color='#04c8e0', linestyle='solid', label='bin=150',alpha=0.6)
plt.plot(ro_t_200, ro_rate_200, color='#329ba8', linestyle='solid', label='bin=200')
plt.title('Readout Neuron Firing Rate', fontname="Palatino", fontsize=12)
plt.xlabel('Time [ms]', fontname="Palatino", fontsize=12)
plt.ylabel('Firing Rate [Hz]', fontname="Palatino", fontsize=12)
plt.legend()
#fig.savefig('Figures/bin_rate_200.png', dpi=800)
# %%
'''Training error'''
pars, ro_rate, ro_t, gesture_ratio  = Net_Simulation(emg_labelled, time_pose, s=3, c=6, rep=4)


t_data, gesture_ratio = standardize_data(np.linspace(0,time_pose[pars['index']],len(gesture_ratio))*1000, gesture_ratio, avg=avg_time, sampling_rate=sampling_rate, visual=False)
t_data, ro_rate = standardize_data(ro_t, ro_rate, avg=avg_time, sampling_rate=sampling_rate, visual=False)

#f = interp1d(t_data, gesture_ratio)
#training_gesture_ratio = f(ro_t)

from sklearn.metrics import mean_squared_error
mse = sklearn.metrics.mean_squared_error(avg_kinematics_high, gesture_ratio)
rmse = math.sqrt(mse)
print('RMS error kinematics:',rmse)
mse = sklearn.metrics.mean_squared_error(avg_ro_activity_high, ro_rate)
rmse = math.sqrt(mse)
print('RMS error activity:',rmse)
# %%
rms_k = np.average([0.151,0.243,0.138,0.0624,0.0022])
rms_a = np.average([6.116,14.72,14.37,7.911,9.347])