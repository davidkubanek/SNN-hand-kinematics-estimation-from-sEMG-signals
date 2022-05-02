
#%%
from sklearn.decomposition import PCA

#%%

if __name__ == '__main__':

    from brian2 import *

    input_current = front_end_data[:,20000:21000]/1000 # Volts, smaller sample of the full data
    R = 2
    input_current_ds = down_sample(input_current, R)
    p = Plots()
    p.plot_EMG(input_current[0], np.shape(input_current)[1]*1/2000, 'Input current')
    p.plot_EMG(input_current_ds[0], np.shape(input_current)[1]*1/2000, 'Input current Downsampled')

    start_scope()
    tau = 10*ms; El = -70
    eqs = '''dv/dt = ((El - v) + stimulus(t, i))/tau : 1'''

    stimulus = TimedArray(input_current[:2], dt=1/sampling_rate*1000*ms)
    #stimulus = TimedArray(np.hstack([[c, c, c, 0, 0]
                                #  for c in np.random.rand(1000)]),
                                    #dt=10*ms)
    G = NeuronGroup(2, eqs,
                    threshold='v>1', reset='v=0', method='euler')
    G.v = 0  # different initial values for the neurons

    '''
    Why is the stimulus not exerting an influence?
    - try sub-sample: PSD before and after (NOT decimate, resample scipy): we want it to look similar (not lose a blob)
    '''


    #%%
    statemon = StateMonitor(G, 'v', record=True) #only record Neuron 0 (RAM savings)
    spikemon = SpikeMonitor(G)

    run(100*ms)

    plot(statemon.t/ms, statemon.v[0])
    plt.show()
    plot(statemon.t/ms, statemon.v[1])
    for t in spikemon.t:
        axvline(t/ms, ls='--', c='C1', lw=3)
    xlabel('Time (ms)')
    ylabel('v');


    # %%
    stimulus = TimedArray(np.swapaxes(input_current[:2], 0, 1),
                                    dt=1/sampling_rate*1000*ms)

    eqs = '''dv/dt = (v + stimulus(t,i))/tau : volt
            stimulus : volt''' 
    tau = 10*ms; El = -70*mV                               
    G = NeuronGroup(100, eqs,
                    threshold='v>1', reset='v=0')
    G.v = 0 # different initial values for the neurons


    # %%
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,7), sharex=True)
    # fig.tight_layout()

    ax1.plot(np.linspace(0, len(data_sample)/sampling_rate, len(data_sample)), data_sample*1000000, color='black') #in microVolts
    ax1.set_title('raw EMG data')
    ax1.set_ylabel('Voltage (\u03BCV)')
    #spike raster plot
    ax2.eventplot([UP_spike_times, DOWN_spike_times], color= 'black', linelengths = 0.5)
    ax2.set_title('Spike encoding raster plot')
    ax2.set_yticks([0,1])
    ax2.set_yticklabels(['DOWN', 'UP'])
    ax2.set_ylabel('Neuron activation')
    # ax2.set_xlabel('Spike time (s)')
    plt.xlabel('Time (s)')
    # fig.savefig('EMG_spike_encoding_example2.svg', format='svg', dpi=1200)
    plt.show()