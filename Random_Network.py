#%%
'''
Random Liquid Network of excitatory and inhibitory LIF neurons. 
We map the input channels from the SNN front-end into a higher dimensional space and monitor the activity of this higher dimensional network.
'''

import importlib
import SNN_front_end
importlib.reload(SNN_front_end)
from SNN_front_end import *
from brian2 import *


# %%
'''
Define Network Architecture
'''
start_scope()
#no. of input channels/neurons
inp_N = int(max(inp_indeces))+1
#if small sample network
# inp_N = 5

'''Parameters'''
N = 256 #no. of neurons in network
frac_ex = 0.8 #fraction of excitatory in the population 
#sparsities of the input connections and of the net
inp_sparsity = 0.2
net_sparsity = 0.3
#neuron model equations
V_th = 1 #mV #Threshold for firing
refractory = 3*ms

ex = int(N*frac_ex) #no. of excitatory neurons
inh = N-ex #no. of inhibiroty neurons

eqs = '''
dv/dt = -(v - El)/tau : 1 (unless refractory)
tau : second
El : radian
''' #leaky integrate and fire without injected current
#(-(v[it] - E_L) + Iinj[it] / g_L)/ tau_m

#input neurons
#defining the input spikes explicitly
P = SpikeGeneratorGroup(inp_N, inp_indeces, inp_spike_times*ms, name='P')
#if small sample network
# P = PoissonGroup(inp_N, np.arange(inp_N)*Hz + 200*Hz)

#Excitatory and inhibitory neuron groups
Ex = NeuronGroup(ex, eqs, threshold=f'v>{V_th}', reset='v = 0', refractory=refractory, method='euler', name='Ex')
In = NeuronGroup(inh, eqs, threshold=f'v>{V_th}', reset='v = 0', refractory=refractory, method='euler', name='In')

def Random_Connect(inp_N, ex, inh, inp_sparsity, net_sparsity, seed=1):
    '''
    Random connections generator
    - creates lists of pre- and post-synaptic connections for each Synapse type
    '''

    inp_pre_idx = []
    inp_post_idx = []
    for i in range(inp_N):
        # pick inp_sparsity*ex random neurons
        inp_pre_idx += [int(i) for idx in range(int(inp_sparsity*ex))]
        np.random.seed(40*i*seed)
        inp_post_idx += np.random.choice(ex, int(inp_sparsity*ex), replace=False).tolist()    

    ee_pre_idx = []
    ee_post_idx = []
    for i in range(ex):
        # pick net_sparsity*ex random neurons
        ee_pre_idx += [int(i) for idx in range(int(np.round(net_sparsity*ex)))]
        e_array = np.delete(np.arange(ex), i)
        np.random.seed(41*i*seed)
        ee_post_idx += np.random.choice(e_array, int(np.round(net_sparsity*ex)), replace=False).tolist()    

    ei_pre_idx = []
    ei_post_idx = []
    for i in range(ex):
        # pick net_sparsity*inh random neurons
        ei_pre_idx += [int(i) for idx in range(int(np.round(net_sparsity*inh)))]
        np.random.seed(42*i*seed)
        ei_post_idx += np.random.choice(inh, int(np.round(net_sparsity*inh)), replace=False).tolist()

    ie_pre_idx = []
    ie_post_idx = []
    for i in range(inh):
        # pick net_sparsity*inh random neurons
        ie_pre_idx += [int(i) for idx in range(int(np.round(net_sparsity*ex)))]
        np.random.seed(43*i*seed)
        ie_post_idx += np.random.choice(ex, int(np.round(net_sparsity*ex)), replace=False).tolist()

    return inp_pre_idx, inp_post_idx, ee_pre_idx, ee_post_idx, ei_pre_idx, ei_post_idx, ie_pre_idx, ie_post_idx
#getting explicit random connection arrays
inp_pre_idx, inp_post_idx, ee_pre_idx, ee_post_idx, ei_pre_idx, ei_post_idx, ie_pre_idx, ie_post_idx = Random_Connect(inp_N, ex, inh, inp_sparsity, net_sparsity, seed=3)
#connecting input to excitatory
S = Synapses(P, Ex, 'w : 1', on_pre='v_post += w', name='Inp_to_Ex')
S.connect(i=inp_pre_idx, j=inp_post_idx)

#Excitatory to Excitatory connections
S_ee = Synapses(Ex, Ex, 'w : 1', on_pre='v_post += w', name='Ex_to_Ex')
S_ee.connect(i=ee_pre_idx, j=ee_post_idx)

#Excitatory to Inhibitory connections
S_ei = Synapses(Ex, In, 'w : 1', on_pre='v_post += w', name='Ex_to_In')
S_ei.connect(i=ei_pre_idx, j=ei_post_idx)

#Inhibitory to Excitatory connections
S_ie = Synapses(In, Ex, 'w : 1', on_pre='v_post -= w', name='In_to_Ex')
S_ie.connect(i=ie_pre_idx, j=ie_post_idx)

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')
    plt.show()
# visualise_connectivity(S)
# visualise_connectivity(S_ee)
# visualise_connectivity(S_ei)
# visualise_connectivity(S_ie)

'''Monitors'''
#input spiking population
spike_P = SpikeMonitor(P, name='spike_P')
pop_P = PopulationRateMonitor(P)

#excitatory spiking population
M_ex = StateMonitor(Ex, 'v', record=True, name='M_ex')
spike_ex = SpikeMonitor(Ex, name='spike_ex')
pop_ex = PopulationRateMonitor(Ex, name='pop_ex')

#inhibitory spiking population
# M_in = StateMonitor(In, 'v', record=True)
spike_in = SpikeMonitor(In, name='spike_in')
pop_in = PopulationRateMonitor(In, name='pop_in')

#store network with connections as defined above
net = Network(P, Ex, In, S, S_ee, S_ei, S_ie)
net.add(M_ex, spike_P, spike_ex, spike_in, pop_P, pop_ex, pop_in)
net.store(name='net', filename='network_1')

print('-----NETWORK ARCHITECTURE DEFINED-----')

#%%
'''
Define Flexible Network Parameters
'''
''''Parameters'''
#synapse weights scaling
inp_w_scale = 2 #input to ex.
ee_w_scale = 2 #ex. to ex.
ei_w_scale = 0.3 #ex. to in.
ie_w_scale = 10 #in. to ex.
#neuron model equations
tau_ex = 10*ms
tau_in = 10*ms
El = -1 #mV #Leak reversal potential

def Run_Net(inp_w_scale=inp_w_scale, ee_w_scale=ee_w_scale, ei_w_scale=ei_w_scale, ie_w_scale=ie_w_scale, tau_ex=tau_ex, tau_in=tau_in, El=El, net = net):

    net.restore(name='net', filename='network_1')
    Ex = net['Ex']
    In = net['In']
    P = net['P']
    S = net['Inp_to_Ex']
    S_ee = net['Ex_to_Ex']
    S_ei = net['Ex_to_In']
    S_ie = net['In_to_Ex']

    print('____________________PARAMETERS____________________')
    print('No. of neurons:', N)
    print('Fraction of excitatory neurons:', frac_ex, f'({ex}:{inh})')
    print('Input sparsity:', inp_sparsity)
    print('Network sparsity:', net_sparsity)
    print('Input (P) to Ex synapse weight scaling:', inp_w_scale)
    print('Ex to Ex synapse weight scaling:', ee_w_scale)
    print('Ex to In synapse weight scaling:', ei_w_scale)
    print('In to Ex synapse weight scaling:', ie_w_scale)
    print('tau Ex [ms]:', tau_ex)
    print('tau In [ms]:', tau_in)
    print('El [mV]:', El)
    print('V_th [mV]:', V_th)
    print('Refractory [ms]:', Ex._refractory, ', saturation:', np.round(1/Ex._refractory))
    print('__________________________________________________')


    #time constants of NeuronGroups
    Ex.tau = tau_ex
    In.tau = tau_in
    #reverse potentials of NeuronGroups
    Ex.El = El
    In.El = El

    #setting random weights
    np.random.seed(0)
    S.w = np.random.rand(1,np.sum(S.N_outgoing_pre))*inp_w_scale
    np.random.seed(1)
    S_ee.w = np.random.rand(1,np.sum(S_ee.N_outgoing_pre))*ee_w_scale
    np.random.seed(2)
    S_ei.w = np.random.rand(1,np.sum(S_ei.N_outgoing_pre))*ei_w_scale
    np.random.seed(3)
    S_ie.w = np.random.rand(1,np.sum(S_ie.N_outgoing_pre))*ie_w_scale


    #plot connectivity with scalled points by weight
    # scatter(S.i, S.j, S.w*20)
    # xlabel('Source neuron ID')
    # ylabel('Target neuron ID');
    # plt.title('Connectivity scaled by weight')
    # plt.show()

    '''Run'''
    run_length = time_pose[index]*1000 #ms #duration of sim
    net.run(run_length*ms)

    '''Plot'''
    if False:
        #plot Ex. neuronal excitation
        fig = plt.figure(figsize=(10,7))
        #generate random integers
        random_int = np.random.randint(0,ex-1,size=4)
        #plot random 4 neurons
        plot(M_ex.t/ms, M_ex.v[random_int[0]], label=f'Neuron {random_int[0]}')
        plot(M_ex.t/ms, M_ex.v[random_int[1]], label=f'Neuron {random_int[1]}')
        plot(M_ex.t/ms, M_ex.v[random_int[2]], label=f'Neuron {random_int[2]}')
        plot(M_ex.t/ms, M_ex.v[random_int[3]], label=f'Neuron {random_int[3]}')
        plt.title('Sample Neuronal Excitation (Ex)')
        xlabel('Time [ms]')
        ylabel('v [mV]')
        legend();
        plt.show()

    if False:
        #raster plot
        fig = plt.figure(figsize=(10,7))
        plot(spike_P.t/ms, spike_P.i, '.k')
        plt.title('Input Population', fontname="Cambria", fontsize=12)
        plt.xlabel('Time [ms]', fontname="Cambria", fontsize=12)
        plt.ylabel('Neuron index [dimensionless]', fontname="Cambria", fontsize=12)
        plt.yticks([int(tick) for tick in range(inp_N)]);
        #plt.yticks([int(tick)*4 for tick in range(int(max(inp_N)/4)+1)]);
        plt.show()

        #raster plot
        fig = plt.figure(figsize=(10,7))
        plot(spike_ex.t/ms, spike_ex.i, '.k')
        plt.title('Excitatory Population', fontname="Cambria", fontsize=12)
        plt.xlabel('Time [ms]', fontname="Cambria", fontsize=12)
        plt.ylabel('Neuron index [dimensionless]', fontname="Cambria", fontsize=12)
        #plt.yticks([int(tick)*4 for tick in range(int(max(inp_indeces)/4)+1)]);
        plt.show()

        #raster plot
        fig = plt.figure(figsize=(10,7))
        plot(spike_in.t/ms, spike_in.i, '.k')
        plt.title('Inhibitory Population', fontname="Cambria", fontsize=12)
        plt.xlabel('Time [ms]', fontname="Cambria", fontsize=12)
        plt.ylabel('Neuron index [dimensionless]', fontname="Cambria", fontsize=12)
        #plt.yticks([int(tick)*4 for tick in range(int(max(inp_indeces)/4)+1)]);
        plt.show()

    '''
    Calc. binned avg. firing rate (sliding window)
    '''
    from scipy import ndimage
    dt = 0.1 #ms #the sampling time of PopulationRateMonitor
    bin = 50 #ms #the desired sliding window size
    idx = int(bin/dt) #length of window in array indeces
    #removing Hz units and cutting padding at end of pose
    pop_ex_r = np.asarray(pop_ex.rate)
    pop_in_r= np.asarray(pop_in.rate)
    pop_P_r = np.asarray(pop_P.rate)
    p_rate = []
    ex_rate = []
    in_rate = []
    f_rate = []
    for b in range(int(run_length/bin)):
        ex_rate += [np.average(pop_ex_r[b*idx:b*idx+idx])]
        in_rate += [np.average(pop_in_r[b*idx:b*idx+idx])]
        p_rate += [np.average(pop_P_r[b*idx:b*idx+idx])]
        f_rate += [np.average(pop_ex_r[b*idx:b*idx+idx])*frac_ex + np.average(pop_in_r[b*idx:b*idx+idx])*(1-frac_ex)]

    #Gaussian smoothing
    p_rate = ndimage.gaussian_filter1d(p_rate, sigma=2)
    ex_rate = ndimage.gaussian_filter1d(ex_rate, sigma=2)
    in_rate = ndimage.gaussian_filter1d(in_rate, sigma=2)
    f_rate = ndimage.gaussian_filter1d(f_rate, sigma=2)
    '''
    Plotting binned firing rates of populations
    - subplots
    '''
    #two by two subplot
    fig, axs = plt.subplots(2, 2, figsize=(10,7))
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    axs[0,0].plot(np.linspace(bin, run_length, int(run_length/bin)), p_rate, color='#04ccc4', label=tag)
    axs[0,0].set_title('Input Population', fontname="Cambria", fontsize=12)
    axs[0,0].set_xlabel('Time [ms]', fontname="Cambria", fontsize=12)
    axs[0,0].set_ylabel('Firing Rate [Hz]', fontname="Cambria", fontsize=12)
    axs[0,1].plot(np.linspace(bin, run_length, int(run_length/bin)), f_rate, color='#04ccc4', label=tag) #04c8e0
    axs[0,1].set_title('Global Population', fontname="Cambria", fontsize=12)
    axs[0,1].set_xlabel('Time [ms]', fontname="Cambria", fontsize=12)
    axs[0,1].set_ylabel('Firing Rate [Hz]', fontname="Cambria", fontsize=12)
    axs[1,0].plot(np.linspace(bin, run_length, int(run_length/bin)), ex_rate, color='#04ccc4', label=tag)
    axs[1,0].set_title('Excitatory Population', fontname="Cambria", fontsize=12)
    axs[1,0].set_xlabel('Time [ms]', fontname="Cambria", fontsize=12)
    axs[1,0].set_ylabel('Firing Rate [Hz]', fontname="Cambria", fontsize=12)
    axs[1,1].plot(np.linspace(bin, run_length, int(run_length/bin)), in_rate, color='#04ccc4', label=tag)
    axs[1,1].set_title('Inhibitory Population', fontname="Cambria", fontsize=12)
    axs[1,1].set_xlabel('Time [ms]', fontname="Cambria", fontsize=12)
    axs[1,1].set_ylabel('Firing Rate [Hz]', fontname="Cambria", fontsize=12)
    plt.legend();
    plt.show()

#%%
'''
Plotting binned firing rates of populations
- Individual plots
'''

t = np.linspace(bin, run_length, int(run_length/bin))
fig = plt.figure(figsize=(10,7))
plt.plot(np.linspace(bin, run_length, int(run_length/bin)), p_rate, color='#52AD89', label=tag)
plt.title('Input Population Firing Rate', fontname="Cambria", fontsize=12)
plt.xlabel('Time [ms]', fontname="Cambria", fontsize=12)
plt.ylabel('Binned Avg. Firing Rate [Hz]', fontname="Cambria", fontsize=12)
plt.legend()

fig = plt.figure(figsize=(10,7))
plt.plot(np.linspace(bin, run_length, int(run_length/bin)), ex_rate, color='#52AD89', label=tag)
plt.title('Ex. Population Firing Rate', fontname="Cambria", fontsize=12)
plt.xlabel('Time [ms]', fontname="Cambria", fontsize=12)
plt.ylabel('Binned Avg. Firing Rate [Hz]', fontname="Cambria", fontsize=12)
plt.legend()

fig = plt.figure(figsize=(10,7))
plt.plot(np.linspace(bin, run_length, int(run_length/bin)), in_rate, color='#52AD89', label=tag)
plt.title('In. Population Firing Rate', fontname="Cambria", fontsize=12)
plt.xlabel('Time [ms]', fontname="Cambria", fontsize=12)
plt.ylabel('Binned Avg. Firing Rate [Hz]', fontname="Cambria", fontsize=12)
plt.legend()

fig = plt.figure(figsize=(10,7))
plt.plot(np.linspace(bin, run_length, int(run_length/bin)), f_rate, color='#52AD89', label=tag)
plt.title('Global Population Firing Rate', fontname="Cambria", fontsize=12)
plt.xlabel('Time [ms]', fontname="Cambria", fontsize=12)
plt.ylabel('Binned Avg. Firing Rate [Hz]', fontname="Cambria", fontsize=12)
plt.legend()


# %%
