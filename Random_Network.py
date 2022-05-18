#%%
import SNN_front_end
import importlib
importlib.reload(SNN_front_end)
from SNN_front_end import *
from brian2 import *


# %%
start_scope()
run_length = time_pose[index]*1000 #ms #duration of sim
#no. of input channels/neurons
inp_N = int(max(inp_indeces))+1
'''parameters'''
N = 256 #no. of neurons in network
frac_ex = 0.8 #fraction of excitatory in the population 
#sparsities of the input connections and of the net
inp_sparsity = 0.2
net_sparsity = 0.3
#synapse weights scaling
inp_w_scale = 0.2 #input to ex.
ee_w_scale = 1.5 #ex. to ex.
ei_w_scale = 1 #ex. to in.
ie_w_scale = 10 #in. to ex.
#neuron model equations
tau=10*ms
El = -1 #mV #Leak reversal potential
V_th = 1 #mV #Threshold for firing

ex = int(N*frac_ex) #no. of excitatory neurons
inh = N-ex #no. of inhibiroty neurons


eqs = '''
dv/dt = -(v - El)/tau : 1
''' #leaky integrate and fire without injected current
#(-(v[it] - E_L) + Iinj[it] / g_L)/ tau_m

#input neurons
# P = PoissonGroup(inp_N, np.arange(inp_N)*Hz + 200*Hz)
#defining the input spikes explicitly
P = SpikeGeneratorGroup(inp_N, inp_indeces, inp_spike_times*ms)
pop_P = PopulationRateMonitor(P)

#Excitatory and inhibitory neuron groups
Ex = NeuronGroup(ex, eqs, threshold=f'v>{V_th}', reset='v = 0', method='euler')
In = NeuronGroup(inh, eqs, threshold=f'v>{V_th}', reset='v = 0', method='euler')
#connecting input to excitatory
S = Synapses(P, Ex, 'w : 1', on_pre='v_post += w')
S.connect(p=inp_sparsity)
#setting random weights
S.w = np.random.rand(1,np.sum(S.N_outgoing_pre))*inp_w_scale

#Excitatory to Excitatory connections
S_ee = Synapses(Ex, Ex, 'w : 1', on_pre='v_post += w')
S_ee.connect(condition='i!=j', p=net_sparsity)

#Excitatory to Inhibitory connections
S_ei = Synapses(Ex, In, 'w : 1', on_pre='v_post += w')
S_ei.connect(p=net_sparsity)

#Inhibitory to Excitatory connections
S_ie = Synapses(In, Ex, 'w : 1', on_pre='v_post -= w')
S_ie.connect(p=net_sparsity)

#setting random weights
S_ee.w = np.random.rand(1,np.sum(S_ee.N_outgoing_pre))*ee_w_scale
S_ei.w = np.random.rand(1,np.sum(S_ei.N_outgoing_pre))*ei_w_scale
S_ie.w = np.random.rand(1,np.sum(S_ie.N_outgoing_pre))*ie_w_scale


def visualise_connectivity(S):
    Ns = len(S.source)*V
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


#plot connectivity with scalled points by weight
# scatter(S.i, S.j, S.w*20)
# xlabel('Source neuron ID')
# ylabel('Target neuron ID');
# plt.title('Connectivity scaled by weight')
# plt.show()

#input spiking population
M_in = SpikeMonitor(P)
#excitatory spiking population
M_ex = StateMonitor(Ex, 'v', record=True)
spike_ex = SpikeMonitor(Ex)
pop_ex = PopulationRateMonitor(Ex)

#inhibitory spiking population
# M_in = StateMonitor(In, 'v', record=True)
spike_in = SpikeMonitor(In)
pop_in = PopulationRateMonitor(In)

run(run_length*ms)

#plot Ex. neuronal excitation
fig = plt.figure(figsize=(10,7))
plot(M_ex.t/ms, M_ex.v[0], label='Neuron 0')
plot(M_ex.t/ms, M_ex.v[1], label='Neuron 1')
plot(M_ex.t/ms, M_ex.v[2], label='Neuron 2')
plot(M_ex.t/ms, M_ex.v[3], label='Neuron 3')
plt.title('Neuronal Excitation')
xlabel('Time (ms)')
ylabel('v')
legend();
plt.show()

#raster plot
fig = plt.figure(figsize=(10,7))
plot(M_in.t/ms, M_in.i, '.k')
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

# %%
'''
Plotting binned firing rates of populations
'''
#two by two subplot
plt.subplots(nrows=2, ncols=2, figsize=(10,7))


t = np.linspace(bin, run_length, int(run_length/bin))
fig = plt.figure()figsize=(10,7)
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
