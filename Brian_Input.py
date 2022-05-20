#%%
'''
Represents the last layer of the front-ent.
Encodes the processed time-series signal into a series of spike trains: LIF input neurons.
'''
import importlib
import Support_Functions
importlib.reload(Support_Functions)
from Support_Functions import *
import SNN_front_end
#importlib.reload(SNN_front_end)
#from SNN_front_end import *


# %%
'''
Custom LIF generator
'''
def default_pars(**kwargs):
  '''
  Setting up a dictionary with model parameters
    - Note that, simulation_time and time_step have the unit ms
  '''
  pars = {}

  # typical neuron parameters#
  pars['V_th'] = -55.     # spike threshold [mV]
  pars['V_reset'] = -75.  # reset potential [mV]
  pars['tau_m'] = 10.     # membrane time constant [ms]
  pars['g_L'] = 10.       # leak conductance [nS]
  pars['V_init'] = -75.   # initial potential [mV]
  pars['E_L'] = -75.      # leak reversal potential [mV]
  pars['tref'] = 2.       # refractory time (ms)

  # simulation parameters #
  pars['T'] = 400.  # Total duration of simulation [ms]
  pars['dt'] = .1   # Simulation time step [ms]

  # external parameters if any #
  for k in kwargs:
    pars[k] = kwargs[k]

  pars['range_t'] = np.arange(0, pars['T'], pars['dt'])  # Vector of discretized time points [ms]

  return pars

def run_LIF(pars, Iinj, stop=False):
  """
  Simulate the LIF dynamics with external input current
  Args:
    pars       : parameter dictionary
    Iinj       : input current [micro A]. The injected current here can be a value
                 or an array
    stop       : boolean. If True, use a current pulse
  Returns:
    v      : membrane potential over simulation period
    rec_spike_times     : spike times
    rec_spikes     : time index of spikes
  
  Comments:
    - with zero injected current, no excitation occurs (flat response)
    - injected positive current brings the voltage closer to threshold (the excitation curve gets steeper) and neuron will fire sooner
    - the injected current has to be of a certain order of magnitude to elicit a meaningful response: if current is so high that it brings neuron above V_th every timestep, the v plot will be flat and no rise towards threshold will be seen (only spikes)
  """

  # Set parameters
  V_th, V_reset = pars['V_th'], pars['V_reset']
  tau_m, g_L = pars['tau_m'], pars['g_L']
  V_init, E_L = pars['V_init'], pars['E_L']
  dt, range_t = pars['dt'], pars['range_t']
  Lt = range_t.size
  tref = pars['tref']

  # Initialize voltage
  v = np.zeros(Lt)
  v[0] = V_init

  # Set current time in pA (conversion from microA to pA)
  Iinj = Iinj * 1000000
  #zero pad the current injection
  Iinj = np.append(Iinj, np.zeros(abs(len(range_t)-len(Iinj))))

  # If current pulse, set beginning and end to 0
  if stop:
    Iinj[:int(len(Iinj) / 2) - 1000] = 0
    Iinj[int(len(Iinj) / 2) + 1000:] = 0

  # Loop over time
  rec_spikes = []  # record spike times
  tr = 0.  # the count for refractory duration

  for it in range(Lt - 1):

    if tr > 0:  # check if in refractory period
      v[it] = V_reset  # set voltage to reset
      tr = tr - 1 # reduce running counter of refractory period

    elif v[it] >= V_th:  # if voltage over threshold
      rec_spikes.append(it)  # record spike event
      v[it] = V_reset  # reset voltage
      tr = tref / dt  # set refractory time (i.e., number of iterations we stay in refractory)

    # Calculate the increment of the membrane potential
    dv = (-(v[it] - E_L) + Iinj[it] / g_L) * (dt / tau_m)

    # Update the membrane potential
    v[it + 1] = v[it] + dv

  # Get spike times in ms
  rec_spikes = np.array(rec_spikes)
  rec_spike_times = rec_spikes * dt

  return v, rec_spike_times, rec_spikes

def Input_Spikes(input_current, sim_run_time, sampling_rate, R=1, scale=1000000, visual=False, Plots_object=None):
  '''
  Input:
    - input_current: time_series of the input current to each neuron cahnnel, shape=(no_inp_channels, samples)
    - sim_run_time: how long is neuron receiving input [ms]
    - R: downsample factor
    - sampling_rate of input_current time-series
    - scale: factor by which divide input_current to produce meaningful spiking behaviour
    - if visual is True: we have to put in a plotting object
  Output:
    - inp_spike_times
    - inp_indeces
  '''
  input_current = down_sample(input_current, R)
  inp_indeces = []
  inp_spike_times = []
  pars = default_pars(T=sim_run_time, dt=1/sampling_rate*R*1000)
  #no. of input channels
  no_inp_ch = input_current.shape[0]
  for ch in range(no_inp_ch):
    #Test stimulus
    # stimulus_test = np.ones(input_current[:2].shape)
    # stimulus_test[:, 25:75] = 500
    # stimulus_test = stimulus_test/1000000

    # stimulus = stimulus_test[0]
    stimulus = input_current[ch]/scale #scaled by a factor to produce meaningful spiking behaviour

    # Simulate LIF model
    v, rec_spike_times, rec_spikes = run_LIF(pars, Iinj=stimulus)

    index = np.ones(rec_spike_times.shape)*ch
    inp_indeces += index.tolist()
    #appends recorded spike times to global list of spike times for all neurons (this is the preferred input to brian spike generator)
    inp_spike_times += rec_spike_times.tolist()

    if visual is True:
      print('V_th [mV]:', pars['V_th'], '\nV_reset [mV]:', pars['V_reset'], '\ntau [ms]:',pars['tau_m'], '\nrefractory [ms]:',pars['tref'], '\nTime [ms]:', pars['T'])

      #plot the stimulus injected current
      Plots_object.plot_EMG(stimulus[:int(sim_run_time/dt)], sim_run_time/1000, 'Injected Current', ylabel='Ampere [\u03BCA]')

      #plot the neuron excitation and spikes
      fig = plt.figure(figsize=(10,7))
      plt.plot(pars['range_t'],v, color='#52AD89')
      plt.title('Neuron Excitation', fontname="Cambria", fontsize=12)
      plt.xlabel('Time [ms]', fontname="Cambria", fontsize=12)
      plt.ylabel('Voltage [mV]', fontname="Cambria", fontsize=12)
      for spike in rec_spike_times:
          plt.axvline(spike, ls='--', lw=3, color='#AD5276')
      plt.show()
      
  return np.array(inp_spike_times), np.array(inp_indeces)

# %%
if __name__ == '__main__':
  '''
  Extracts sample spike trains from a given input current and instantiates with this info a brian2 SpikeGenerator
  '''
  #Extracting input spike trains
  sim_run_time = 200 #ms
  p = Plots()
  inp_spike_times, inp_indeces = Input_Spikes(front_end_data, sim_run_time, sampling_rate, R=1, scale=1000000, visual=False, Plots_object=p)


  #raster plot
  fig = plt.figure(figsize=(10,7))
  plt.plot(inp_spike_times, inp_indeces, '.k')
  plt.title('Input Spikes', fontname="Cambria", fontsize=12)
  plt.xlabel('Time [ms]', fontname="Cambria", fontsize=12)
  plt.ylabel('Neuron index [dimensionless]', fontname="Cambria", fontsize=12)
  plt.yticks([int(tick)*4 for tick in range(int(max(inp_indeces)/4)+1)]);



# %%
  '''
  Plotting original EMG signal and the final spike encoding
  '''
  data_sample = emg_data[0]
  fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,7), sharex=True)
  # fig.tight_layout()
  #raw EMG data
  ax1.plot(np.linspace(0, sim_run_time, int(sim_run_time/dt)), data_sample[:int(sim_run_time/dt)], color='black') #in microVolts
  ax1.set_title('Raw EMG Data', fontname="Cambria", fontsize=12)
  ax1.set_ylabel('Voltage [\u03BCV]', fontname="Cambria", fontsize=12)
  #spike raster plot
  ax2.eventplot([rec_spike_times], color= 'black', linelengths = 0.5)
  ax2.set_title('Spike Encoding Raster Plot', fontname="Cambria", fontsize=12)
  ax2.set_yticks([])
  ax2.set_yticklabels([])
  ax2.set_ylabel('Neuron Activation', fontname="Cambria", fontsize=12)
  # ax2.set_xlabel('Spike time (s)')
  plt.xlabel('Time [ms]', fontname="Cambria", fontsize=12)
  # fig.savefig('EMG_spike_encoding_example2.svg', format='svg', dpi=1200)
  plt.show()
# %%
  '''
  Brian2 Input Layer
  '''
  from brian2 import *

  start_scope()
  #no. of input channels
  no_inp_ch = int(max(inp_indeces))+1
  #defining the input spikes explicitly
  G = SpikeGeneratorGroup(no_inp_ch, inp_indeces, inp_spike_times*ms)
  spikemon = SpikeMonitor(G)

  run(sim_run_time*ms)

  #raster plot
  fig = plt.figure(figsize=(10,7))
  plot(spikemon.t/ms, spikemon.i, '.k')
  plt.title('Input Spikes', fontname="Cambria", fontsize=12)
  plt.xlabel('Time [ms]', fontname="Cambria", fontsize=12)
  plt.ylabel('Neuron index [dimensionless]', fontname="Cambria", fontsize=12)
  plt.yticks([int(tick)*4 for tick in range(int(max(inp_indeces)/4)+1)]);
# %%
