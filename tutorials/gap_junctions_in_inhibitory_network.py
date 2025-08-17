""" 
NEST simulation of gap Junctions in an inhibitory network

https://nest-simulator.readthedocs.io/en/stable/auto_examples/gap_junctions_inhibitory_network.html

modified by: Fabrizio Musacchio
date: Jun 26, 2024
"""
# %% IMPORTS
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import nest
import nest.raster_plot
# set the verbosity of the NEST simulator:
nest.set_verbosity("M_WARNING")
# set global properties for all plots:
plt.rcParams.update({'font.size': 12})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False
# create a folder "figures" to save the plots (if it does not exist):
if not os.path.exists('figures'):
    os.makedirs('figures')
# %% MAIN
# for reproducibility:
np.random.seed(1)
nest.ResetKernel()

# set the parameters of the network:
n_neuron        = 500   # number of neurons
gap_per_neuron  = 60    # number of gap junctions per neuron
inh_per_neuron  = 50    # number of inhibitory synapses per neuron
delay           = 1.0   # synaptic delay [ms]
j_exc           = 300.0 # excitatory synaptic weight [pA]
j_inh           = -50.0 # inhibitory synaptic weight [pA]
gap_weight      = 0.00  # gap junction weight [nS]

# set the resolution and the number of threads of the simulation:
simtime         = 501.0 # simulation time [ms]
threads         = 8     # number of threads for the simulation
stepsize        = 0.05  # step size of the simulation [ms] (default: 0.1)
nest.resolution = stepsize
nest.total_num_virtual_procs = threads
nest.print_time = True

# Settings for waveform relaxation. Waveform relaxation is a technique
# to solve the differential equations of the neuron models in parallel.
# If 'use_wfr' is set to False, communication takes place in every step 
# instead of using an iterative solution:
nest.use_wfr = True
nest.wfr_comm_interval = 1.0
nest.wfr_tol = 0.0001
nest.wfr_max_iterations = 15
nest.wfr_interpolation_order = 3

# create and connect nodes:
neurons = nest.Create("hh_psc_alpha_gap", n_neuron)
spikerecorder = nest.Create("spike_recorder")
pg = nest.Create("poisson_generator", params={"rate": 500.0})
"""
poisson_generator: A Poisson generator that generates spikes according to a Poisson process.
rate controls the average rate of the Poisson process in Hz, i.e., the average number of 
input spikes per second.
"""

conn_dict = {"rule": "fixed_indegree", 
             "indegree": inh_per_neuron, 
             "allow_autapses": False, 
             "allow_multapses": True}

syn_dict = {"synapse_model": "static_synapse", 
            "weight": j_inh, 
            "delay": delay}

nest.Connect(neurons, neurons, conn_dict, syn_dict)

nest.Connect(pg, neurons, "all_to_all", 
             syn_spec={"synapse_model": "static_synapse", 
                       "weight": j_exc, 
                       "delay": delay})

nest.Connect(neurons, spikerecorder)

# set the initial membrane potential of the neurons as a random value between -80 and -40 mV:
neurons.V_m = nest.random.uniform(min=-80.0, max=-40.0)

# create gap junctions between the neurons:
n_connection = int(n_neuron * gap_per_neuron / 2)
neuron_list  = neurons.tolist()
connections  = np.random.choice(neuron_list, [n_connection, 2])
for source_node_id, target_node_id in connections:
    nest.Connect(
        nest.NodeCollection([source_node_id]),
        nest.NodeCollection([target_node_id]),
        {"rule": "one_to_one", 
         "make_symmetric": True},
        {"synapse_model": "gap_junction", 
         "weight": gap_weight})

# simulate the network:    
nest.Simulate(simtime)


# extract the number of events and the firing rate:
n_spikes = spikerecorder.n_events
hz_rate = (1000.0 * n_spikes / simtime) / n_neuron

# extract the spike times and neuron IDs from the excitatory spike recorder:
spike_events = nest.GetStatus(spikerecorder, "events")[0]
spike_times = spike_events["times"]
neuron_ids = spike_events["senders"]

# combine the spike times and neuron IDs into a single array and sort by time:
spike_data = np.vstack((spike_times, neuron_ids)).T
spike_data_sorted = spike_data[spike_data[:, 0].argsort()]

# extract sorted spike times and neuron IDs:
sorted_spike_times = spike_data_sorted[:, 0]
sorted_neuron_ids = spike_data_sorted[:, 1]

# spike raster plot and histogram of spiking rate:
fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(5, 1)

# create the first subplot (3/4 of the figure)
ax1 = plt.subplot(gs[0:4, :])
ax1.scatter(sorted_spike_times, sorted_neuron_ids, s=9.0, color='mediumaquamarine', alpha=1.0)
plt.title(f"Network of {n_neuron} inhibitory neurons with gap junctions\ngap_weight: {gap_weight}, average spike rate: {hz_rate:.2f} Hz")
#ax1.set_xlabel("time [ms]")
ax1.set_xticks([])
ax1.set_ylabel("neuron ID")
ax1.set_xlim([0, simtime+5])
ax1.set_ylim([0, n_neuron+1])
ax1.set_yticks(np.arange(0, n_neuron+1, 100))

# create the second subplot (1/4 of the figure)
ax2 = plt.subplot(gs[4, :])
hist_binwidth = 5.0
t_bins = np.arange(np.amin(sorted_spike_times), np.amax(sorted_spike_times), hist_binwidth)
n, bins = np.histogram(sorted_spike_times, bins=t_bins)
heights = 1000 * n / (hist_binwidth * (n_neuron))
ax2.bar(t_bins[:-1], heights, width=hist_binwidth, color='violet')
#ax2.set_title(f"histogram of spiking rate vs. time")
ax2.set_ylabel("firing rate\n[Hz]")
ax2.set_xlabel("time [ms]")
ax2.set_xlim([0, simtime+5])
#ax2.set_ylim([0, 200])

plt.tight_layout()
plt.savefig(f"figures/gap_junctions_in_inhibitory_network_gap_weight{gap_weight}.png", dpi=200)
plt.show()

# calculate the population rate and PRVI (population rate variability index):
# bin spikes:
binwidth = 2.0  # ms; (e.g., 2 ms bins)
t_min, t_max = sorted_spike_times.min(), sorted_spike_times.max()
bins = np.arange(t_min, t_max + binwidth, binwidth)
counts, _ = np.histogram(sorted_spike_times, bins=bins)
rate = (1000.0 / binwidth) * counts / n_neuron  # Hz
# PRVI: coefficient of variation of the population rate:
prvi = np.std(rate) / (np.mean(rate) + 1e-12)
# power spectrum and peak frequency:
freqs = np.fft.rfftfreq(len(rate), d=binwidth/1000.0)  # Hz
power = np.abs(np.fft.rfft(rate))**2
peak_freq = freqs[np.argmax(power[1:]) + 1]  # skip the DC component at index 0
print(f"for a gap junction weight of {gap_weight} nS:")
print(f"PRVI (synchrony index): {prvi:.3f}")
print(f"Peak frequency: {peak_freq:.1f} Hz")
# %% PRVI PLOT
"""
I've collected these PRVI values for different gap junction weights:

for a gap junction weight of 0.0 nS:
PRVI (synchrony index): 1.528
Peak frequency: 51.8 Hz

for a gap junction weight of 0.1 nS:
PRVI (synchrony index): 1.972
Peak frequency: 46.7 Hz

for a gap junction weight of 0.3 nS:
PRVI (synchrony index): 2.832
Peak frequency: 40.8 Hz

for a gap junction weight of 0.5 nS:
PRVI (synchrony index): 3.738
Peak frequency: 26.5 Hz

for a gap junction weight of 0.53 nS:
PRVI (synchrony index): 3.218
Peak frequency: 26.7 Hz

for a gap junction weight of 0.7 nS:
PRVI (synchrony index): 3.570
Peak frequency: 26.4 Hz
"""

gap_weight_array = np.array([0.0, 0.1, 0.3, 0.5, 0.53, 0.7])
prvi_array = np.array([1.528, 1.972, 2.832, 3.738, 3.218, 3.570])
peak_freq_array = np.array([51.8, 46.7, 40.8, 26.5, 26.7, 26.4])

# plot PRVI vs. gap junction weight:
plt.figure(figsize=(11, 4))
plt.plot(gap_weight_array, prvi_array, marker='o', color='mediumaquamarine', label='PRVI')
plt.xlabel("gap junction weight [nS]")
plt.ylabel("PRVI (synchrony index)")
plt.title("PRVI vs. gap junction weight")
plt.xticks(gap_weight_array)
plt.grid()
plt.tight_layout()
plt.savefig("figures/gap_junctions_inhibitory_network_prvi_vs_gap_weight.png", dpi=200)
plt.show()

# plot frequency vs. gap junction weight:
plt.figure(figsize=(11, 4))
plt.plot(gap_weight_array, peak_freq_array, marker='o', color='violet', label='peak frequency')
plt.xlabel("gap junction weight [nS]")
plt.ylabel("peak frequency [Hz]")
plt.ylim([0, 60])
plt.title("Peak frequency vs. gap junction weight")
plt.xticks(gap_weight_array)
plt.grid()
plt.tight_layout()
plt.savefig("figures/gap_junctions_inhibitory_network_peak_freq_vs_gap_weight.png", dpi=200)
plt.show()

# %% END