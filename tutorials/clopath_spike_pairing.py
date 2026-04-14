""" 
NEST simulation of a spike pairing experiment with the Clopath synapse model

source: https://nest-simulator.readthedocs.io/en/stable/auto_examples/clopath_synapse_spike_pairing.html

Part of the blog post: http://www.fabriziomusacchio.com/blog/2026-04-14-clopath_rule/

modified by: Fabrizio Musacchio
date: Jun 26, 2024
"""
# %% IMPORTS
import os
import matplotlib.pyplot as plt
import numpy as np
import nest
#import nest.raster_plot
# set the verbosity of the NEST simulator:
nest.set_verbosity("M_WARNING")

# Set global properties for all plots
plt.rcParams.update({'font.size': 12})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False

# create a folder "figures" to save the plots (if it does not exist):
if not os.path.exists('figures'):
    os.makedirs('figures')
# %% MAIN
nest.ResetKernel()

# define the simulation resolution:
resolution = 0.1

# define the parameters of the neuron:
clopath_neuron_params = {
    "V_m": -70.6,           # [mV] membrane potential
    "E_L": -70.6,           # [mV] leak reversal potential
    "C_m": 281.0,           # [pF] membrane capacitance
    "theta_minus": -70.6,   # [mV] threshold for u (LTD)
    "theta_plus": -45.3,    # [mV] threshold for u_bar (LTP)
    "A_LTD": 14.0e-5,       # [nS] amplitude of LTD
    "A_LTP": 8.0e-5,        # [nS] amplitude of LTP
    "tau_u_bar_minus": 10.0,# [ms] time constant for u_bar_minus (LTD)
    "tau_u_bar_plus": 7.0,  # [ms] time constant for u_bar_plus (LTP)
    "delay_u_bars": 4.0,    # [ms] delay with which u_bar_[plus/minus] are processed to compute the synaptic weights.
    "a": 4.0,               # [nS] subthreshold adaptation
    "b": 0.0805,            # [pA] spike-triggered adaptation
    "V_reset": -70.6 + 21.0,# [mV] reset potential
    "V_clamp": 33.0,        # [mV] clamping potential
    "t_clamp": 2.0,         # [ms] clamping time
    "t_ref": 0.0,           # [ms] refractory period
}

# define the spike times for the pre- and postsynaptic neurons:
spike_times_pre = [
    # Presynaptic spike before the postsynaptic
    [20.0, 120.0, 220.0, 320.0, 420.0],
    [20.0, 70.0, 120.0, 170.0, 220.0],
    [20.0, 53.3, 86.7, 120.0, 153.3],
    [20.0, 45.0, 70.0, 95.0, 120.0],
    [20.0, 40.0, 60.0, 80.0, 100.0],
    # Presynaptic spike after the postsynaptic
    [120.0, 220.0, 320.0, 420.0, 520.0, 620.0],
    [70.0, 120.0, 170.0, 220.0, 270.0, 320.0],
    [53.3, 86.6, 120.0, 153.3, 186.6, 220.0],
    [45.0, 70.0, 95.0, 120.0, 145.0, 170.0],
    [40.0, 60.0, 80.0, 100.0, 120.0, 140.0]]

spike_times_post = [
    [10.0, 110.0, 210.0, 310.0, 410.0],
    [10.0, 60.0, 110.0, 160.0, 210.0],
    [10.0, 43.3, 76.7, 110.0, 143.3],
    [10.0, 35.0, 60.0, 85.0, 110.0],
    [10.0, 30.0, 50.0, 70.0, 90.0],
    [130.0, 230.0, 330.0, 430.0, 530.0, 630.0],
    [80.0, 130.0, 180.0, 230.0, 280.0, 330.0],
    [63.3, 96.6, 130.0, 163.3, 196.6, 230.0],
    [55.0, 80.0, 105.0, 130.0, 155.0, 180.0],
    [50.0, 70.0, 90.0, 110.0, 130.0, 150.0]]

# set the initial weight of the synapse::
init_w = 0.5
syn_weights = []

# run the simulation for each pair of spike times:
for s_t_pre, s_t_post in zip(spike_times_pre, spike_times_post):
    nest.ResetKernel()
    nest.resolution = resolution

    # create one adaptive exponential integrate-and-fire neuron neuron with Clopath synapse:
    clopath_neuron = nest.Create("aeif_psc_delta_clopath", 1, clopath_neuron_params)

    # We need a parrot neuron for technical reasons since spike generators can only
    # be connected with static connections:
    parrot_neuron = nest.Create("parrot_neuron", 1)

    # create and connect spike generators:
    spike_gen_pre = nest.Create("spike_generator", {"spike_times": s_t_pre})
    nest.Connect(spike_gen_pre, parrot_neuron, syn_spec={"delay": resolution})

    spike_gen_post = nest.Create("spike_generator", {"spike_times": s_t_post})
    nest.Connect(spike_gen_post, clopath_neuron, syn_spec={"delay": resolution, "weight": 80.0})

    # create weight recorder:
    weightrecorder = nest.Create("weight_recorder")

    # create Clopath connection with weight recorder:
    nest.CopyModel("clopath_synapse", "clopath_synapse_rec", {"weight_recorder": weightrecorder})
    syn_dict = {"synapse_model": "clopath_synapse_rec", "weight": init_w, "delay": resolution}
    nest.Connect(parrot_neuron, clopath_neuron, syn_spec=syn_dict)

    # simulation:
    simulation_time = 10.0 + max(s_t_pre[-1], s_t_post[-1])
    nest.Simulate(simulation_time)

    # extract and save synaptic weights:
    weights = weightrecorder.get("events", "weights")
    syn_weights.append(weights[-1])

syn_weights = np.array(syn_weights)
# scaling of the weights so that they are comparable to Clopath et al (2010):
syn_weights = 100.0 * 15.0 * (syn_weights - init_w) / init_w + 100.0

# plot results:
plt.figure(figsize=(4.5, 3.5))
plt.plot([10.0, 20.0, 30.0, 40.0, 50.0], syn_weights[5:], 
         lw=2.5, ls="-", label="pre-post pairing")
plt.plot([10.0, 20.0, 30.0, 40.0, 50.0], syn_weights[:5], 
         lw=2.5, ls="-", label="post-pre pairing")
plt.ylabel("normalized weight change")
plt.xlabel("firing rate of the presynaptic neuron [Hz]")
"""
[10.0, 20.0, 30.0, 40.0, 50.0] are the firing rates of the presynaptic neuron in Hz.
This frequency is crucial because the rate of presynaptic spikes can significantly 
impact the degree of synaptic plasticity. """
plt.legend()
plt.title(f"Synaptic weight using\nthe Clopath rule")
plt.tight_layout()
plt.savefig("figures/clopath_synapse_spike_pairing.png", dpi=200)
plt.show()
# %% END