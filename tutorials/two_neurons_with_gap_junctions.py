""" 
NEST simulation of two neurons with gap junctions 

source: https://nest-simulator.readthedocs.io/en/stable/auto_examples/gap_junctions_two_neurons.html

author: Fabrizio Musacchio
date: Jun 26, 2024
"""
# %% IMPORTS
import os
import matplotlib.pyplot as plt
import numpy as np
import nest
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
nest.ResetKernel()

# set the simulation resolution and time:
nest.resolution = 0.05 # ms
T = 350.0 # ms

# create two neurons with Hodgkin-Huxley dynamics and gap junctions:
neuron = nest.Create("hh_psc_alpha_gap", 2)

# set the parameters of the neurons:
neuron.I_e    = 100.0 # constant external input current [pA]
neuron[0].V_m = -10.0 # initial membrane potential of neuron 1 [mV]

# create a voltmeter to record the membrane potential:
voltmeter = nest.Create("voltmeter", params={"interval": 0.1})

# connect the voltmeter to the neurons:
nest.Connect(voltmeter, neuron, "all_to_all")

# connect the neurons with gap junctions:
nest.Connect(
    neuron, neuron, 
    {"rule": "all_to_all", "allow_autapses": False}, 
    {"synapse_model": "gap_junction", "weight": 0.5})

# simulate the network:
nest.Simulate(T)

# extract the data from the voltmeter:
senders = voltmeter.events["senders"]
times = voltmeter.events["times"]
v_m_values = voltmeter.events["V_m"]

# plots:
plt.figure(figsize=(6, 4))
plt.plot(times[np.where(senders == 1)], v_m_values[np.where(senders == 1)], label="neuron 1")
plt.plot(times[np.where(senders == 2)], v_m_values[np.where(senders == 2)], label="neuron 2")
plt.xlabel("time [ms]")
plt.ylabel("membrane potential [mV]")
plt.title("Membrane potential of two HH neurons\nwith gap junctions")
plt.legend()
plt.tight_layout()
plt.savefig("figures/two_neurons_with_gap_junctions.png", dpi=200)
plt.show()
# %% END