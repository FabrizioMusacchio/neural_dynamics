""" 
NEST simulation of structural plasticity (modified)

source: https://nest-simulator.readthedocs.io/en/stable/auto_examples/structural_plasticity.html

modified by: Fabrizio Musacchio
date: Jun 26, 2024
"""
# %% IMPORTS
import os
import matplotlib.pyplot as plt
import numpy as np
import nest
import nest.raster_plot

# Set global properties for all plots
plt.rcParams.update({'font.size': 12})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False

# Create a folder "figures" to save the plots (if it does not exist)
if not os.path.exists('figures'):
    os.makedirs('figures')
# %% MAIN

# set simulation parameters:
t_sim = 200000.0 # simulation time in ms
dt = 0.1 # simulation resolution in ms (is also the resolution 
         # of the update of the synaptic elements/structural plasticity)
number_excitatory_neurons = 800 # number of excitatory neurons
number_inhibitory_neurons = 200 # number of inhibitory neurons
update_interval = 10000.0 # i.e., define how often the connectivity is updated inside the network
                          # synaptic elements and connections change on different time scales
record_interval = 1000.0
bg_rate = 10000.0 # background rate (i.e. rate of Poisson sources)

nest.ResetKernel()
nest.set_verbosity("M_ERROR")
nest.resolution = dt
nest.structural_plasticity_update_interval = update_interval
#nest.local_num_threads = 100 # structural plasticity can not be used with multiple threads

# initialize variables for postsynaptic currents:
psc_e   = 585.0 # excitatory postsynaptic current in pA
psc_i   = -585.0 # inhibitory postsynaptic current in pA
psc_ext = 6.2 # external postsynaptic current in pA

# define neuron model parameters:
neuron_model = "iaf_psc_exp"
model_params = {
    "tau_m": 10.0,      # membrane time constant (ms)
    "tau_syn_ex": 0.5,  # excitatory synaptic time constant (ms)
    "tau_syn_in": 0.5,  # inhibitory synaptic time constant (ms)
    "t_ref": 2.0,       # absolute refractory period (ms)
    "E_L": -65.0,       # resting membrane potential (mV)
    "V_th": -50.0,      # spike threshold (mV)
    "C_m": 250.0,       # membrane capacitance (pF)
    "V_reset": -65.0    # reset potential (mV)
}

# copy synaptic models (will be the base for our structural plasticity synapses):
nest.CopyModel("static_synapse", "synapse_ex")
nest.SetDefaults("synapse_ex", {"weight": psc_e, "delay": 1.0})
nest.CopyModel("static_synapse", "synapse_in")
nest.SetDefaults("synapse_in", {"weight": psc_i, "delay": 1.0})

# define structural plasticity properties:
nest.structural_plasticity_synapses = {
        "synapse_ex": {
        "synapse_model": "synapse_ex",
        "post_synaptic_element": "Den_ex",
        "pre_synaptic_element": "Axon_ex"},
        "synapse_in": {
        "synapse_model": "synapse_in",
        "post_synaptic_element": "Den_in",
        "pre_synaptic_element": "Axon_in"}}

# define growth curves for synaptic elements:
growth_curve_e_e = {"growth_curve": "gaussian", "growth_rate": 0.0001, "continuous": False, "eta": 0.0, "eps": 0.05}
growth_curve_e_i = {"growth_curve": "gaussian", "growth_rate": 0.0001, "continuous": False, "eta": 0.0, "eps": 0.05}
growth_curve_i_e = {"growth_curve": "gaussian", "growth_rate": 0.0004, "continuous": False, "eta": 0.0, "eps": 0.2}
growth_curve_i_i = {"growth_curve": "gaussian", "growth_rate": 0.0001, "continuous": False, "eta": 0.0, "eps": 0.2}

# define synaptic elements:
synaptic_elements   = {"Den_ex": growth_curve_e_e, 
                       "Den_in": growth_curve_e_i, 
                       "Axon_ex": growth_curve_e_e}
synaptic_elements_i = {"Den_ex": growth_curve_i_e, 
                       "Den_in": growth_curve_i_i, 
                       "Axon_in": growth_curve_i_i}

# create_nodes:
nodes_e = nest.Create("iaf_psc_alpha", number_excitatory_neurons, 
                      params={"synaptic_elements": synaptic_elements})
nodes_i = nest.Create("iaf_psc_alpha", number_inhibitory_neurons, 
                      params={"synaptic_elements": synaptic_elements_i})

# show recordables:
#print(nest.GetDefaults(nodes_e.model[0])['recordables'])

# create a Poisson generator for external input and make connections:
noise = nest.Create("poisson_generator", params={"rate": bg_rate})
nest.Connect(noise, nodes_e, syn_spec={"weight": psc_ext, "delay": 1.0})
nest.Connect(noise, nodes_i, syn_spec={"weight": psc_ext, "delay": 1.0})

# create some lists to store results:
mean_ca_e = [] # mean calcium concentration of excitatory neurons
mean_ca_i = [] # mean calcium concentration of inhibitory neurons
total_connections_e = [] # total number of connections of excitatory neurons
total_connections_i = [] # total number of connections of inhibitory neurons

# define a function for recording the calcium concentration of all neurons:
def record_ca():
    global mean_ca_e, mean_ca_i
    ca_e = nest.GetStatus(nodes_e, "Ca")
    ca_i = nest.GetStatus(nodes_i, "Ca")
    mean_ca_e.append(np.mean(ca_e))
    mean_ca_i.append(np.mean(ca_i))

# define a function for recording the number of connections:
def record_connectivity():
    """ 
    We retrieve the number of connected pre-synaptic elements of each neuron. 
    The total amount of excitatory connections is equal to the total amount of 
    connected excitatory pre-synaptic elements. The same applies for inhibitory 
    connections.
    """
    global total_connections_e, total_connections_i
    syn_elems_e = nest.GetStatus(nodes_e, "synaptic_elements")
    syn_elems_i = nest.GetStatus(nodes_i, "synaptic_elements")
    total_connections_e.append(sum(neuron["Axon_ex"]["z_connected"] for neuron in syn_elems_e))
    total_connections_i.append(sum(neuron["Axon_in"]["z_connected"] for neuron in syn_elems_i))

# Structural plasticity can not be used with multiple threads:
if nest.NumProcesses() > 1:
    # set the number of threads to 1:
    nest.set_num_threads(1)
    raise Warning("This example only works for a single process --> Forcing number of threads to 1.")

# simulate:
nest.EnableStructuralPlasticity()
print("Starting simulation...")
sim_steps = np.arange(0, t_sim, record_interval)
for i, step in enumerate(sim_steps):
    nest.Simulate(record_interval)
    record_ca()
    record_connectivity()
    if i % 20 == 0:
        print(f"  progress: {i / 2}%")
print("...simulation finished.")

# plots:
fig, ax1 = plt.subplots(figsize=(6, 4.5))
ax1.plot(mean_ca_e, "b", label="Ca concentration excitatory neurons", linewidth=2.0)
ax1.plot(mean_ca_i, "r", label="Ca concentration inhibitory neurons", linewidth=2.0)
ax1.axhline(growth_curve_i_e["eps"], linewidth=4.0, color="#FF9999") # plot the growth curve for inhibitory neurons
ax1.axhline(growth_curve_e_e["eps"], linewidth=4.0, color="#9999FF") # plot the growth curve for excitatory neurons
ax1.set_ylim([0, 0.28])
ax1.set_xlabel("time in [s]")
ax1.set_ylabel("Ca concentration")
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.plot(total_connections_e, "m", label="excitatory connections", linewidth=2.0, linestyle="--")
ax2.plot(total_connections_i, "k", label="inhibitory connections", linewidth=2.0, linestyle="--")
ax2.set_ylim([0, 2500])
ax2.set_ylabel("Connections")
ax2.legend(loc='lower right')
plt.tight_layout()
plt.savefig("figures/structural_plasticity.png", dpi=200)
plt.show()
# %% END