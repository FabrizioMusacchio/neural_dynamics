""" 
NEST simulation of biderectional connections according to the Clopath rule

source: https://nest-simulator.readthedocs.io/en/stable/auto_examples/clopath_synapse_small_network.html

Part of the blog post: http://www.fabriziomusacchio.com/blog/2026-04-14-clopath_rule/

modified by: Fabrizio Musacchio
date: Jun 26, 2024
"""
# %% IMPORTS
import os
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import nest
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
# set the simulation resolution and time:
simulation_time = 1.0e4
resolution = 0.1
delay = resolution
nest.ResetKernel()
nest.resolution = resolution

# for reproducibility:
np.random.seed(1)

# poisson_generator parameters:
pg_A = 30.0     # amplitude of Gaussian
pg_sigma = 10.0 # std deviation

# create neurons and devices:
nrn_model = "aeif_psc_delta_clopath"
nrn_params = {
    "V_m": -30.6,           # [mV] membrane potential
    "g_L": 30.0,            # [nS] leak conductance
    "w": 0.0,               # [nS] adaptation conductance
    "tau_u_bar_plus": 7.0,  # [ms] time constant for u_bar_plus (LTP)
    "tau_u_bar_minus": 10.0,# [ms] time constant for u_bar_minus (LTD)
    "tau_w": 144.0,         # [ms] time constant for w
    "a": 4.0,               # [nS] subthreshold adaptation conductance
    "C_m": 281.0,           # [pF] membrane capacitance
    "Delta_T": 2.0,         # [mV] slope factor
    "V_peak": 20.0,         # [mV] spike cut-off
    "t_clamp": 2.0,         # [ms] clamping time
    "A_LTP": 8.0e-6,        # [nS] amplitude of LTP
    "A_LTD": 14.0e-6,       # [nS] amplitude of LTD
    "A_LTD_const": False,   
    "b": 0.0805,            # [nA] spike-triggered adaptation
    "u_ref_squared": 60.0**2# [nA^2] squared threshold for u
    }

pop_exc = nest.Create(nrn_model, 10, nrn_params) # create 10 excitatory neurons
pop_inh = nest.Create(nrn_model, 3, nrn_params)  # create 3 inhibitory neurons

pop_input = nest.Create("parrot_neuron", 500)    # helper neurons (for technical reasons)
pg = nest.Create("poisson_generator", 500)       # poisson generators; i.e., 500 input neurons
wr = nest.Create("weight_recorder")              # create a weight recorder

nest.Connect(pg, pop_input, "one_to_one", 
             {"synapse_model": "static_synapse", 
              "weight": 1.0, 
              "delay": delay})

nest.CopyModel("clopath_synapse", "clopath_input_to_exc", {"Wmax": 3.0})
conn_dict_input_to_exc = {"rule": "all_to_all"}
syn_dict_input_to_exc = {"synapse_model": "clopath_input_to_exc",
                         "weight": nest.random.uniform(0.5, 2.0),
                         "delay": delay}
nest.Connect(pop_input, pop_exc, conn_dict_input_to_exc, syn_dict_input_to_exc)

# create input->inh connections:
conn_dict_input_to_inh = {"rule": "all_to_all"}
syn_dict_input_to_inh = {"synapse_model": "static_synapse", "weight": nest.random.uniform(0.0, 0.5), "delay": delay}
nest.Connect(pop_input, pop_inh, conn_dict_input_to_inh, syn_dict_input_to_inh)

# create exc->exc connections:
nest.CopyModel("clopath_synapse", "clopath_exc_to_exc", {"Wmax": 0.75, "weight_recorder": wr})
syn_dict_exc_to_exc = {"synapse_model": "clopath_exc_to_exc", "weight": 0.25, "delay": delay}
conn_dict_exc_to_exc = {"rule": "all_to_all", "allow_autapses": False}
nest.Connect(pop_exc, pop_exc, conn_dict_exc_to_exc, syn_dict_exc_to_exc)

# create exc->inh connections:
syn_dict_exc_to_inh = {"synapse_model": "static_synapse", "weight": 1.0, "delay": delay}
conn_dict_exc_to_inh = {"rule": "fixed_indegree", "indegree": 8}
nest.Connect(pop_exc, pop_inh, conn_dict_exc_to_inh, syn_dict_exc_to_inh)

# create inh->exc connections:
syn_dict_inh_to_exc = {"synapse_model": "static_synapse", "weight": 1.0, "delay": delay}
conn_dict_inh_to_exc = {"rule": "fixed_outdegree", "outdegree": 6}
nest.Connect(pop_inh, pop_exc, conn_dict_inh_to_exc, syn_dict_inh_to_exc)

# set initial membrane potentials:
pop_exc.V_m = nest.random.normal(-60.0, 25.0)
pop_inh.V_m = nest.random.normal(-60.0, 25.0)

# simulate the network:
sim_interval = 100.0
for i in range(int(simulation_time / sim_interval)):
    # set rates of poisson generators:
    rates = np.empty(500)
    # pg_mu will be randomly chosen out of 25,75,125,...,425,475
    pg_mu = 25 + random.randint(0, 9) * 50
    for j in range(500):
        rates[j] = pg_A * np.exp((-1 * (j - pg_mu) ** 2) / (2 * pg_sigma**2))
        pg[j].rate = rates[j] * 1.75
    nest.Simulate(sim_interval)
    
# sort weights according to sender and reshape:
exc_conns = nest.GetConnections(pop_exc, pop_exc)
exc_conns_senders = np.array(exc_conns.source)
exc_conns_targets = np.array(exc_conns.target)
exc_conns_weights = np.array(exc_conns.weight)
idx_array = np.argsort(exc_conns_senders)
targets = np.reshape(exc_conns_targets[idx_array], (10, 10 - 1))
weights = np.reshape(exc_conns_weights[idx_array], (10, 10 - 1))

# sort according to target:
for i, (trgs, ws) in enumerate(zip(targets, weights)):
    idx_array = np.argsort(trgs)
    weights[i] = ws[idx_array]

weight_matrix = np.zeros((10, 10))
tu9 = np.triu_indices_from(weights)
tl9 = np.tril_indices_from(weights, -1)
tu10 = np.triu_indices_from(weight_matrix, 1)
tl10 = np.tril_indices_from(weight_matrix, -1)
weight_matrix[tu10[0], tu10[1]] = weights[tu9[0], tu9[1]]
weight_matrix[tl10[0], tl10[1]] = weights[tl9[0], tl9[1]]

# difference between initial and final value:
init_w_matrix = np.ones((10, 10)) * 0.25
init_w_matrix -= np.identity(10) * 0.25


# plot synapse weights of the synapses within the excitatory population:
fig, ax = plt.subplots(figsize=(4.85, 4.5))
img = ax.imshow(weight_matrix - init_w_matrix, aspect='auto')

# create an axes on the right side of ax for the colorbar:
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(img, cax=cax)
cbar.ax.tick_params()
cbar.set_ticks(np.arange(-0.002, 0.0115, 0.002))

# adjust the labels and title positions:
ax.set_xlabel("to neuron")
ax.set_ylabel("from neuron")
ax.set_title("Change of synaptic weights\nbefore and after simulation")

# set x and y ticks:
xticklabels = ["1", "3", "5", "7", "9"]
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_xticklabels(xticklabels)
yticklabels = ["1", "3", "5", "7", "9"]
ax.set_yticks([0, 2, 4, 6, 8])
ax.set_yticklabels(yticklabels)

plt.tight_layout()
plt.savefig("figures/clopath_synapse_synaptic_weights.png", dpi=200)
plt.show()
# %% END