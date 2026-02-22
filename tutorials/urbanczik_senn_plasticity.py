""" 
NEST simulation of weight adaptation according to the Urbanczik-Senn plasticity

source: https://nest-simulator.readthedocs.io/en/stable/auto_examples/urbanczik_synapse_example.html

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

# Set global properties for all plots
plt.rcParams.update({'font.size': 12})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False

# create a folder "figures" to save the plots (if it does not exist):
if not os.path.exists('figures'):
    os.makedirs('figures')
# %% FUNCTIONS

# define a function for the inhibitory input:
def g_inh(amplitude, t_start, t_end):
    """
    returns weights for the spike generator that drives the inhibitory
    somatic conductance.
    """
    return lambda t: np.piecewise(t, [(t >= t_start) & (t < t_end)], [amplitude, 0.0])

# define a function for the excitatory input:
def g_exc(amplitude, freq, offset, t_start, t_end):
    """
    returns weights for the spike generator that drives the excitatory
    somatic conductance.
    """
    return lambda t: np.piecewise(
        t, [(t >= t_start) & (t < t_end)], [lambda t: amplitude * np.sin(freq * t) + offset, 0.0]
    )

# define the matching potential:
def matching_potential(g_E, g_I, nrn_params):
    """
    returns the matching potential as a function of the somatic conductances.
    """
    E_E = nrn_params["soma"]["E_ex"]
    E_I = nrn_params["soma"]["E_in"]
    return (g_E * E_E + g_I * E_I) / (g_E + g_I)

# define the dendritic prediction of the somatic membrane potential:
def V_w_star(V_w, nrn_params):
    """
    returns the dendritic prediction of the somatic membrane potential.
    """
    g_D = nrn_params["g_sp"]
    g_L = nrn_params["soma"]["g_L"]
    E_L = nrn_params["soma"]["E_L"]
    return (g_L * E_L + g_D * V_w) / (g_L + g_D)

# define the rate function phi:
def phi(U, nrn_params):
    """
    rate function of the soma
    """
    phi_max = nrn_params["phi_max"]
    k = nrn_params["rate_slope"]
    beta = nrn_params["beta"]
    theta = nrn_params["theta"]
    return phi_max / (1.0 + k * np.exp(beta * (theta - U)))

# define the derivative of the rate function phi:
def h(U, nrn_params):
    """
    derivative of the rate function phi
    """
    k = nrn_params["rate_slope"]
    beta = nrn_params["beta"]
    theta = nrn_params["theta"]
    return 15.0 * beta / (1.0 + np.exp(-beta * (theta - U)) / k)
# %% MAIN SIMULATION
nest.ResetKernel()

# set simulation parameters:
n_pattern_rep     = 100  # number of repetitions of the spike pattern
pattern_duration  = 200.0
t_start           = 2.0 * pattern_duration
t_end             = n_pattern_rep * pattern_duration + t_start
simulation_time   = t_end + 2.0 * pattern_duration
n_rep_total       = int(np.around(simulation_time / pattern_duration))
resolution        = 0.1
nest.resolution   = resolution

# set neuron parameters:
nrn_model = "pp_cond_exp_mc_urbanczik"
nrn_params = {
    "t_ref": 3.0,       # refractory period
    "g_sp": 600.0,      # soma-to-dendritic coupling conductance
    "soma": {
        "V_m": -70.0,   # initial value of V_m
        "C_m": 300.0,   # capacitance of membrane
        "E_L": -70.0,   # resting potential
        "g_L": 30.0,    # somatic leak conductance
        "E_ex": 0.0,    # resting potential for exc input
        "E_in": -75.0,  # resting potential for inh input
        "tau_syn_ex": 3.0,  # time constant of exc conductance
        "tau_syn_in": 3.0,  # time constant of inh conductance
    },
    "dendritic": {
        "V_m": -70.0,  # initial value of V_m
        "C_m": 300.0,  # capacitance of membrane
        "E_L": -70.0,  # resting potential
        "g_L": 30.0,   # dendritic leak conductance
        "tau_syn_ex": 3.0,  # time constant of exc input current
        "tau_syn_in": 3.0,  # time constant of inh input current
    },
    # set parameters of rate function:
    "phi_max": 0.15,    # max rate
    "rate_slope": 0.5,  # called 'k' in the paper
    "beta": 1.0 / 3.0,
    "theta": -55.0,
}

# set synapse params:
syns = nest.GetDefaults(nrn_model)["receptor_types"]
init_w = 0.3 * nrn_params["dendritic"]["C_m"]
syn_params = {
    "synapse_model": "urbanczik_synapse_wr",
    "receptor_type": syns["dendritic_exc"],
    "tau_Delta": 100.0,  # time constant of low pass filtering of the weight change
    "eta": 0.17,  # learning rate
    "weight": init_w,
    "Wmax": 4.5 * nrn_params["dendritic"]["C_m"],
    "delay": resolution,
}


# in case you want to use the unitless quantities as in Urbanczik and Senn (2014), uncomment the following lines:
""" 
# set neuron params:
nrn_model = 'pp_cond_exp_mc_urbanczik'
nrn_params = {
    't_ref': 3.0,
    'g_sp': 2.0,
    'soma': {
        'V_m': 0.0,
        'C_m': 1.0,
        'E_L': 0.0,
        'g_L': 0.1,
        'E_ex': 14.0 / 3.0,
        'E_in': -1.0 / 3.0,
        'tau_syn_ex': 3.0,
        'tau_syn_in': 3.0,
    },
    'dendritic': {
        'V_m': 0.0,
        'C_m': 1.0,
        'E_L': 0.0,
        'g_L': 0.1,
        'tau_syn_ex': 3.0,
        'tau_syn_in': 3.0,
    },
    # set parameters of rate function
    'phi_max': 0.15,
    'rate_slope': 0.5,
    'beta': 5.0,
    'theta': 1.0,
}

# set synapse params:
syns = nest.GetDefaults(nrn_model)['receptor_types']
init_w = 0.2*nrn_params['dendritic']['g_L']
syn_params = {
    'synapse_model': 'urbanczik_synapse_wr',
    'receptor_type': syns['dendritic_exc'],
    'tau_Delta': 100.0,
    'eta': 0.0003 / (15.0*15.0*nrn_params['dendritic']['C_m']),
    'weight': init_w,
    'Wmax': 3.0*nrn_params['dendritic']['g_L'],
    'delay': resolution,
}
"""


# set somatic input:
ampl_exc = 0.016 * nrn_params["dendritic"]["C_m"] # amplitude of the excitatory input in nA
offset = 0.018 * nrn_params["dendritic"]["C_m"]   # offset of the excitatory input in nA
ampl_inh = 0.06 * nrn_params["dendritic"]["C_m"]  # amplitude of the inhibitory input in nA
freq = 2.0 / pattern_duration                     # frequency of the excitatory input in Hz
soma_exc_inp = g_exc(ampl_exc, 2.0 * np.pi * freq, offset, t_start, t_end) # excitatory input
soma_inh_inp = g_inh(ampl_inh, t_start, t_end)                              # inhibitory input

# set dendritic input: create spike pattern by recording the spikes of a simulation of n_pg
# poisson generators. The recorded spike times are then given to spike generators:
n_pg = 200      # number of poisson generators
p_rate = 10.0  # rate in Hz
pgs = nest.Create("poisson_generator", n=n_pg, params={"rate": p_rate}) # poisson generators
prrt_nrns_pg = nest.Create("parrot_neuron", n_pg)                       # parrot neurons (for technical reasons)
nest.Connect(pgs, prrt_nrns_pg, {"rule": "one_to_one"})
spikerecorder = nest.Create("spike_recorder", n_pg)                # create the spike recorder
nest.Connect(prrt_nrns_pg, spikerecorder, {"rule": "one_to_one"})
nest.Simulate(pattern_duration)
t_srs = [ssr.get("events", "times") for ssr in spikerecorder]

""" 
After simulating the spike pattern, the spike times are stored in the variable t_srs and
we need to reset the simulation kernel to start the actual simulation:
"""
nest.ResetKernel()
nest.resolution = resolution


# define neuron and devices:
nrn = nest.Create(nrn_model, params=nrn_params) # create the Urbanczik neuron

# poisson generators are connected to parrot neurons which are connected to the mc neuron:
prrt_nrns = nest.Create("parrot_neuron", n_pg)

# create excitatory input to the soma:
spike_times_soma_inp = np.arange(resolution, simulation_time, resolution)
sg_soma_exc = nest.Create("spike_generator", 
                          params={"spike_times": spike_times_soma_inp, 
                                  "spike_weights": soma_exc_inp(spike_times_soma_inp)})
# create inhibitory input to the soma:
sg_soma_inh = nest.Create("spike_generator", 
                          params={"spike_times": spike_times_soma_inp, 
                                  "spike_weights": soma_inh_inp(spike_times_soma_inp)})

# create excitatory input to the dendrite:
sg_prox = nest.Create("spike_generator", n=n_pg)

# create a multimeter for recording all parameters of the Urbanczik neuron:
rqs = nest.GetDefaults(nrn_model)["recordables"]
multimeter = nest.Create("multimeter", params={"record_from": rqs, "interval": 0.1})

# create a weight_recorder for recoding the synaptic weights of the Urbanczik synapses:
wr = nest.Create("weight_recorder")

# create another spike recorder for recording the spiking of the soma:
spikerecorder_soma = nest.Create("spike_recorder")

# connect all nodes:
nest.Connect(sg_prox, prrt_nrns, {"rule": "one_to_one"})
nest.CopyModel("urbanczik_synapse", "urbanczik_synapse_wr", {"weight_recorder": wr[0]})
nest.Connect(prrt_nrns, nrn, syn_spec=syn_params)
nest.Connect(multimeter, nrn, syn_spec={"delay": 0.1})
nest.Connect(sg_soma_exc, nrn, 
             syn_spec={"receptor_type": syns["soma_exc"], 
                       "weight": 10.0 * resolution, 
                       "delay": resolution})
nest.Connect(sg_soma_inh, nrn, 
             syn_spec={"receptor_type": syns["soma_inh"], 
                       "weight": 10.0 * resolution, 
                       "delay": resolution})
nest.Connect(nrn, spikerecorder_soma)

# start the simulation, which is divided into intervals of the pattern duration:
for i in np.arange(n_rep_total):
    # Set the spike times of the pattern for each spike generator
    for sg, t_sp in zip(sg_prox, t_srs):
        nest.SetStatus(sg, {"spike_times": np.array(t_sp) + i * pattern_duration})
    nest.Simulate(pattern_duration)


# read out devices for plotting:
# multimeter:
mm_events = multimeter.events
t = mm_events["times"]
V_s = mm_events["V_m.s"]
V_d = mm_events["V_m.p"]
V_d_star = V_w_star(V_d, nrn_params)
g_in = mm_events["g_in.s"]
g_ex = mm_events["g_ex.s"]
I_ex = mm_events["I_ex.p"]
I_in = mm_events["I_in.p"]
U_M = matching_potential(g_ex, g_in, nrn_params)

# weight recorder:
wr_events = wr.events
senders = wr_events["senders"]
targets = wr_events["targets"]
weights = wr_events["weights"]
times = wr_events["times"]

# spike recorder:
spike_times_soma = spikerecorder_soma.get("events", "times")

# %% PLOTS
# plot the results:
lw = 1.0
fig1, (axA, axB, axC, axD, axE) = plt.subplots(5, 1, sharex=True, figsize=(8, 12))

# plot membrane potentials and matching potential:
axA.plot(t, V_s, lw=lw, label=r"$V_s$ (soma)", color="darkblue")
axA.plot(t, V_d, lw=lw, label=r"$V_d$ (dendritic)", color="deepskyblue")
axA.plot(t, V_d_star, lw=lw, label=r"$V_d^\ast$ (predic. dendritic)", color="b", ls="--")
axA.plot(t, U_M, lw=lw, label=r"$U_M$ (matching, soma)", color="r", ls="-", alpha=0.5)
axA.set_ylabel("membrane pot [mV]", )
axA.legend(loc="upper right")

# plot somatic conductances:
axB.plot(t, g_in, lw=lw, label=r"$g_I$", color="r")
axB.plot(t, g_ex, lw=lw, label=r"$g_E$", color="magenta")
axB.set_ylabel("somatic\nconductance [nS]")
axB.legend(loc="upper right")

# plot dendritic currents:
axC.plot(t, I_in, lw=lw, label=f"$I_{{in}}$", color="r")
axC.plot(t, I_ex, lw=lw, label=f"$I_{{ex}}$", color="magenta")
axC.set_ylabel("dendritic\ncurrent [nA]")
axC.legend(loc="upper right")

# plot rates:
axD.plot(t, phi(V_s, nrn_params), lw=lw, label=r"$\phi(V_s)$", color="darkblue")
axD.plot(t, phi(V_d, nrn_params), lw=lw, label=r"$\phi(V_d)$", color="deepskyblue")
axD.plot(t, phi(V_d_star, nrn_params), lw=lw, label=r"$\phi(V_d^\ast)$", color="b", ls="--")
axD.plot(t, phi(V_s, nrn_params) - phi(V_d_star, nrn_params), lw=lw, 
         label=r"$\phi(V_s) - \phi(V_d^\ast)$", color="r", ls="-")
axD.plot(spike_times_soma, 0.15 * np.ones(len(spike_times_soma)), ".", 
         color="g", markersize=2, label="spike")
axD.legend(loc="upper right")

axE.plot(t, h(V_d_star, nrn_params), lw=lw, label=r"$h(V_d^\ast)$", color="g", ls="-")
axE.set_ylabel("rate derivative")
axE.legend(loc="upper right")
axE.set_xlim([0, 5000]) # we don't need to plot the whole simulation time

plt.tight_layout()
plt.savefig("figures/urbanczik_senn_plasticity.png", dpi=200)
plt.show()



# plot panels above in separate figures:
lw = 1.0
# plot membrane potentials and matching potential:
fig1, axA = plt.subplots(1, 1, figsize=(8, 3))
axA.plot(t, V_s, lw=lw, label=r"$V_s$ (soma)", color="darkblue")
axA.plot(t, V_d, lw=lw, label=r"$V_d$ (dendritic)", color="deepskyblue")
axA.plot(t, V_d_star, lw=lw, label=r"$V_d^\ast$ (predic. dendritic)", color="b", ls="--")
axA.plot(t, U_M, lw=lw, label=r"$U_M$ (matching, soma)", color="r", ls="-", alpha=0.5)
axA.set_ylabel("membrane pot [mV]")
axA.legend(loc="upper right")
axA.set_xlim([0, 5000]) # we don't need to plot the whole simulation time
plt.tight_layout()
plt.savefig("figures/urbanczik_senn_plasticity_membrane_potentials.png", dpi=200)
plt.show()

# plot somatic conductances:
fig2, axB = plt.subplots(1, 1, figsize=(8, 3))
axB.plot(t, g_in, lw=lw, label=r"$g_I$", color="r")
axB.plot(t, g_ex, lw=lw, label=r"$g_E$", color="magenta")
axB.set_ylabel("somatic\nconductance [nS]")
axB.legend(loc="upper right")
axB.set_xlim([0, 5000]) # we don't need to plot the whole simulation time
plt.tight_layout()
plt.savefig("figures/urbanczik_senn_plasticity_somatic_conductances.png", dpi=200)
plt.show()

# plot dendritic currents:
fig3, axC = plt.subplots(1, 1, figsize=(8, 3))
axC.plot(t, I_in, lw=lw, label=f"$I_{{in}}$", color="r")
axC.plot(t, I_ex, lw=lw, label=f"$I_{{ex}}$", color="magenta")
axC.set_ylabel("dendritic\ncurrent [nA]")
axC.legend(loc="upper right")
axC.set_xlim([0, 5000]) # we don't need to plot the whole simulation time
plt.tight_layout()
plt.savefig("figures/urbanczik_senn_plasticity_dendritic_currents.png", dpi=200)
plt.show()

# plot rates:
fig4, axD = plt.subplots(1, 1, figsize=(8, 3))
axD.plot(t, phi(V_s, nrn_params), lw=lw, label=r"$\phi(V_s)$", color="darkblue")
axD.plot(t, phi(V_d, nrn_params), lw=lw, label=r"$\phi(V_d)$", color="deepskyblue")
axD.plot(t, phi(V_d_star, nrn_params), lw=lw, label=r"$\phi(V_d^\ast)$", color="b", ls="--")
axD.plot(t, phi(V_s, nrn_params) - phi(V_d_star, nrn_params), lw=lw, 
         label=r"$\phi(V_s) - \phi(V_d^\ast)$", color="r", ls="-")
axD.plot(spike_times_soma, 0.15 * np.ones(len(spike_times_soma)), ".", 
         color="g", markersize=2, label="spike")
axD.set_ylabel("rate [Hz]")
axD.legend(loc="upper right")
axD.set_xlim([0, 5000]) # we don't need to plot the whole simulation time
plt.tight_layout()
plt.savefig("figures/urbanczik_senn_plasticity_rates.png", dpi=200)
plt.show()

# plot rate derivative:
fig5, axE = plt.subplots(1, 1, figsize=(8, 3))
axE.plot(t, h(V_d_star, nrn_params), lw=lw, label=r"$h(V_d^\ast)$", color="g", ls="-")
axE.set_ylabel("rate derivative")
axE.legend(loc="upper right")
axE.set_xlim([0, 5000]) # we don't need to plot the whole simulation time
plt.tight_layout()
plt.savefig("figures/urbanczik_senn_plasticity_rate_derivative.png", dpi=200)
plt.show()



# plot synaptic weights:
fig2, axA = plt.subplots(1, 1, figsize=(5, 4))
for i in np.arange(2, 200, 10):
    index = np.intersect1d(np.where(senders == i), np.where(targets == 1))
    if not len(index) == 0:
        axA.plot(times[index], weights[index], label="pg_{}".format(i - 2), lw=2)

axA.set_title("Synaptic weights of Urbanczik synapses")
axA.set_xlabel("time [ms]")
axA.set_ylabel("weight")
axA.legend(fontsize=7, loc="upper right")
axA.set_xlim([0, 10000])
plt.tight_layout()
plt.savefig("figures/urbanczik_senn_plasticity_weight_adaption.png", dpi=200)
plt.show()
# %% END