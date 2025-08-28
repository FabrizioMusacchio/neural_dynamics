""" 
NEST simulation of a population rate model of GIF neurons.

source: https://nest-simulator.readthedocs.io/en/stable/auto_examples/gif_pop_psc_exp.html
        https://nest-simulator.readthedocs.io/en/stable/models/gif_pop_psc_exp.html

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
# %% MESOSCOPIC SIMULATION
nest.ResetKernel()
nest.local_num_threads = 100
nest.rng_seed = 1

# define simulation resolutions:
dt = 0.5       # simulation resolution [ms]
dt_rec = 1.0   # resolution of the recordings [ms]
t_end = 2000.0 # simulation time [ms]

nest.resolution = dt
nest.print_time = False # set to False if the code is not executed in a Jupyter notebook or VS Code's interactive window
t0 = nest.biological_time # biological time refers to the time of the NEST kernel

# define the size of the population rate model:
size = 200
N = np.array([4, 1]) * size # number of neurons in each population; here: 800 excitatory and 200 inhibitory neurons
M = len(N)  # number of populations

# neuronal parameters:
t_ref       = 4.0 * np.ones(M)  # absolute refractory period
tau_m       = 20 * np.ones(M)   # membrane time constant
mu          = 24.0 * np.ones(M) # constant base current mu=R*(I0+Vrest)
c           = 10.0 * np.ones(M) # base rate of exponential link function
Delta_u     = 2.5 * np.ones(M)  # softness of exponential link function
V_reset     = 0.0 * np.ones(M)  # Reset potential
V_th        = 15.0 * np.ones(M) # baseline threshold (non-accumulating part)
tau_sfa_exc = [100.0, 1000.0]   # adaptation time constants of excitatory neurons
tau_sfa_inh = [100.0, 1000.0]   # adaptation time constants of inhibitory neurons
J_sfa_exc   = [1000.0, 1000.0]  # size of feedback kernel theta (= area under exponential) in mV*ms
J_sfa_inh   = [1000.0, 1000.0]  # in mV*ms
tau_theta   = np.array([tau_sfa_exc, tau_sfa_inh])
J_theta     = np.array([J_sfa_exc, J_sfa_inh])

# define connectivity parameters:
J = 0.3  # excitatory synaptic weight in mV if number of input connections is C0 (see below)
g = 5.0  # inhibition-to-excitation ratio
pconn = 0.2 * np.ones((M, M)) # connection probability
delay = 1.0 * np.ones((M, M)) # synaptic delay in ms

C0 = np.array([[800, 200], [800, 200]]) * 0.2  # constant reference matrix
C = np.vstack((N, N)) * pconn                  # numbers of input connections

# final synaptic weights scaling as 1/C:
J_syn = np.array([[J, -g * J], [J, -g * J]]) * C0 / C

taus1_ = [3.0, 6.0]  # time constants of exc./inh. postsynaptic currents (PSCs)
taus1 = np.array([taus1_ for k in range(M)])

# step current input:
step = [[20.0], [20.0]]  # jump size of mu in mV
tstep = np.array([[1500.0], [1500.0]])  # times of jumps

# synaptic time constants of excitatory and inhibitory connections:
tau_ex = 3.0  # in ms
tau_in = 6.0  # in ms

# create the populations of GIF neurons:
nest_pops = nest.Create("gif_pop_psc_exp", M)

C_m = 250.0  # irrelevant value for membrane capacity, cancels out in simulation
g_L = C_m / tau_m

params = [
    {"C_m": C_m,
     "I_e": mu[i] * g_L[i],
     "lambda_0": c[i],  # in Hz!
     "Delta_V": Delta_u[i],
     "tau_m": tau_m[i],
     "tau_sfa": tau_theta[i],
     "q_sfa": J_theta[i] / tau_theta[i],  # [J_theta]= mV*ms -> [q_sfa]=mV
     "V_T_star": V_th[i],
     "V_reset": V_reset[i],
     "len_kernel": -1,  # -1 triggers automatic history size
     "N": N[i],
     "t_ref": t_ref[i],
     "tau_syn_ex": max([tau_ex, dt]),
     "tau_syn_in": max([tau_in, dt]),
     "E_L": 0.0}
    for i in range(M)]
nest_pops.set(params)

# connect the populations:
g_syn = np.ones_like(J_syn)  # synaptic conductance
g_syn[:, 0] = C_m / tau_ex
g_syn[:, 1] = C_m / tau_in
for i in range(M):
    for j in range(M):
        nest.Connect(
            nest_pops[j],
            nest_pops[i],
            syn_spec={"weight": J_syn[i, j] * g_syn[i, j] * pconn[i, j], "delay": delay[i, j]})

# monitor the output using a multimeter (this only records with dt_rec!):
nest_mm = nest.Create("multimeter")
nest_mm.set(record_from=["n_events", "mean"], interval=dt_rec)
nest.Connect(nest_mm, nest_pops)

# monitor the output using a spike recorder:
spikerecorder = []
for i in range(M):
    spikerecorder.append(nest.Create("spike_recorder"))
    spikerecorder[i].time_in_steps = True
    nest.Connect(nest_pops[i], spikerecorder[i], syn_spec={"weight": 1.0, "delay": dt})

# set initial value (at t0+dt) of step current generator to zero:
tstep = np.hstack((dt * np.ones((M, 1)), tstep))
step  = np.hstack((np.zeros((M, 1)), step))

# create the step current devices:
nest_stepcurrent = nest.Create("step_current_generator", M)
# set the parameters for the step currents
for i in range(M):
    nest_stepcurrent[i].set(amplitude_times=tstep[i] + t0, amplitude_values=step[i] * g_L[i], origin=t0, stop=t_end)
    pop_ = nest_pops[i]
    nest.Connect(nest_stepcurrent[i], pop_, syn_spec={"weight": 1.0, "delay": dt})

# simulate the network:
t = np.arange(0.0, t_end, dt_rec)
A_N = np.ones((t.size, M)) * np.nan
Abar = np.ones_like(A_N) * np.nan
nest.Simulate(t_end + dt) # simulate 1 step longer to make sure all t are simulated:

# extract the data from the multimeter and the spike recorder:
data_mm = nest_mm.events
for i, nest_i in enumerate(nest_pops):
    a_i = data_mm["mean"][data_mm["senders"] == nest_i.global_id]
    a = a_i / N[i] / dt
    min_len = np.min([len(a), len(Abar)])
    Abar[:min_len, i] = a[:min_len]

    data_sr = spikerecorder[i].get("events", "times")
    data_sr = data_sr * dt - t0
    bins = np.concatenate((t, np.array([t[-1] + dt_rec])))
    A = np.histogram(data_sr, bins=bins)[0] / float(N[i]) / dt_rec
    A_N[:, i] = A
    
""" # plots:
plt.figure(1, figsize=(6, 5))
# plot population activities (in Hz):
plt.subplot(2, 1, 1)
plt.plot(t, A_N[:,0] * 1000, label=f"$A_N$ population activity")
plt.plot(t, Abar[:,0] * 1000, label=f"$\\bar A$ instantaneous population rate")
#plt.plot(t, A_N[:,1] * 1000, '-', alpha=0.5, label="inhibitory population")
plt.ylabel(f"$A_N$ / $\\bar A$ [Hz]")
#plt.xticks([])
plt.annotate("excitatory population", xy=(0.5, 0.96), xycoords="axes fraction", ha="center", va="center")
plt.legend(frameon=False, loc="upper left")
plt.title("Population activities (mesoscopic sim.)")

# plot instantaneous population rates (in Hz):
plt.subplot(2, 1, 2)
plt.plot(t, A_N[:,1] * 1000, label=f"$A_N$")
plt.plot(t, Abar[:,1] * 1000, label=f"$\\bar A$")
#plt.plot(t, A_N[:,1] * 1000, '-', alpha=0.5, label="inhibitory population")
plt.ylabel(f"$A_N$ / $\\bar A$ [Hz]")
plt.xlabel("time [ms]")
plt.annotate("inhibitory population", xy=(0.5, 0.96), xycoords="axes fraction", ha="center", va="center")
plt.tight_layout()
plt.savefig("figures/population_rate_model_population_activity.png", dpi=200) """
# %% MICROSCOPIC SIMULATION
nest.ResetKernel()
nest.resolution = dt
nest.print_time = False
nest.local_num_threads = 1
nest.rng_seed = 1

t0 = nest.biological_time

# create the 2 populations of GIF neurons (excitatory and inhibitory):
nest_pops = []
for k in range(M):
    nest_pops.append(nest.Create("gif_psc_exp", N[k]))

# set single neuron properties:
for i in range(M):
    nest_pops[i].set(
        C_m=C_m,
        I_e=mu[i] * g_L[i],
        lambda_0=c[i],
        Delta_V=Delta_u[i],
        g_L=g_L[i],
        tau_sfa=tau_theta[i],
        q_sfa=J_theta[i] / tau_theta[i],
        V_T_star=V_th[i],
        V_reset=V_reset[i],
        t_ref=t_ref[i],
        tau_syn_ex=max([tau_ex, dt]),
        tau_syn_in=max([tau_in, dt]),
        E_L=0.0,
        V_m=0.0)

# connect the populations:
for i, nest_i in enumerate(nest_pops):
    for j, nest_j in enumerate(nest_pops):
        if np.allclose(pconn[i, j], 1.0):
            conn_spec = {"rule": "all_to_all"}
        else:
            conn_spec = {"rule": "fixed_indegree", "indegree": int(pconn[i, j] * N[j])}

        nest.Connect(nest_j, nest_i, conn_spec, syn_spec={"weight": J_syn[i, j] * g_syn[i, j], "delay": delay[i, j]})

# monitor the output using a multimeter and a spike recorder:
spikerecorder = []
for i, nest_i in enumerate(nest_pops):
    spikerecorder.append(nest.Create("spike_recorder"))
    spikerecorder[i].time_in_steps = True

    # record all spikes from population to compute population activity
    nest.Connect(nest_i, spikerecorder[i], syn_spec={"weight": 1.0, "delay": dt})

# record the membrane potential of the first Nrecord neurons of each population:
Nrecord = [5, 0]  # for each population "i" the first Nrecord[i] neurons are recorded
multimeter = []
for i, nest_i in enumerate(nest_pops):
    multimeter.append(nest.Create("multimeter"))
    multimeter[i].set(record_from=["V_m"], interval=dt_rec)
    if Nrecord[i] != 0:
        nest.Connect(multimeter[i], nest_i[: Nrecord[i]], syn_spec={"weight": 1.0, "delay": dt})

# create the step current devices and set its parameters:
nest_stepcurrent = nest.Create("step_current_generator", M)
for i in range(M):
    nest_stepcurrent[i].set(amplitude_times=tstep[i] + t0, amplitude_values=step[i] * g_L[i], origin=t0, stop=t_end)
    nest_stepcurrent[i].set(amplitude_times=tstep[i] + t0, amplitude_values=step[i] * g_L[i], origin=t0, stop=t_end)
    # optionally a stopping time may be added by: 'stop': sim_T + t0
    pop_ = nest_pops[i]
    nest.Connect(nest_stepcurrent[i], pop_, syn_spec={"weight": 1.0, "delay": dt})


# simulate 1 step longer to make sure all t are simulated
nest.Simulate(t_end + dt)

# extract the data from the spike recorder:
t_micro = np.arange(0.0, t_end, dt_rec)
A_N_micro = np.ones((t.size, M)) * np.nan
for i in range(len(nest_pops)):
    data_sr = spikerecorder[i].get("events", "times") * dt - t0
    bins = np.concatenate((t_micro, np.array([t[-1] + dt_rec])))
    A = np.histogram(data_sr, bins=bins)[0] / float(N[i]) / dt_rec
    A_N_micro[:, i] = A * 1000  # in Hz


# plot excitatory population:
plt.figure(1, figsize=(6, 5))
plt.subplot(2, 1, 1)
plt.plot(t, A_N[:,0] * 1000, label=f"$A_N$ population activity")
plt.plot(t, Abar[:,0] * 1000, label=f"$\\bar A$ instantaneous population rate")
plt.ylabel(f"population activity\n[Hz]")
#plt.xticks([])
plt.annotate(f"mesoscopic\nsimulation", xy=(0.75, 0.85), fontweight="normal",
             xycoords="axes fraction", ha="left", va="center")
plt.legend(frameon=False, loc="upper left")
plt.ylim([0, 120])
plt.yticks(np.arange(0, 101, 25))
plt.title("Population activities (excitatory population)")

plt.subplot(2, 1, 2)
plt.plot(t_micro, A_N_micro[:,0], label=f"$A_N$")
#plt.plot(t, A_N[:,1] * 1000, '-', alpha=0.5, label="inhibitory population")
plt.ylabel(f"population activity\n[Hz]")
plt.xlabel("time [ms]")
plt.annotate(f"microscopic\nsimulation", xy=(0.75, 0.85), fontweight="normal",
             xycoords="axes fraction", ha="left", va="center")
plt.ylim([0, 120])
plt.yticks(np.arange(0, 101, 25))
plt.tight_layout()
plt.savefig("figures/population_rate_model_population_activity_excitatory.png", dpi=200)

# plot inhibitory population:
plt.figure(2, figsize=(6, 5))
plt.subplot(2, 1, 1)
plt.plot(t, A_N[:,1] * 1000, label=f"$A_N$ population activity")
plt.plot(t, Abar[:,1] * 1000, label=f"$\\bar A$ instantaneous population rate")
#plt.plot(t, A_N[:,1] * 1000, '-', alpha=0.5, label="inhibitory population")
plt.ylabel(f"population activity\n[Hz]")
#plt.xticks([])
plt.annotate(f"mesoscopic\nsimulation", xy=(0.75, 0.85), fontweight="normal",
             xycoords="axes fraction", ha="left", va="center")
plt.legend(frameon=False, loc="upper left")
plt.ylim([0, 120])
plt.yticks(np.arange(0, 101, 25))
plt.title("Population activities (inhibitory population)")

# plot instantaneous population rates (in Hz):
plt.subplot(2, 1, 2)
plt.plot(t_micro, A_N_micro[:,1], label=f"$A_N$")
#plt.plot(t, A_N[:,1] * 1000, '-', alpha=0.5, label="inhibitory population")
plt.ylabel(f"population activity\n[Hz]")
plt.xlabel("time [ms]")
plt.annotate(f"microscopic\nsimulation", xy=(0.75, 0.85), fontweight="normal",
             xycoords="axes fraction", ha="left", va="center")
plt.ylim([0, 120])
plt.yticks(np.arange(0, 101, 25))
plt.tight_layout()
plt.savefig("figures/population_rate_model_population_activity_inhibitory.png", dpi=200)



# extract the membrane potential of the first Nrecord neurons of the first population:
voltage = []
for i in range(M):
    if Nrecord[i] > 0:
        senders = multimeter[i].get("events", "senders")
        v = multimeter[i].get("events", "V_m")
        voltage.append(np.array([v[np.where(senders == j)] for j in set(senders)]))
    else:
        voltage.append(np.array([]))

# plot the membrane potential of the first Nrecord neurons of the first population:
f, ax = plt.subplots(Nrecord[0], sharex=True, figsize=(6, 5))
for i in range(Nrecord[0]):
    ax[i].plot(voltage[0][i])
    ax[i].set_yticks((0, 15, 30))
ax[i].set_xlabel("time [ms]")
ax[2].set_ylabel("membrane potential [mV]")
ax[0].set_title("5 example GIF neurons (microscopic sim.)")
plt.tight_layout()
plt.savefig("figures/population_rate_model_microscopic_membrane_potential.png", dpi=200)
plt.show()
# %% END