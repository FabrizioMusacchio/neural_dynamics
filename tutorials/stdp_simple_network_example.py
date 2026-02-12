"""
Spike-Timing-Dependent Plasticity (STDP) vs. No-STDP demonstration
=================================================================

This script demonstrates the effect of spike-timing-dependent plasticity (STDP)
on synaptic self-organization in a minimal spiking network model and contrasts it
with an otherwise identical control condition without plasticity.

Model overview
--------------

The network consists of:
* a population of N independent presynaptic neurons generating Poisson spike trains,
* a single postsynaptic leaky integrate-and-fire (LIF) neuron,
* all-to-one excitatory synaptic connectivity from the presynaptic population to the postsynaptic neuron.

Synapses are bounded between 0 and gmax and optionally implement an STDP learning rule.

Two variants are implemented
-----------------------------

1) **STDP-enabled network**
   Synapses implement a classical pair-based STDP rule using presynaptic and
   postsynaptic eligibility traces. Weight updates are driven by the relative timing
   of spikes:
   * presynaptic spikes interact with the postsynaptic trace,
   * postsynaptic spikes interact with the presynaptic trace.

   This leads to:
   * competition between synapses,
   * spontaneous differentiation of synaptic strengths,
   * a non-uniform final weight distribution,
   * characteristic drift and saturation of individual synaptic weights over time.

2) **No-STDP control network**
   The same network architecture and input statistics are used, but synaptic weights
   are fixed after initialization. No plasticity is applied.

   This leads to:
   * synaptic weights remaining at their initial random values,
   * no self-organization or competition,
   * a stationary weight distribution,
   * flat weight trajectories over time.

What is shown in the plots
-------------------------

For each variant, the following diagnostics are produced:

* **Final synaptic weights vs. synapse index**  
  Shows the distribution of synaptic strengths across all presynaptic inputs after
  the simulation.

* **Histogram of synaptic weights**  
  Visualizes whether the weight distribution remains uniform (no STDP) or becomes
  structured and bimodal (with STDP).

* **Example synaptic weight trajectories**  
  Tracks the temporal evolution of a small number of individual synapses to illustrate
  either plastic drift (STDP) or constancy (no STDP).

Interpretation
--------------

This script illustrates a central conceptual point of STDP:

STDP is not merely a local weight update rule, but a mechanism for *self-organization*
driven by spike timing. Even under stationary Poisson input, STDP induces competition
between synapses and leads to structured synaptic states. In contrast, without STDP,
the same network remains statistically identical to its initialization.

The example is intentionally minimalistic and pedagogical. It is not intended to model
a specific biological circuit, but to isolate and visualize the qualitative consequences
of spike-timing-dependent plasticity.

Installation
------------
For reproducibility, create a new conda environment and install the required packages:

```bash
conda create -n stdp_example python=3.10
conda activate stdp_example
conda install -c conda-forge brian2 matplotlib numpy
```

I recommend to run this script cell by cell, e.g., in VS Code's interactive Python mode.

Acknowledgments
----------------
The STDP implementation is adopted from the Brian2 documentation:

https://brian2.readthedocs.io/en/stable/examples/synapses.STDP.html

author: Fabrizio Musacchio
date: Feb 10, 2026
"""
# %% IMPORTS
import os
from pdb import run
from turtle import clear
#from brian2 import *
import brian2 as b2
from brian2 import ms, mV, nS, Hz, second
import matplotlib.pyplot as plt

# set global properties for all plots:
plt.rcParams.update({'font.size': 12})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False
# %% PARAMETERS
N = 1000            # number of presynaptic neurons
taum = 10*ms        # membrane time constant
taupre = 20*ms      # STDP time constant for presynaptic trace
taupost = taupre    # STDP time constant for postsynaptic trace (often set equal to taupre)
Ee = 0*mV           # excitatory reversal potential
vt = -54*mV         # spike threshold
vr = -60*mV         # reset potential
El = -74*mV         # leak reversal potential
taue = 5*ms         # excitatory synaptic time constant
F = 15*Hz           # firing rate of Poisson input
gmax = .01          # maximum synaptic weight
dApre = .01         # increment applied to the presynaptic eligibility trace Apre on each presynaptic spike (sets the scale of potentiation via Apre)
dApost = -dApre * taupre / taupost * 1.05  # increment applied to the postsynaptic eligibility trace Apost on each postsynaptic spike (negative; slightly stronger magnitude as a stabilizing heuristic)
dApost *= gmax      # scale trace increments to the same order of magnitude as w (since Apre/Apost are added directly to w)
dApre *= gmax       # same scaling for Apre

RESULTS_PATH = "figures"
os.makedirs(RESULTS_PATH, exist_ok=True)
# %% EXAMPLE 1: STDP

""" 
Define the brian2 model with STDP synapses. 

The concrete model used here is a simple leaky integrate-and-fire neuron 
receiving input from a population of Poisson spike generators, with STDP 
implemented on the synapses. This is a common setup for demonstrating STDP 
in a simple network. The parameters (e.g., time constants, learning rates) 
are chosen to be in a reasonable range for this type of model.

We will constrain the synaptic weights to be between 0 and gmax. We 
will also set up monitors to record the synaptic weights over time 
and the presynaptic spike times.
"""
eqs_neurons = '''
dv/dt = (ge * (Ee-v) + El - v) / taum : volt
dge/dt = -ge / taue : 1
'''

""" 
Here, we use a simple all-to-one network architecture with a single 
postsynaptic neuron receiving input from a population of presynaptic 
neurons. The presynaptic neurons fire independent Poisson spike trains, 
and the synapses between the presynaptic population and the postsynaptic 
neuron implement STDP.
"""

poisson_input = b2.PoissonGroup(N, rates=F)
neurons = b2.NeuronGroup(1, 
                         eqs_neurons, 
                         threshold='v>vt', 
                         reset='v = vr',
                         method='euler')

# Here we define the synapses with STDP. The model includes the synaptic weight w,
# and the presynaptic and postsynaptic eligibility traces Apre and Apost, which decay
# exponentially and are updated on pre- and postsynaptic spikes according to the STDP rule.
S = b2.Synapses(poisson_input, neurons,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''ge += w
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)''',
             on_post='''Apost += dApost
                     w = clip(w + Apre, 0, gmax)''',
             )
S.connect() # all-to-one connectivity from the Poisson input to the single postsynaptic neuron
S.w = 'rand() * gmax' # random initialization of weights between 0 and gmax
mon = b2.StateMonitor(S, 'w', record=[0, 1]) # record the weights of two example synapses to see how they evolve over time
# s_mon = b2.SpikeMonitor(poisson_input) # monitor presynaptic spikes to see the timing relationship with weight changes (but not used in the plots here)

# run the simulation:
b2.run(100*b2.second, report='text')

# plots:
plt.figure(figsize=(5, 8))
plt.subplot(3, 1, 1)
plt.plot(S.w / gmax, '.', c='mediumaquamarine', markersize=4)
plt.ylabel('w/gmax')
plt.xlabel('synapse index')
plt.title('STDP: weights after simulation')

plt.subplot(3, 1, 2)
plt.hist(S.w / gmax, 20, color='mediumaquamarine')
plt.xlabel('w/gmax')
plt.title('STDP: weight distribution')

plt.subplot(3, 1, 3)
plt.plot(mon.t/b2.second, mon.w[0]/gmax, label='Synapse 0')
plt.plot(mon.t/b2.second, mon.w[1]/gmax, label='Synapse 1')
plt.xlabel('t [s]')
plt.ylabel('w/gmax')
plt.ylim(0.0,1.1)
plt.legend()
plt.title('STDP: example synapses vary over time')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "stdp_example_weights.png"), dpi=300)
#plt.show()
plt.close()
# %% CLEAR MODEL AND MONITORS
# clear the previous network and monitors:
del S, mon, neurons, poisson_input

""" 
It can happen, that the command above doesn't fully clear the model and monitors 
from memory, which can lead to issues when we run the next example. If you encounter 
such issues, you can try restarting the Python kernel and re-run the script from the 
beginning, but skipping the first part with the STDP example. 
"""
# %% EXAMPLE 2: NO STDP

# we define the same network architecture and neuron model as before: 
eqs_neurons = '''
dv/dt = (ge * (Ee-v) + El - v) / taum : volt
dge/dt = -ge / taue : 1
'''

poisson_input = b2.PoissonGroup(N, rates=F)

neurons = b2.NeuronGroup(
    1, eqs_neurons,
    threshold='v > vt',
    reset='v = vr',
    method='euler')

# ...but now we will use synapses without STDP, i.e., with fixed
# weights that do not change over time:
S2 = b2.Synapses(
    poisson_input, neurons,
    model='w : 1',
    on_pre='ge += w')

S2.connect()
S2.w = 'rand() * gmax'  # same random initialization as in the STDP example

# Monitor two example synapses and their weights (should remain constant)
mon = b2.StateMonitor(S2, 'w', record=[0, 1])

# run the simulation:
b2.run(100*b2.second, report='text')

# plots:
plt.figure(figsize=(5, 8))
plt.subplot(3, 1, 1)
plt.plot(S2.w/gmax, '.', c='mediumaquamarine', markersize=4)
plt.ylabel('w/gmax')
plt.xlabel('synapse index')
plt.title('No STDP: weights remain at initialization')

plt.subplot(3, 1, 2)
plt.hist(S2.w/gmax, 20, color='mediumaquamarine')
plt.xlabel('w/gmax')
plt.title('No STDP: weight distribution stays ~uniform')

plt.subplot(3, 1, 3)
plt.plot(mon.t/b2.second, mon.w[0]/gmax, label='Synapse 0')
plt.plot(mon.t/b2.second, mon.w[1]/gmax, label='Synapse 1')
plt.xlabel('t [s]')
plt.ylabel('w/gmax')
plt.title('No STDP: example synapses are constant')
plt.ylim(0.0,1.1)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "stdp_example_no_stdp_weights.png"), dpi=300)
#plt.show()
plt.close()
# %% END