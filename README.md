# Neural Dynamics: Educational Python scripts for Computational Neuroscience

![img](thumb/single_neuron_iaf_psc_alpha_Ie_376.0.png)

## What is neural dynamics?
Neural dynamics studies how the activity of neurons and neural networks evolves over time. It focuses on the mathematical and computational description of spiking behavior, membrane potentials, oscillations, synchronization, attractor states, and learning through synaptic plasticity. Typical approaches include differential-equation-based neuron models, spiking neural networks, and dynamical-systems analysis. Neural dynamics forms a central part of computational neuroscience, providing theoretical tools to understand how neural circuits generate computation, memory, and behavior.



## Purpose of this collection
This repository provides a collection of educational Python scripts that explore core concepts in neural dynamics. The focus is on interpretable models and simulations such as integrate-and-fire neurons, spiking neural networks, and plasticity rules. The goal is to offer concise, well-documented examples that support learning, experimentation, and theoretical exploration, rather than large-scale machine-learning frameworks. The scripts are intended for everyone who want to build intuition for dynamical processes in neural systems and reproduce classic results from theoretical neuroscience.

Each script in this repository is accompanied by a detailed blog post that explains the underlying concepts, mathematical formulations, and implementation details. The blog posts provide context and guidance to help you to understand the significance of each simulation and how it relates to broader topics in computational neuroscience.

## Contents

### NEST Simulator tutorials
This is a collection of tutorials showing how to use the [NEST Simulator](https://nest-simulator.readthedocs.io/en/stable/index.html). Each tutorial is usually based on the corresponding original tutorial from the NEST Simulator documentation, but modified and expanded to include more explanations and additional examples.

* [NEST simulator – A powerful tool for simulating large-scale spiking neural networks](https://www.fabriziomusacchio.com/blog/2024-06-09-nest_SNN_simulator/) (overview article)
* [Step-by-step NEST single neuron simulation](https://www.fabriziomusacchio.com/blog/2024-06-16-nest_single_neuron_example/)
    * ⟶ `tutorials/single_neuron_simulation.py`
* [Connection concepts in NEST](https://www.fabriziomusacchio.com/blog/2024-06-25-nest_connection_concepts/) (overview article)
* [Izhikevich SNN simulated with NEST](https://www.fabriziomusacchio.com/blog/2024-06-30-nest_izhikevich_snn/)
    * ⟶ `tutorials/izhikevich_snn.py`
* [Oscillatory population dynamics of GIF neurons simulated with NEST](https://www.fabriziomusacchio.com/blog/2024-07-14-oscillating_gif_neuron_population/)
    * ⟶ `tutorials/gif_neuron_model_with_oscillations.py`
* [Brunel network: A comprehensive framework for studying neural network dynamics](https://www.fabriziomusacchio.com/blog/2024-07-21-brunel_network/)
    * ⟶ `tutorials/brunel_network.py`
* [Example of a neuron driven by an inhibitory and excitatory neuron population](https://www.fabriziomusacchio.com/blog/2024-07-28-neuron_driven_by_populations/)
    * ⟶ `tutorials/neuron_with_population_inputs.py`
* [What are alpha-shaped post-synaptic currents?](https://www.fabriziomusacchio.com/blog/2024-08-04-alpha_shaped_input_currents/)
    * ⟶ `tutorials/alpha_function.py`
* [Frequency-current (f-I) curves](https://www.fabriziomusacchio.com/blog/2024-08-11-fi_curves/)
    * ⟶ `tutorials/fi_curve.py`
* [Olfactory processing via spike-time based computation](https://www.fabriziomusacchio.com/blog/2024-08-21-olfactory_processing_via_spike_time_bases_computation/)
    * ⟶ `tutorials/spike_synchronization_through_oscillation.py`
* [Exponential (EIF) and adaptive exponential Integrate-and-Fire (AdEx) model](https://www.fabriziomusacchio.com/blog/2024-08-25-EIF_and_AdEx_model/)
    * ⟶ `tutorials/aeif_neuron.py`
    * ⟶ `tutorials/aeif_neuron_multple_rices_and_decays.py`
* [Campbell and Siegert approximation for estimating the firing rate of a neuron](https://www.fabriziomusacchio.com/blog/2024-09-01-campbell_siegert_approximation/)
    * ⟶ `tutorials/campbell_siegert_approximation.py`
* [Bienenstock-Cooper-Munro (BCM) rule](https://www.fabriziomusacchio.com/blog/2024-09-08-bcm_rule/)
    * ⟶ `tutorials/bcm_rule.py`
    * ⟶ `tutorials/bcm_rule_with_decay_term.py`
* [On the role of gap junctions](https://www.fabriziomusacchio.com/blog/2025-08-15-gap_junctions/)
    * ⟶ `tutorials/two_neurons_with_gap_junctions.py`
* [On the role of gap junctions in neural modelling: Network example](https://www.fabriziomusacchio.com/blog/2025-08-17-gap_junctions_network_example/)
    * ⟶ `tutorials/gap_junctions_network_example.py`
* [Rate models as a tool for studying collective neural activity](https://www.fabriziomusacchio.com/blog/2025-08-28-rate_models/)
* Incorporating structural plasticity in neural network models (coming soon
    * ⟶ `tutorials/structural_plasticity.py`

### General computational neuroscience scripts
The following scripts are not based on the NEST Simulator, but rather implement models and simulations from scratch using standard Python libraries such as NumPy and Matplotlib (but not limited to them).

* [Building a neural network from scratch using NumPy](https://www.fabriziomusacchio.com/blog/2024-02-25-ann_from_scratch_using_numpy/)
  * ⟶ not part of this repository, but you can find it [here](https://github.com/FabrizioMusacchio/ann_with_numpy)
* [Understanding Hebbian learning in Hopfield networks](https://www.fabriziomusacchio.com/blog/2024-03-03-hebbian_learning_and_hopfield_networks/)
  * ⟶ not part of this repository, but you can find it [here](https://github.com/FabrizioMusacchio/hopfield_network)
* [Integrate and Fire Model: A simple neuronal model](https://www.fabriziomusacchio.com/blog/2023-07-03-integrate_and_fire_model/)
  * ⟶ not part of this repository, but you can find it [here](https://github.com/FabrizioMusacchio/integrate_and_fire_model/tree/master)
* [Rössler attractor](https://www.fabriziomusacchio.com/blog/2024-03-10-roessler_attractor/)
  * ⟶ not part of this repository, but you can find it [here](https://github.com/FabrizioMusacchio/roessler_attractor)
* [Using phase plane analysis to understand dynamical systems](https://www.fabriziomusacchio.com/blog/2024-03-17-phase_plane_analysis/)
  * ⟶ not part of this repository, but you can find it [here](https://github.com/FabrizioMusacchio/phase_plane_analysis)
* [Nullclines and fixed points of the Rössler attractor](https://www.fabriziomusacchio.com/blog/2024-03-19-roesler_attractor_nullcines_and_fixed_points/)
  * ⟶ not part of this repository, but you can find it [here](https://github.com/FabrizioMusacchio/roessler_attractor)
* [Van der Pol oscillator](https://www.fabriziomusacchio.com/blog/2024-03-24-van_der_pol_oscillator/)
  * ⟶ not part of this repository, but you can find it [here](https://github.com/FabrizioMusacchio/phase_plane_analysis)
* [FitzHugh-Nagumo model](https://www.fabriziomusacchio.com/blog/2024-04-07-fitzhugh_nagumo_model/)
  * ⟶ not part of this repository, but you can find it [here](https://github.com/FabrizioMusacchio/phase_plane_analysis)
* [Hodgkin-Huxley model](https://www.fabriziomusacchio.com/blog/2024-04-21-hodgkin_huxley_model/)
  * ⟶ not part of this repository, but you can find it [here](https://github.com/FabrizioMusacchio/Hodgkin-Huxley-Model)
* [Izhikevich model](https://www.fabriziomusacchio.com/blog/2024-04-29-izhikevich_model/)
  * ⟶ not part of this repository, but you can find it [here](https://www.fabriziomusacchio.com/blog/2024-04-29-izhikevich_model/)
* [Simulating spiking neural networks with Izhikevich neurons](https://www.fabriziomusacchio.com/blog/2024-05-19-izhikevich_network_model/)
  * ⟶ not part of this repository, but you can find it [here](https://github.com/FabrizioMusacchio/izhikevich_model)


## Installation
For reproducibility:

```bash
conda create -n nest -y python=3.11 mamba
conda activate nest
mamba install -y ipykernel matplotlib numpy pandas nest-simulator
```

## Usage
Each script can be run directly using the Python environment described above. In particular, they are written in such a way, that they can be interactively executed cell-by-cell, e.g., in VS Code's interactive window. You can also place them in a Jupyter notebook for step-by-step execution.

## Citation
If you use code from this repository for your own research, teaching material, or derived software, please consider citing the Zenodo archive associated with this repository (WILL BE ADDED SOON). Proper citation helps acknowledge the original source, provides context for the implemented physical models and numerical assumptions, and supports reproducibility.

When appropriate, citing the specific blog post that discusses the underlying concepts and numerical methods in detail is encouraged in addition to the repository itself.

If you use substantial parts of the code in an academic publication, a reference to both the repository and the associated blog article is recommended.

Here is the suggested citation format for the repository:

> WILL BE ADDED SOON (Feb 2026)


Thank you for considering proper citation practices.

## Contact and support
For questions or suggestions, please open an issue on GitHub or contact the author via email: [Fabrizio Musacchio](mailto:fabrizio.musacchio@posteo.de)
