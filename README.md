# NEST Simulator tutorials

This is a collection of tutorials showing how to use the [NEST Simulator](https://nest-simulator.readthedocs.io/en/stable/index.html). You can find a detailed description for each tutorial in the following blog posts:

* [NEST simulator – A powerful tool for simulating large-scale spiking neural networks](https://www.fabriziomusacchio.com/blog/2024-06-09-nest_SNN_simulator/)
* [Step-by-step NEST single neuron simulation](https://www.fabriziomusacchio.com/blog/2024-06-16-nest_single_neuron_example/)
* [Connection concepts in NEST](https://www.fabriziomusacchio.com/blog/2024-06-25-nest_connection_concepts/)
* [Izhikevich SNN simulated with NEST](https://www.fabriziomusacchio.com/blog/2024-06-30-nest_izhikevich_snn/)
* [Oscillatory population dynamics of GIF neurons simulated with NEST](https://www.fabriziomusacchio.com/blog/2024-07-14-oscillating_gif_neuron_population/)
* [Brunel network: A comprehensive framework for studying neural network dynamics](https://www.fabriziomusacchio.com/blog/2024-07-21-brunel_network/)
* [Example of a neuron driven by an inhibitory and excitatory neuron population](https://www.fabriziomusacchio.com/blog/2024-07-28-neuron_driven_by_populations/)
* [What are alpha-shaped post-synaptic currents?](https://www.fabriziomusacchio.com/blog/2024-08-04-alpha_shaped_input_currents/)
* [Frequency-current (f-I) curves](https://www.fabriziomusacchio.com/blog/2024-08-11-fi_curves/)
* [Olfactory processing via spike-time based computation](https://www.fabriziomusacchio.com/blog/2024-08-21-olfactory_processing_via_spike_time_bases_computation/)
* [Exponential (EIF) and adaptive exponential Integrate-and-Fire (AdEx) model](https://www.fabriziomusacchio.com/blog/2024-08-25-EIF_and_AdEx_model/)
* [Campbell and Siegert approximation for estimating the firing rate of a neuron](https://www.fabriziomusacchio.com/blog/2024-09-01-campbell_siegert_approximation/)
* [Bienenstock-Cooper-Munro (BCM) rule](https://www.fabriziomusacchio.com/blog/2024-09-08-bcm_rule/)
* [On the role of gap junctions](https://www.fabriziomusacchio.com/blog/2025-08-15-gap_junctions/)
* [On the role of gap junctions in neural modelling: Network example](https://www.fabriziomusacchio.com/blog/2025-08-17-gap_junctions_network_example/)
* [Rate models as a tool for studying collective neural activity](https://www.fabriziomusacchio.com/blog/2025-08-28-rate_models/)



For reproducibility:

```bash
conda create -n nest -y python=3.11 mamba
conda activate nest
mamba install -y ipykernel matplotlib numpy pandas nest-simulator
```


![img](thumb/single_neuron_iaf_psc_alpha_Ie_376.0.png)