# Release notes for the Neural Dynamics repository

## 🚀 Release v1.0.2
In this release, we added a new script `nervos_snn_mnist.py` that implements a spiking neural network (SNN) for MNIST pattern recognition using the `nervos` library. This script provides an end-to-end workflow for configuring experiment parameters, training the SNN, and analyzing the results with additional utilities for diagnostics and plots.

## 🚀 Release v1.0.1
In this release, we added two new scripts `stdp_weight_plot.py` and  `stdp_simple_network_example.py` to the repository that implement a simple spiking neural network example using spike-timing-dependent plasticity (STDP). These scripts are designed to illustrate the basic principles of STDP and how it can lead to learning in a neural network.

## 🚀 Release v1.0.0
This is the initial release of the Neural Dynamics repository, featuring a comprehensive collection of educational Python scripts for computational neuroscience. The release includes tutorials on using the NEST Simulator, as well as standalone scripts implementing various neuron models and network dynamics.

### 📦 Scope and content
This release includes educational Python scripts covering a broad range of core topics in neural dynamics, such as:
- Single neuron models (Integrate-and-Fire, Izhikevich, AdEx, EIF)
- Spiking neural networks (Brunel network, oscillatory dynamics)
- Synaptic plasticity rules (BCM rule)
- Gap junctions and their role in neural modeling
- Rate models for collective neural activity

The repository structure reflects this thematic organization and mirrors the progression of the blog series.

### 🧠 Conceptual focus
The scripts in this repository are designed as **didactic and conceptual examples**. Emphasis is placed on:

* Clarity and interpretability of models
* Step-by-step explanations accompanying each script
* Reproducibility of classic results in theoretical neuroscience
* Encouraging experimentation and exploration of neural dynamics concepts
* Bridging theoretical neuroscience with practical implementation

Many models deliberately rely on reduced geometries, simplified boundary conditions, or idealized assumptions to keep the underlying mechanisms explicit.


### 🔬 Reproducibility and usage
All scripts are compatible with a lightweight Python environment based on NumPy, SciPy, and Matplotlib, along with the NEST Simulator for spiking neural network simulations. Instructions for setting up the environment and running the scripts are provided in the README file. The scripts are written to support both direct execution and interactive, cell-by-cell exploration in development environments such as VS Code or Jupyter.

This release provides a stable baseline for reuse in:

* teaching and coursework
* self-study
* illustrative figures and animations
* methodological extensions

Backward compatibility across future releases is not guaranteed, but changes will primarily serve conceptual clarification rather than feature expansion.

### 📝 License
All code is released under the GPL-3.0 License.

### ✨ Outlook
Future releases may expand individual examples, refine numerical implementations, or add complementary scripts aligned with new blog posts. Any such extensions will build on the conceptual baseline established with this release.