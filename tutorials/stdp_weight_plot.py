""" 
Graphical illustration of the STDP learning window W(Δt).

Notation follows the mathematical formulation in the accompanying blog post:
- presynaptic neuron: i, spike times t_i^f
- postsynaptic neuron: j, spike times t_j^n
- relative spike timing: Δt = t_j^n - t_i^f

Positive Δt corresponds to t_i^f < t_j^n (pre fires before post, LTP),
negative Δt corresponds to t_j^n < t_i^f (post fires before pre, LTD).

author: Fabrizio Musacchio
date: Feb 10, 2026
"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt

# set global properties for all plots:
plt.rcParams.update({'font.size': 12})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False
# %% MAIN PLOT
RESULTS_PATH = "figures"
os.makedirs(RESULTS_PATH, exist_ok=True)

# --- STDP parameters ---
tau_plus  = 20.0
tau_minus = 20.0
A_plus    = 0.01
A_minus   = 0.0105

# relative spike timing Δt = t_j^n - t_i^f
delta_t = np.linspace(-100, 100, 2000)

# STDP learning window
W = np.where(
    delta_t > 0,
    A_plus  * np.exp(-delta_t / tau_plus),
   -A_minus * np.exp( delta_t / tau_minus)
)

# --- Figure layout ---
fig, (ax_spike, ax_std) = plt.subplots(2, 1,
        figsize=(5.5, 5),
        sharex=True,
        gridspec_kw=dict(height_ratios=[1, 3], hspace=0.05))

# ======================
# Spike timing subplot
# ======================
ax_spike.set_ylim(-0.5, 1.5)
ax_spike.axis("off")

# Horizontal timelines (schematic, Δt-axis)
ax_spike.hlines(1.0, -80, 100, linewidth=1.5, color="k")
ax_spike.hlines(0.0, -80, 100, linewidth=1.5, color="k")

# Color coding: functional regimes
c_ltp = "tab:red"    # Δt > 0, LTP
c_ltd = "tab:green"  # Δt < 0, LTD

# LTD example: t_j^n < t_i^f (post fires before pre)
ax_spike.vlines(-40, 0.85, 1.15, color=c_ltd, linewidth=4)   # presynaptic spike t_i^f
ax_spike.vlines(-10, -0.15, 0.15, color=c_ltd, linewidth=4)  # postsynaptic spike t_j^n

# LTP example: t_i^f < t_j^n (pre fires before post)
ax_spike.vlines( 10, 0.85, 1.15, color=c_ltp, linewidth=4)   # presynaptic spike t_i^f
ax_spike.vlines( 40, -0.15, 0.15, color=c_ltp, linewidth=4)  # postsynaptic spike t_j^n

# Vertical reference line at Δt = 0
ax_spike.axvline(0, color="k", linewidth=1)

# Labels
ax_spike.text(-85, 1.0, r"pre-syn. spike $t_i^f$",  va="center", ha="right")
ax_spike.text(-85, 0.0, r"post-syn. spike $t_j^n$", va="center", ha="right")


# ======================
# STDP window subplot
# ======================
mask_pos = delta_t > 0
mask_neg = delta_t < 0

# Plot branches separately to avoid connection across Δt = 0
ax_std.plot(delta_t[mask_pos], W[mask_pos], linewidth=2, color=c_ltp)
ax_std.plot(delta_t[mask_neg], W[mask_neg], linewidth=2, color=c_ltd)

# Emphasize LTP/LTD regions
ax_std.fill_between(delta_t[mask_pos], 0, W[mask_pos], alpha=0.15, color=c_ltp)
ax_std.fill_between(delta_t[mask_neg], 0, W[mask_neg], alpha=0.15, color=c_ltd)

# Reference axes
ax_std.axhline(0, color="k", linewidth=1)
ax_std.axvline(0, color="k", linewidth=1)

ax_std.set_xlabel(r"$\Delta t = t_j^n - t_i^f$ (ms)")
ax_std.set_ylabel(r"$W(\Delta t)$")

# Annotations
ax_std.text( 55, 0.6*np.max(W), "LTP\n(pre fires\nbefore post)", ha="center")
ax_std.text(-55, 0.6*np.min(W), "LTD\n(post fires\nbefore pre)", ha="center")

# Layout adjustments
plt.subplots_adjust(left=0.20, right=1.0, top=0.98, bottom=0.12) # the plot is a bit tight, so we adjust a little
plt.savefig(os.path.join(RESULTS_PATH, "stdp_window.png"), dpi=300)
plt.close()

# %% END