""" 
Train, analyze, and visualize a spiking neural network (SNN) on MNIST using
the `nervos` library, with extra utilities for diagnostics and plots.

This script provides an end-to-end workflow for:

1) configuring experiment parameters,
2) loading MNIST classes as spike trains,
3) training an STDP-based SNN,
4) evaluating predictions and confusion metrics,
5) inspecting internal dynamics (spike rasters, receptive fields, weight evolution,
    and membrane potential traces),
6) saving figures and model checkpoints.
A compatibility monkey patch is included to enable epoch-wise model saving on
POSIX systems (Linux/macOS) for `nervos==0.0.5`, where default save behavior
is limited.

Main components
---------------
- `MNIST_SNN`:
  Wrapper around `nv.Module` for class-filtered MNIST loading, prediction, and sample plotting.
- Visualization helpers:
  - class-summed synapses (`visualize_synapse`)
  - raster plots with highlighted winner neurons (`rasterplot`)
  - receptive fields for individual neurons (`plot_rf_of_neuron`)
  - class templates from neuron-label mapping (`plot_label_template`)
  - winner RF evolution across epochs (`plot_winner_rf_evolution_over_epochs`)
  - winner membrane potential traces (`plot_winner_membrane_potential`)
- Evaluation helpers:
  - inference loop accuracy (`accuracy`)
  - NumPy confusion matrix + plotting
  - overall and balanced accuracy metrics

Expected data/model conventions
-------------------------------
- Input images are converted to spike trains by `nervos` (shape typically `(784, T)`).
- Output layer has configurable neuron count (`output_layer_size`).
- Trained synapses are interpreted as 28x28 receptive fields per output neuron.
- Winner neuron is defined by max output spike count for a sample/epoch.

How to use
1) Install dependencies (example):
    - `numpy`, `matplotlib`, `requests`, `nervos`
2) Edit key configuration near the top:
    - `CLASSES` (e.g., `list(range(6))`)
    - `parameters_dict` (dataset sizes, epochs, STDP/neuron params, etc.)
    - `RESULTS_PATH`
3) Run the script.
4) Inspect outputs:
    - saved checkpoints in `storage/<identifier>/Epoch_<n>-<acc>/model.red`
    - plots in `RESULTS_PATH` (random samples, learned synapses, confusion matrices,
      rasters, RFs, RF evolution summaries, membrane potentials).

Important notes
---------------
- Memory usage can be high, especially with:
  - `m.get_spikeplots = True`
  - `m.get_weight_evolution = True`
  Reduce dataset sizes or disable these flags if needed.
- `nervos` simulation time is discrete (no explicit physical units).
- Membrane potentials in `m.layerpotentials` are post-update states (including reset/inhibition),
  so spike rasters are more reliable for spike-event timing interpretation.
- Epoch training order is not reshuffled by default; samples may appear in the same order each epoch.
write a docsting for the whole file, describing the purpose of this code and how to use it.

Installation
-----------

conda create -n nervos python=3.12 mamba -y
conda activate nervos
mamba install -y numpy matplotlib ipykernel requests
pip install nervos

Acknowledgements
----------------
This tutorial is based on the nervos MNIST example and extended 
with additional utilities for analysis and visualization:

https://nervos.readthedocs.io/en/latest/notebooks/mnist.html
https://github.com/jsmaskeen/nervos

Please cite nervos as: 

> Maskeen, Jaskirat Singh; Lashkare, Sandip, *A Unified Platform to 
  Evaluate STDP Learning Rule and Synapse Model using Pattern Recognition 
  in a Spiking Neural Network*, 2025, arXiv:2506.19377, 
  DOI: [10.48550/arXiv.2506.19377](https://arxiv.org/abs/2506.19377)

Author and date
---------------
author: Fabrizio Musacchio
date: Feb 2026
"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import nervos as nv

# monkey patch:
""" 
In the current version of nervos 0.0.5, the save_model function is not yet 
implemented for POSIX systems (Linux, macOS). The following code defines 
a save_epoch method that saves the model in a POSIX-compatible way, and
monkey patches the Module class to use this method. This allows us to save
models during training without modifying the nervos library itself.
"""
from pathlib import Path
def save_epoch_posix(
    self,
    epoch: int,
    synapses,
    neuron_label_map,
    accuracy: float) -> None:
    base = Path(nv.utils.common.cwd) / "storage"  # or: Path(common.cwd)
    identifier = self.identifier
    if self.str_t0 != self.identifier:
        identifier = f"{self.identifier}_{self.str_t0}"

    epoch_dir = base / identifier / f"Epoch_{epoch}-{round(accuracy,3)}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    model_path = epoch_dir / "model.red"

    # save the model using the common save_model function, which should work on all platforms:
    nv.utils.common.save_model(neuron_label_map, synapses, self.parameters, str(model_path))
# now monkey patch the Module class to use our POSIX-compatible save_epoch method:
nv.utils.module.Module.save_epoch = save_epoch_posix

# set global properties for all plots:
plt.rcParams.update({'font.size': 12})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False
# %% PARAMETERS
RESULTS_PATH = "figures"
os.makedirs(RESULTS_PATH, exist_ok=True)

# choose classes here:
CLASSES = list(range(6))   # choose any value between 1 and 10
identifier_name = f"{len(CLASSES)}classmnist"

p = nv.Parameters()
#p.from_url('https://pastebin.com/raw/X9fAjKGR')
""" 
the parameters stores at the url are:

{
    "training_images_amount": 100,
    "testing_images_amount": 20,
    "training_duration": 100,
    "past_window": -10,
    "epochs": 3,
    "image_size": [28, 28],
    "resting_potential": -70,
    "input_layer_size": 784,
    "output_layer_size": 80,
    "inhibitory_potential": -100,
    "spike_threshold": -55,
    "reset_potential": -90,
    "spike_drop_rate": 0.8,
    "threshold_drop_rate": 0.4,
    "min_weight": 1e-05,
    "max_weight": 1.0,
    "A_up": 0.8,
    "A_down": -0.3,
    "tau_up": 5,
    "tau_down": 5,
    "eta": 0.03,
    "min_frequency": 1,
    "max_frequency": 50,
    "refractory_time": 15,
    "tau_m": 10,
    "conductance": 10
}

store them directly in the parameters object, which will be passed to the MNIST_SNN module:
"""
parameters_dict = {
    "training_images_amount": 500, # nervos is very memory hungry especially when m.get_spikeplots = True and m.get_weight_evolution = True (!); either use smaller numbers here, or set those to False below, or run it on a machine with high RAM
    "testing_images_amount": 150,
    "training_duration": 100, # discrete simulation time units; nervos has no built-in concept of "real time" like s or ms.
    "past_window": -10,
    "epochs": 3,  # note: after each epoch, the training set is not reshuffled, so the same images are presented in the same order.
    "image_size": [28, 28],
    "resting_potential": -70,
    "input_layer_size": 784,
    "output_layer_size": 80,
    "inhibitory_potential": -100,
    "spike_threshold": -55,
    "reset_potential": -90,
    "spike_drop_rate": 0.8,
    "threshold_drop_rate": 0.4,
    "min_weight": 1e-05,
    "max_weight": 1.0,
    "A_up": 0.8,
    "A_down": -0.3,
    "tau_up": 5,
    "tau_down": 5,
    "eta": 0.03,
    "min_frequency": 1,
    "max_frequency": 50,
    "refractory_time": 15,
    "tau_m": 10,
    "conductance": 10
}
for key, value in parameters_dict.items():
    setattr(p, key, value)
# %% FUNCTIONS
class MNIST_SNN(nv.Module):
    def __init__(self, parameters, identifier=None, classes=None, train_size=None, test_size=None, seed=None):
        super().__init__(parameters, identifier)

        # set default (5) if not provided
        if classes is None:
            classes = list(range(5))
        self.classes = list(classes)

        self.dataloader = nv.dataloader.MNISTLoader(parameters, classes=self.classes)

        # if you want the loader sizes to be controllable from outside:
        if train_size is None:
            train_size = getattr(parameters, "training_images_amount", 100)
        if test_size is None:
            test_size = getattr(parameters, "testing_images_amount", 20)

        self.X_train, self.Y_train = self.dataloader.dataloader(
            preprocess=True, pca=False, size=int(train_size), seed=seed
        )
        self.X_test, self.Y_test = self.dataloader.dataloader(
            preprocess=True, train=False, pca=False, size=int(test_size), seed=seed
        )

    def predict(self, un_processed_image, model_location):
        spike_train = np.array(self.dataloader.img2spiketrain(un_processed_image))
        synapses, neuron_label_map = self.load_model(model_location)
        return self.get_prediction(spike_train, synapses, neuron_label_map)
    
    def plot_random_samples(self, N=10, train=True, aggregate="sum", seed=None, cmap="hot_r", figsize=(10, 10)):
        """
        Plot N random MNIST samples from train or test set.

        Parameters
        ----------
        N : int
            Number of samples to plot.
        train : bool
            If True: use training set, else test set.
        aggregate : str
            "sum" or "mean" over time to reconstruct image from spike train.
        seed : int or None
            Optional seed for reproducibility.
            
            
        Note:
        -----
        nervos' dataloader directly returns spike trains for the MNIST images, 
        which we can visualize here before training the model. This also means,
        the MNIST images are not stored as pixel values in the model, but only 
        as spike trains. We therefore need to aggregate the spike trains over 
        time to reconstruct the original image for visualization.
        """

        if seed is not None:
            np.random.seed(seed)

        X = self.X_train if train else self.X_test
        Y = self.Y_train if train else self.Y_test

        indices = np.random.choice(len(X), size=min(N, len(X)), replace=False)

        cols = min(N, 5)
        rows = int(np.ceil(N / cols))

        plt.figure(figsize=figsize)

        for i, idx in enumerate(indices):
            spike_train = X[idx]  # shape: (784, T)

            if aggregate == "sum":
                img_vec = spike_train.sum(axis=1)
            elif aggregate == "mean":
                img_vec = spike_train.mean(axis=1)
            else:
                raise ValueError("aggregate must be 'sum' or 'mean'")

            img = img_vec.reshape(28, 28)

            plt.subplot(rows, cols, i + 1)
            plt.imshow(img, cmap=cmap, interpolation="nearest")
            plt.title(f"Label: {Y[idx]}")
            plt.axis("off")

        plt.tight_layout()
        
def visualize_synapse(synapses, labels, cmap="hot_r", figsize=(10, 30), ncols=5):
    kk = 28
    labels = np.asarray(labels)

    classes = {i: np.zeros((kk, kk)) for i in np.unique(labels)}
    for idx in range(len(synapses)):
        classes[labels[idx]] += synapses[idx].reshape((kk, kk))

    class_keys = sorted(classes.keys())
    n_classes = len(class_keys)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_classes / ncols))

    plt.figure(figsize=figsize)
    for i, k in enumerate(class_keys, start=1):
        plt.subplot(nrows, ncols, i)
        plt.imshow(classes[k], cmap=cmap, interpolation="nearest")
        plt.title(f"{k}")
        plt.axis("off")

    plt.tight_layout()
    
def accuracy(m2, classes, parameters_dict):
    loader = nv.dataloader.MNISTLoader(m2.parameters, classes=list(classes))
    spike_trains, labels = loader.dataloader(train=False, preprocess=True, 
                                             seed=123, size=parameters_dict["testing_images_amount"])

    t = 0
    c = 0
    preds = []
    print("Calculating Accuracy")
    for st, label in zip(spike_trains, labels):
        pred = m2.get_prediction(st)
        preds.append(pred)
        c += int(pred == label)
        t += 1
        print(f"\rTested {t} images", end="")
    print()
    print(c / t)
    return labels, preds

def confusion_matrix_np(y_true, y_pred, labels):
    """
    Calculate the confusion matrix.
    Rows: true labels
    Cols: predicted labels
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = np.asarray(labels, dtype=int)

    k = labels.size
    idx = {lab: i for i, lab in enumerate(labels)}
    C = np.zeros((k, k), dtype=int)

    for t, p in zip(y_true, y_pred):
        if (t in idx) and (p in idx):
            C[idx[t], idx[p]] += 1
    return C

def plot_confusion_matrix(C, labels, normalize=False, title=None, cmap="Greys"):
    """
    Plot confusion matrix. If normalize=True: row-normalize 
    (per true label).
    """
    C = np.asarray(C)
    if normalize:
        row_sums = C.sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0.0] = 1.0
        M = C / row_sums
    else:
        M = C

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, interpolation="nearest", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=[str(x) for x in labels],
        yticklabels=[str(x) for x in labels],
        xlabel="Predicted Labels",
        ylabel="True Labels",
        title=title if title is not None else ("Confusion matrix (normalized)" if normalize else "Confusion matrix"))

    # annotate:
    fmt = ".2f" if normalize else "d"
    thresh = M.max() * 0.6 if M.size else 0.0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(
                j, i, format(M[i, j], fmt),
                ha="center", va="center",
                color="white" if M[i, j] > thresh else "black")

    plt.tight_layout()
    return fig, ax

def accuracy_metrics(C):
    """
    From confusion matrix C (rows=true, cols=pred):
    accuracy and balanced accuracy.
    """
    C = np.asarray(C, dtype=float)
    total = C.sum()
    acc = np.trace(C) / total if total > 0 else np.nan

    row_sums = C.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.diag(C) / row_sums
    bal_acc = np.nanmean(recall)

    return acc, bal_acc, recall

def rasterplot_OLD(spike_train, title="raster", xlim=None):
    # spike_train shape: (N, T) with 0/1 entries
    N, T = spike_train.shape
    ys, xs = np.where(spike_train == 1)
    plt.figure(figsize=(10, 4))
    plt.scatter(xs, ys, s=2)
    if xlim is not None:
        plt.xlim(xlim)
    plt.xlabel("time step")
    plt.ylabel("neuron index")
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)
    plt.title(title)
    plt.tight_layout()

def rasterplot(
    spike_train: np.ndarray,
    title: str = "raster",
    xlim=None,
    highlight_neuron_idx: int | None = None,
    highlight_color: str = "orange",
    highlight2_neuron_idx: int | None = None,
    highlight2_color: str = "magenta",
    base_color: str = "0.35",
    s_base: float = 2.0,
    s_highlight: float = 8.0,
    s_highlight2: float = 8.0):
    """
    Raster plot for a binary spike matrix with up to two highlighted neurons.

    spike_train: shape (N, T), entries 0/1
    highlight_neuron_idx: epoch wise winner (orange)
    highlight2_neuron_idx: final winner (magenta, retroactive)
    If both indices are equal, only one overlay is drawn (orange), but the legend
    label indicates "epoch winner = final winner".
    """
    spike_train = np.asarray(spike_train)
    if spike_train.ndim != 2:
        raise ValueError(f"spike_train must be 2D (N,T), got shape {spike_train.shape}")

    N, T = spike_train.shape
    ys, xs = np.where(spike_train == 1)

    plt.figure(figsize=(10, 4))
    plt.scatter(xs, ys, s=s_base, color=base_color, linewidths=0)

    handles = []

    def _overlay(idx: int, color: str, size: float, label: str):
        if not (0 <= idx < N):
            raise ValueError(f"highlight idx={idx} out of bounds for N={N}")
        xs_h = np.where(spike_train[idx] == 1)[0]
        if xs_h.size == 0:
            return None
        ys_h = np.full(xs_h.shape, idx, dtype=int)
        return plt.scatter(xs_h, ys_h, s=size, color=color, linewidths=0, label=label)

    j1 = int(highlight_neuron_idx) if highlight_neuron_idx is not None else None
    j2 = int(highlight2_neuron_idx) if highlight2_neuron_idx is not None else None
    same = (j1 is not None) and (j2 is not None) and (j1 == j2)

    # epoch winner (always, if provided)
    if j1 is not None:
        label1 = f"epoch winner = final winner idx {j1}" if same else f"epoch winner idx {j1}"
        h1 = _overlay(j1, highlight_color, s_highlight, label1)
        if h1 is not None:
            handles.append(h1)

    # final winner (only if provided and different)
    if (j2 is not None) and (not same):
        h2 = _overlay(j2, highlight2_color, s_highlight2, f"final winner idx {j2}")
        if h2 is not None:
            handles.append(h2)

    if handles:
        plt.legend(loc="best")

    if xlim is not None:
        plt.xlim(xlim)

    plt.xlabel("time step")
    plt.ylabel("neuron index")
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)
    plt.title(title)
    plt.tight_layout()

def plot_rf_of_neuron(
    synapses_0: np.ndarray,
    neuron_idx: int,
    title: str = "",
    cmap: str = "viridis",
    figsize=(3.0, 3.0)) -> None:
    """
    Plot receptive field (weights reshaped to 28x28) of one output neuron.

    synapses_0: shape (n_out, 784)
    """
    w = np.asarray(synapses_0)[neuron_idx]  # (784,)
    img = w.reshape(28, 28)

    plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

def plot_label_template(
    synapses_0: np.ndarray,
    neuron_label_map: np.ndarray,
    label: int,
    title: str = "",
    cmap: str = "viridis",
    mode: str = "sum",  # "sum" or "mean"
    figsize=(3.0, 3.0)) -> None:
    """
    Plot label template: aggregate RFs of all neurons assigned to a given label.
    """
    synapses_0 = np.asarray(synapses_0)
    nlm = np.asarray(neuron_label_map)

    idx = np.where(nlm == int(label))[0]
    plt.figure(figsize=figsize)

    """ 
    idx.size indicates how many neurons are mapped to this label. 
    If idx.size == 0, it means no neuron is mapped to this label.
    """

    if idx.size == 0:
        plt.text(0.5, 0.5, f"No neurons mapped to label {label}", ha="center", va="center")
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        return

    W = synapses_0[idx]  # (n_label_neurons, 784)
    if mode == "sum":
        img = W.sum(axis=0).reshape(28, 28)
    elif mode == "mean":
        img = W.mean(axis=0).reshape(28, 28)
    else:
        raise ValueError("mode must be 'sum' or 'mean'")

    plt.imshow(img, cmap=cmap, interpolation="nearest")
    plt.title(title + f" (neurons mapping: {idx.size}/{synapses_0.shape[0]})")
    plt.axis("off")
    plt.tight_layout()


def get_winner_neuron_idx(m, epoch, train_image_idx):
    spk_out = np.asarray(m.spikeplots[epoch][train_image_idx][-1])  # (n_out, T)
    spike_counts = spk_out.sum(axis=1)
    return int(np.argmax(spike_counts)), spike_counts

def get_last_weight_snapshot_for_sample(m, epoch, train_image_idx):
    """
    weight_evolution[epoch][sample] is a list of snapshots.
    Each snapshot has shape (n_out, n_in).
    """
    snapshots = m.weight_evolution[epoch][train_image_idx]
    if snapshots is None or len(snapshots) == 0:
        raise ValueError("No weight evolution stored. Set m.get_weight_evolution=True before training.")
    return np.asarray(snapshots[-1])  # (n_out, n_in)

def plot_winner_rf_evolution_over_epochs(
    m,
    train_image_idx,
    last_epoch=None,
    cmap="viridis",
    parameters_dict=None,
    nlm_final=None,):
    """
    For each epoch, pick the winner neuron by spike count
    (same definition as in your raster loop), then plot its RF from the
    epoch specific weight snapshot and compute summary metrics for that same winner.

    Notes:
      - The "winner" can change across epochs (that is the point).
      - If nlm_final is given, we also show map=... in the titles (final neuron label map).
    """
    if last_epoch is None:
        last_epoch = m.parameters.epochs - 1

    true_label = int(m.Y_train[train_image_idx])

    rfs = []
    norms_l1 = []
    norms_l2 = []
    means = []

    winner_idxs = []
    winner_counts = []
    winner_maps = []

    for ep in range(m.parameters.epochs):
        # epoch specific winner from spikes (same as your raster loop)
        spk_out = np.asarray(m.spikeplots[ep][train_image_idx][-1])  # (n_out, T)
        spike_counts = spk_out.sum(axis=1)
        winner_idx = int(np.argmax(spike_counts))
        winner_count = int(spike_counts[winner_idx])

        winner_map = None
        if nlm_final is not None and winner_idx < len(nlm_final):
            winner_map = int(nlm_final[winner_idx])

        # epoch specific weights (last snapshot for this sample at this epoch)
        W = get_last_weight_snapshot_for_sample(m, ep, train_image_idx)  # (n_out, 784)
        w = W[winner_idx]  # (784,)

        if w.size != 28 * 28:
            raise ValueError(
                f"Expected 784 weights for RF, got {w.size}. W shape is {W.shape}.")

        rfs.append(w.reshape(28, 28))
        norms_l1.append(np.sum(np.abs(w)))
        norms_l2.append(np.sqrt(np.sum(w**2)))
        means.append(np.mean(w))

        winner_idxs.append(winner_idx)
        winner_counts.append(winner_count)
        winner_maps.append(winner_map)

    # RF tiles
    n = len(rfs)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)
    for ep in range(n):
        ax = axes[0, ep]
        ax.imshow(rfs[ep], cmap=cmap, interpolation="nearest")

        if winner_maps[ep] is None:
            ax.set_title(f"Epoch {ep}\nidx={winner_idxs[ep]}, spikes={winner_counts[ep]}")
        else:
            ax.set_title(
                f"Epoch {ep}\nidx={winner_idxs[ep]}, spikes={winner_counts[ep]}, map={winner_maps[ep]}"
            )

        ax.axis("off")

    plt.suptitle(
        f"Winner RF evolution for sample {train_image_idx}\ntrue={true_label}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f"winner_rf_evolution_sample{train_image_idx}_tiles.png"),dpi=200)
    plt.close()


    # summary metrics:
    fig, ax1 = plt.subplots(figsize=(6, 4))

    l1_line, = ax1.plot(norms_l1, marker="o", label="L1 norm")
    l2_line, = ax1.plot(norms_l2, marker="o", label="L2 norm")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("L1 / L2 value")
    ax1.set_title(f"Weight summary for epoch wise winners (sample {train_image_idx})")
    # annotate eg at the L2 dots, which winner idx they correspond to:
    for ep in range(n):
        # ax1.annotate(f"winner\nidx {winner_idxs[ep]}", (ep, norms_l2[ep]), 
        #              textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        ax1.annotate(f"winner\nidx: {winner_idxs[ep]}", (ep, np.max(norms_l1)*1.1), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=10,
                     va="top")
    ax1.set_ylim(bottom=0, top=np.max(norms_l1)*1.2)

    # secondary axis for mean weight:
    ax2 = ax1.twinx()
    mean_line, = ax2.plot(means, marker="o", linestyle="--", color="gray", label="Mean weight")
    ax2.set_ylabel("Mean weight", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    if parameters_dict is not None:
        ax2.set_ylim(bottom=parameters_dict["min_weight"], top=parameters_dict["max_weight"])
    else:
        ax2.set_ylim(bottom=0, top=1.01)

    # unified legend
    lines = [l1_line, l2_line, mean_line]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f"winner_rf_evolution_sample{train_image_idx}_summary.png"), dpi=200)
    plt.close()


def get_membrane_potential_trace(
    m,
    epoch: int,
    train_image_idx: int,
    neuron_idx: int,
    layer_idx: int = 0,
    duration: int | None = None,
    default_v: float | None = None):
    """
    Extract membrane potential trace V(t) for one neuron from nervos storage.

    m.layerpotentials[epoch][sample][layer][neuron] is expected to be a dict:
        {time_step: potential_value, ...}

    duration: number of time steps to extract (default: m.parameters.training_duration)
    default_v: value used if a time step is missing in the dict
               (default: m.parameters.inhibitory_potential if present, else -100)
    """
    if duration is None:
        duration = int(getattr(m.parameters, "training_duration", 100))

    if default_v is None:
        default_v = float(getattr(m.parameters, "inhibitory_potential", -100))

    pot_dict = m.layerpotentials[epoch][train_image_idx][layer_idx][neuron_idx]

    # nervos uses time steps starting at 1 in several places, keep that convention
    t = np.arange(1, duration + 1, dtype=int)
    v = np.array([pot_dict.get(int(tt), default_v) for tt in t], dtype=float)

    return t, v

def plot_winner_membrane_potential(
    m,
    epoch: int,
    train_image_idx: int,
    layer_idx: int = 0,
    duration: int | None = None,
    show_spikes: bool = True,
    parameters_dict: dict | None = None):
    """
    Plot V(t) for the winner neuron (by spike count) for a given epoch and training sample.
    """
    winner_idx, spike_counts = get_winner_neuron_idx(m, epoch, train_image_idx)

    # extract membrane potential
    t, v = get_membrane_potential_trace(
        m,
        epoch=epoch,
        train_image_idx=train_image_idx,
        neuron_idx=winner_idx,
        layer_idx=layer_idx,
        duration=duration)

    # optional spike times of the same neuron
    spike_times = None
    if show_spikes and spike_times is not None and spike_times.size > 0:
        # Ensure spike indices are valid for v
        valid = spike_times < len(v)
        st = spike_times[valid]
        # If your time axis t is 1..len(v), shift indices by +1 for plotting
        ax.scatter(st + 1, v[st], s=18)

    # plot
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, v, linewidth=1.5)

    if show_spikes and spike_times is not None and spike_times.size > 0:
        # spike_times are indices in [0..T-1]; your t starts at 1..duration
        # shift by +1 so axes match
        ax.scatter(spike_times + 1, v[(spike_times)], s=18)

    if parameters_dict is not None:
        # indicate spike threshold and resting potential as horizontal lines:
        if "spike_threshold" in parameters_dict:
            ax.axhline(parameters_dict["spike_threshold"], color="red", linestyle="--", label="Spike threshold")
        if "resting_potential" in parameters_dict:
            ax.axhline(parameters_dict["resting_potential"], color="blue", linestyle="--", label="Resting potential")
        ax.legend(loc="best")

    label = int(m.Y_train[train_image_idx])
    ax.set_title(
        f"Winner membrane potential, epoch {epoch}, sample {train_image_idx}\n(label {label}, winner neuron {winner_idx})")
    ax.set_xlabel("time step")
    ax.set_ylabel("membrane potential")
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()


# %% LOAD THE DATA
m = MNIST_SNN(p, identifier_name, classes=CLASSES)
m.initialise_layers([784,80])

m.plot_random_samples(N=25, train=True, aggregate="sum", seed=42, cmap="viridis", figsize=(8,9))
plt.suptitle("Random samples from the training set (aggregated over time (sum))")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "random_samples_train.png"), dpi=200)
plt.close()
m.plot_random_samples(N=25, train=False, aggregate="sum", cmap="viridis", figsize=(8,9))
plt.suptitle("Random samples from the test set (aggregated over time (sum))")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "random_samples_test.png"), dpi=200)
plt.close()
# %% RUN AND EVALUATE THE MODEL

# main code to run the model. This will train the model and save the learned synapses and 
# neuron label map in the storage directory (created if it doesn't exist). The model is 
# saved at the end of each epoch, with a subdirectory named "Epoch_{epoch_number}-{accuracy}" 
# for easy identification of the best epoch later on:
m.get_spikeplots = True
m.get_weight_evolution = True
y = m.train()

# evaluate the model by visualizing the learned synapses and calculating accuracy on test set:
visualize_synapse(m.learned_synapses[0], m.learned_neuron_label_map, figsize=(8, 5.0), cmap="viridis")
plt.suptitle("Learned synapses\n(summed over output neurons of the same predicted class)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "learned_synapses.png"), dpi=200)
plt.close()

# evaluate accuracy on test set:
y_true, y_pred = accuracy(m, classes=CLASSES, parameters_dict=parameters_dict)

# calculate confusion matrix and metrics:
labels = CLASSES  # use the selected classes
C = confusion_matrix_np(y_true, y_pred, labels=labels)
acc, bal_acc, recall = accuracy_metrics(C)
print("Accuracy:", acc)
print("Balanced accuracy:", bal_acc)
print("Recall:", recall)
# plot confusion matrix:
plot_confusion_matrix(C, labels=labels, normalize=False, title="Confusion matrix (counts)",cmap="Blues")
plt.savefig(os.path.join(RESULTS_PATH, "confusion_matrix_counts.png"), dpi=200)
plt.close()
plot_confusion_matrix(C, labels=labels, normalize=True, title="Confusion matrix (row-normalized)", cmap="Blues")
plt.savefig(os.path.join(RESULTS_PATH, "confusion_matrix_normalized.png"), dpi=200)
plt.close()

# spike rasterplots 
# m.spikeplots: [epoch][sample][layer] = spike_train array
#train_image_idx_list = [381, 481]
train_image_idx_list = [41, 61]
synapses_final = m.learned_synapses[0]      # shape (n_out, 784); these are the weights from input to output layer after training
                                            # we have 28x28=784 input neurons and 80 output neurons, so each row of synapses_final 
                                            # corresponds to the weights of one output neuron to all input neurons, which can be 
                                            # reshaped to a 28x28 image to visualize the receptive field of that output neuron.
                                            # thus, when plotting the receptive fields below, they are not built from the spike
                                            # activity of the output neurons, but from the synaptic weights that connect the input 
                                            # to the output layer. the spike counts just determine the "winner" neuron for which 
                                            # we plot the RF. Remember: the spikes of each neuron determine the weights of that 
                                            # neuron via STDP!
nlm_final      = m.learned_neuron_label_map # shape (n_out,)
for train_image_idx in train_image_idx_list:
    # train_image_idx = 381
    true_label = int(m.Y_train[train_image_idx])
    final_epoch = p.epochs - 1
    final_winner_idx, _ = get_winner_neuron_idx(m, final_epoch, train_image_idx)
    for epoch in range(p.epochs):
        # epoch = 1
        
        # pick winner from spikes (epoch-specific)
        spk_in  = m.spikeplots[epoch][train_image_idx][0]    # epoch, sample/train image, layer (0: Input, 1: Output)
        spk_out = m.spikeplots[epoch][train_image_idx][-1] 
        spike_counts = spk_out.sum(axis=1)
        winner_idx   = int(np.argmax(spike_counts))
        winner_count = int(spike_counts[winner_idx])
        
        # spike raster plot for the output layer at this epoch and this training image:
        # print(f"spk_out's shape: {spk_out.shape} (n_out, time points)")
        rasterplot(spk_out,
            title=(f"Output raster, epoch {epoch} sample {train_image_idx} "
                   f"(true={true_label}, current winner={winner_idx}, final winner={final_winner_idx})"),
            xlim=(0, spk_out.shape[1]),
            highlight_neuron_idx=winner_idx,
            highlight_color="orange",
            highlight2_neuron_idx=final_winner_idx,
            highlight2_color="magenta")
        plt.savefig(os.path.join(RESULTS_PATH, f"raster_output_neurons_epoch{epoch}_sample{train_image_idx}.png"), dpi=200)
        plt.close()
        

        # predicted label according to the (final) neuron_label_map:
        winner_label = int(nlm_final[winner_idx]) if winner_idx < len(nlm_final) else -1

        
        # epoch specific weights:
        W_ep = get_last_weight_snapshot_for_sample(m, epoch, train_image_idx)  # (n_out, 784)

        # 1. RF of the CURRENT epoch winner (this is what you want additionally):
        winner_label = int(nlm_final[winner_idx]) if winner_idx < len(nlm_final) else -1
        plot_rf_of_neuron(
            W_ep,
            winner_idx,
            title=(
                f"Epoch winner RF in epoch {epoch}\n on sample {train_image_idx}: neuron idx={winner_idx},\n"
                f"spikes={winner_count}, map={winner_label}, true={true_label}"),
            cmap="viridis",
            figsize=(3.8, 4.0))
        plt.savefig(os.path.join(RESULTS_PATH, f"rf_epochWinner_epoch{epoch}_sample{train_image_idx}.png"), dpi=200)
        plt.close()

        # 2. RF of the FINAL winner, but using CURRENT epoch weights (optional, if you also want this):
        final_label = int(nlm_final[final_winner_idx]) if final_winner_idx < len(nlm_final) else -1
        final_count_this_epoch = int(spike_counts[final_winner_idx])
        plot_rf_of_neuron(
            W_ep,
            final_winner_idx,
            title=(
                f"Final winner RF in epoch {epoch}\n on sample {train_image_idx}: neuron idx={final_winner_idx},\n"
                f"spikes(ep)={final_count_this_epoch}, map={final_label}, true={true_label}"),
            cmap="viridis",
            figsize=(3.8, 4.0))
        plt.savefig(os.path.join(RESULTS_PATH, f"rf_finalWinner_asOfEpoch{epoch}_sample{train_image_idx}.png"), dpi=200)
        plt.close()
        

        # label template:
        plot_label_template(
            synapses_final,
            nlm_final,
            true_label,
            title=f"Template sample {train_image_idx} in epoch {epoch}:\ntrue={true_label}",
            cmap="viridis",
            mode="mean",              # mean is usually less "messy" than sum
            figsize=(3.8, 4.0))
        plt.savefig(os.path.join(RESULTS_PATH, f"rf_template_epoch{epoch}_sample{train_image_idx}.png"), dpi=200)
        plt.close()
""" 
The raster plot shows the spiking activity of the output layer neurons over time for a single 
training image at a specific epoch. Here, we have 80 output neurons (as defined in the parameters) 
and the x-axis represents the discrete time steps of the simulation (100 + 1). Each dot in the raster 
plot corresponds to a spike from a particular neuron at a specific time step. The plots will show
an initial firing during the ongoing exposure to the input image, followed by a damping of activity 
due to adapatation or inhibition. The exact pattern of spiking will depend on the learned synaptic 
weights and the input. Later, it can happen, the we reach threshold and the output neurons fire again, 
which can be seen as a second wave of spiking in the raster plot. This behavior is controlled by

* refractory_time
* spike_drop_rate
* adaptive_threshold

The timing and pattern of these spikes are crucial for the model's predictions and learning process.

However, how nervos implements the SNN, this is no biologically detailed simulation with
biological plausible temporal dynamics as we have

* no synaptic delays
* no continuous integration of membrane potential over time (instead, the potential is updated in discrete time steps based on incoming spikes and current synaptic weights)
* no real membrane potential dynamics/ODE (like leaky integration, conductance-based synapses, etc.)
* no real WTA (winner-takes-all) inhibition between output neurons (instead, a simple global inhibition is applied to all output neurons when one fires)
"""


# let's plot the evolution of the synaptic weights over epochs for the winner neuron of the last epoch:
for train_image_idx in train_image_idx_list:
    plot_winner_rf_evolution_over_epochs(m, train_image_idx=train_image_idx, cmap="viridis",
                                         parameters_dict=parameters_dict, nlm_final=nlm_final)

    
# let's plot the potential of the winner neuron over time for the last epoch:
for train_image_idx in train_image_idx_list:
    plot_winner_membrane_potential(m, epoch=2, train_image_idx=train_image_idx, show_spikes=True)
    plt.savefig(os.path.join(RESULTS_PATH, f"winner_membrane_potential_epoch2_sample{train_image_idx}.png"), dpi=200)
    plt.close()
""" Interpretation of membrane potential traces:
In nervos, the membrane potentials stored in `m.layerpotentials` do not 
represent the raw pre-threshold voltage trajectory of a neuron. Instead, 
nervos logs the membrane state *after* the update step of each time step. 
In particular:

- If a neuron fires, its stored potential at that time step is the
  reset potential (e.g. -90), not the peak value that crossed threshold.
- If a neuron is inhibited, the stored value is the inhibitory potential
  (e.g. -100).
- The actual spike decision is based on the neuron’s adaptive threshold,
  not necessarily the fixed parameter `spike_threshold`.

Consequently, these traces should be interpreted as post-update state
trajectories (including reset and inhibition effects), rather than as
continuous voltage traces that visibly cross threshold.

Spike events are therefore more reliably visualized via the spike raster
(`m.spikeplots`), while the membrane potential traces illustrate the
internal state dynamics after each discrete update step.
"""



# %% END