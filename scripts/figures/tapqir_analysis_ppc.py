"""
Figure 3-Figure supplement 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reproduction of experimental data by posterior predictive sampling.

Image file: ``figures/tapqir_analysis_ppc.png``

To generate the image file, run::

  python scripts/figures/tapqir_analysis_ppc.py

Input data:

* ``experimental/DatasetA`` (panel A)
* ``experimental/DatasetB`` (panel B)
* ``experimental/DatasetC`` (panel C)
* ``experimental/DatasetD`` (panel D)
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pyro
import torch
from pyro.infer import Predictive
from tapqir.models import Cosmos

mpl.rc("text", usetex=True)
mpl.rcParams.update({"font.size": 8})

fig = plt.figure(figsize=(7.2, 4.4), constrained_layout=False)
gs = fig.add_gridspec(
    nrows=2,
    ncols=2,
    left=0.1,
    top=0.9,
    bottom=0.1,
    right=0.98,
    hspace=0.6,
    wspace=0.3,
)

# panel a
path_data = Path("experimental/DatasetA")
model = Cosmos()
model.load(path_data, data_only=False)

model.load_checkpoint(param_only=True)
predictive = Predictive(
    pyro.poutine.uncondition(model.model), guide=model.guide, num_samples=1
)

aois = [51, 131, 51, 131, 163, 76]
frames = [1, 1, 559, 604, 638, 534]

model.n = torch.tensor(aois)
pyro.set_rng_seed(0)
samples = predictive()

gsa = gs[0, 0].subgridspec(
    3, 6, width_ratios=[1, 1, 1, 1, 1, 1], height_ratios=[1, 1, 1.2]
)
for i, n, f in zip(torch.arange(6), aois, frames):
    # experiment
    ax = fig.add_subplot(gsa[0, i])
    ax.imshow(
        model.data.images[n, f, model.cdx].numpy(),
        vmin=250,
        vmax=650,
        cmap="gray",
    )
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title(fr"${n}$" + "\n" + fr"${f}$", fontsize=8)
    if i == 0:
        ax.text(
            -25,
            -13,
            r"\textbf{A}",
        )
        ax.text(
            -13,
            -3,
            "AOI\nFrame",
            horizontalalignment="center",
        )
        ax.text(
            -13,
            8,
            "Experiment",
            horizontalalignment="center",
        )

    # prediction
    ax = fig.add_subplot(gsa[1, i])
    samples = predictive()
    img_sample = samples["data"][0, i, f]
    ax.imshow(img_sample.numpy(), vmin=250, vmax=650, cmap="gray")
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    if i == 0:
        ax.text(
            -13,
            10,
            "Posterior\nprediction",
            horizontalalignment="center",
            color="C0",
        )

    # histogram
    ax = fig.add_subplot(gsa[2, i])
    ax.hist(
        model.data.images[n, f, model.cdx].flatten().numpy(),
        range=(200, 700),
        density=True,
        bins=10,
        histtype="step",
        lw=0.5,
        color="k",
    )

    ax.hist(
        img_sample.flatten().numpy(),
        range=(200, 700),
        density=True,
        bins=20,
        histtype="step",
        lw=0.5,
        color="C0",
    )
    ax.set_ylim(0, 0.012)
    ax.set_xlim(200, 700)
    ax.set_xticks([300, 600])
    ax.set_yticks([0, 0.01])
    if i == 0:
        ax.set_yticklabels([r"$0$", r"$0.01$"])
        ax.set_ylabel("Density")
    elif i == 2:
        ax.set_yticklabels([])
        ax.set_xlabel("Intensity (a.u.)")
    else:
        ax.set_yticklabels([])

# panel b
path_data = Path("experimental/DatasetB")
model = Cosmos()
model.load(path_data, data_only=False)

model.load_checkpoint(param_only=True)
predictive = Predictive(
    pyro.poutine.uncondition(model.model), guide=model.guide, num_samples=1
)

aois = [2, 3, 2, 3, 2, 4]
frames = [2, 699, 912, 892, 980, 4099]

model.n = torch.tensor(aois)
pyro.set_rng_seed(0)
samples = predictive()

gsb = gs[0, 1].subgridspec(
    3, 6, width_ratios=[1, 1, 1, 1, 1, 1], height_ratios=[1, 1, 1.2]
)
for i, n, f in zip(torch.arange(6), aois, frames):
    # experiment
    ax = fig.add_subplot(gsb[0, i])
    ax.imshow(
        model.data.images[n, f, model.cdx].numpy(),
        vmin=1070,
        vmax=1200,
        cmap="gray",
    )
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title(fr"${n}$" + "\n" + fr"${f}$", fontsize=8)
    if i == 0:
        ax.text(
            -25,
            -13,
            r"\textbf{B}",
        )
        ax.text(
            -13,
            -3,
            "AOI\nFrame",
            horizontalalignment="center",
        )
        ax.text(
            -13,
            8,
            "Experiment",
            horizontalalignment="center",
        )

    # prediction
    ax = fig.add_subplot(gsb[1, i])
    samples = predictive()
    img_sample = samples["data"][0, i, f]
    ax.imshow(img_sample.numpy(), vmin=1070, vmax=1200, cmap="gray")
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    if i == 0:
        ax.text(
            -13,
            10,
            "Posterior\nprediction",
            horizontalalignment="center",
            color="C0",
        )

    # histogram
    ax = fig.add_subplot(gsb[2, i])
    ax.hist(
        model.data.images[n, f, model.cdx].flatten().numpy(),
        range=(1070, 1250),
        density=True,
        bins=10,
        histtype="step",
        lw=0.5,
        color="k",
    )

    ax.hist(
        img_sample.flatten().numpy(),
        range=(1070, 1250),
        density=True,
        bins=20,
        histtype="step",
        lw=0.5,
        color="C0",
    )
    ax.set_ylim(0, 0.05)
    ax.set_xlim(1070, 1250)
    ax.set_xticks([1100, 1200])
    ax.set_yticks([0, 0.02, 0.04])
    if i == 0:
        ax.set_xticklabels([r"$1.1$", r"$1.2$"])
        ax.set_yticklabels([r"$0$", r"$0.02$", r"$0.04$"])
        ax.set_ylabel("Density")
    elif i == 2:
        ax.set_xticklabels([r"$1.1$", r"$1.2$"])
        ax.set_yticklabels([])
        ax.set_xlabel("Intensity (a.u.)")
    elif i == 5:
        ax.set_yticklabels([])
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    else:
        ax.set_xticklabels([r"$1.1$", r"$1.2$"])
        ax.set_yticklabels([])

# panel c
path_data = Path("experimental/DatasetC")
model = Cosmos()
model.load(path_data, data_only=False)

model.load_checkpoint(param_only=True)
predictive = Predictive(
    pyro.poutine.uncondition(model.model), guide=model.guide, num_samples=1
)

aois = [1, 43, 1, 1, 1, 43]
frames = [2, 1140, 69, 2866, 2886, 1148]

model.n = torch.tensor(aois)
pyro.set_rng_seed(0)
samples = predictive()

gsc = gs[1, 0].subgridspec(
    3, 6, width_ratios=[1, 1, 1, 1, 1, 1], height_ratios=[1, 1, 1.2]
)
for i, n, f in zip(torch.arange(6), aois, frames):
    # experiment
    ax = fig.add_subplot(gsc[0, i])
    ax.imshow(
        model.data.images[n, f, model.cdx].numpy(),
        vmin=330,
        vmax=550,
        cmap="gray",
    )
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title(fr"${n}$" + "\n" + fr"${f}$", fontsize=8)
    if i == 0:
        ax.text(
            -25,
            -13,
            r"\textbf{C}",
        )
        ax.text(
            -13,
            -3,
            "AOI\nFrame",
            horizontalalignment="center",
        )
        ax.text(
            -13,
            8,
            "Experiment",
            horizontalalignment="center",
        )

    # prediction
    ax = fig.add_subplot(gsc[1, i])
    samples = predictive()
    img_sample = samples["data"][0, i, f]
    ax.imshow(img_sample.numpy(), vmin=330, vmax=550, cmap="gray")
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    if i == 0:
        ax.text(
            -13,
            10,
            "Posterior\nprediction",
            horizontalalignment="center",
            color="C0",
        )

    # histogram
    ax = fig.add_subplot(gsc[2, i])
    ax.hist(
        model.data.images[n, f, model.cdx].flatten().numpy(),
        range=(280, 680),
        density=True,
        bins=10,
        histtype="step",
        lw=0.5,
        color="k",
    )

    ax.hist(
        img_sample.flatten().numpy(),
        range=(280, 680),
        density=True,
        bins=20,
        histtype="step",
        lw=0.5,
        color="C0",
    )
    ax.set_ylim(0, 0.025)
    ax.set_xlim(280, 680)
    ax.set_xticks([300, 550])
    ax.set_yticks([0, 0.02])
    if i == 0:
        ax.set_yticklabels([r"$0$", r"$0.02$"])
        ax.set_ylabel("Density")
    elif i == 2:
        ax.set_yticklabels([])
        ax.set_xlabel("Intensity (a.u.)")
    else:
        ax.set_yticklabels([])

# panel d
path_data = Path("experimental/DatasetD")
model = Cosmos()
model.load(path_data, data_only=False)

model.load_checkpoint(param_only=True)
predictive = Predictive(
    pyro.poutine.uncondition(model.model), guide=model.guide, num_samples=1
)

aois = [26, 26, 32, 43, 26, 28]
frames = [2261, 2290, 4775, 4835, 2311, 54]

model.n = torch.tensor(aois)
pyro.set_rng_seed(0)
samples = predictive()

gsd = gs[1, 1].subgridspec(
    3, 6, width_ratios=[1, 1, 1, 1, 1, 1], height_ratios=[1, 1, 1.2]
)
for i, n, f in zip(torch.arange(6), aois, frames):
    # experiment
    ax = fig.add_subplot(gsd[0, i])
    ax.imshow(
        model.data.images[n, f, model.cdx].numpy(),
        vmin=340,
        vmax=550,
        cmap="gray",
    )
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title(fr"${n}$" + "\n" + fr"${f}$", fontsize=8)
    if i == 0:
        ax.text(
            -25,
            -13,
            r"\textbf{D}",
        )
        ax.text(
            -13,
            -3,
            "AOI\nFrame",
            horizontalalignment="center",
        )
        ax.text(
            -13,
            8,
            "Experiment",
            horizontalalignment="center",
        )

    # prediction
    ax = fig.add_subplot(gsd[1, i])
    samples = predictive()
    img_sample = samples["data"][0, i, f]
    ax.imshow(img_sample.numpy(), vmin=340, vmax=550, cmap="gray")
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    if i == 0:
        ax.text(
            -13,
            10,
            "Posterior\nprediction",
            horizontalalignment="center",
            color="C0",
        )

    # histogram
    ax = fig.add_subplot(gsd[2, i])
    ax.hist(
        model.data.images[n, f, model.cdx].flatten().numpy(),
        range=(300, 550),
        density=True,
        bins=10,
        histtype="step",
        lw=0.5,
        color="k",
    )

    ax.hist(
        img_sample.flatten().numpy(),
        range=(300, 550),
        density=True,
        bins=20,
        histtype="step",
        lw=0.5,
        color="C0",
    )
    ax.set_ylim(0, 0.03)
    ax.set_xlim(300, 550)
    ax.set_xticks([350, 500])
    ax.set_yticks([0, 0.02])
    if i == 0:
        ax.set_yticklabels([r"$0$", r"$0.02$"])
        ax.set_ylabel("Density")
    elif i == 2:
        ax.set_yticklabels([])
        ax.set_xlabel("Intensity (a.u.)")
    else:
        ax.set_yticklabels([])

plt.savefig("figures/tapqir_analysis_ppc.png", dpi=900)
