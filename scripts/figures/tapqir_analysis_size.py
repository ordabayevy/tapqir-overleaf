"""
Figure 3-Figure supplement 4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Effect of AOI size on analysis of experimental data.

Image file: ``figures/tapqir_analysis_size.png``

To generate the image file, run::

  python scripts/figures/tapqir_analysis_size.py

Input data:

* ``experimental/DatasetA`` (14x14 AOIs)
* ``experimental/P10DatasetA`` (10x10 AOIs)
* ``experimental/P6DatasetA`` (6x6 AOIs)
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from tapqir.models import Cosmos

mpl.rc("text", usetex=True)
mpl.rcParams.update({"font.size": 8})

fig = plt.figure(figsize=(7.2, 4.2), constrained_layout=False)

gsa = fig.add_gridspec(
    nrows=4,
    ncols=1,
    top=0.95,
    bottom=0.6,
    left=0.08,
    right=0.99,
    height_ratios=[1, 0.6, 0.4, 1.5],
)

gsb = fig.add_gridspec(
    nrows=4,
    ncols=1,
    top=0.45,
    bottom=0.1,
    left=0.08,
    right=0.99,
    height_ratios=[1, 0.6, 0.4, 1.5],
)

# P = 14
path_data = Path("experimental/DatasetA")
model = Cosmos()
model.load(path_data, data_only=False)

n = 163
f1, f2 = 625, 646
frames = range(f1, f2)
vmin, vmax = 340, 635
gs1 = gsa[0, 0].subgridspec(1, 21)
for i, f in enumerate(frames):
    ax = fig.add_subplot(gs1[0, i])
    plt.imshow(
        model.data.images[n, f, model.cdx].numpy(), vmin=vmin, vmax=vmax, cmap="gray"
    )
    ax.set_title(rf"${f}$", fontsize=8)
    ax.axis("off")

    if i == 0:
        ax.text(-25, -7, r"\textbf{A}")
        ax.text(-23, 8, r"$P = 14$", color="C2")

# pspecific
axa = fig.add_subplot(gsa[3, 0])
axa.plot(
    torch.arange(f1, f2),
    model.params["z_probs"][n, f1:f2],
    "o-",
    ms=2,
    lw=0.5,
    color="C2",
)
plt.minorticks_on()
axa.tick_params(
    direction="in",
    which="minor",
    length=1,
    bottom=True,
    top=True,
    left=True,
    right=True,
)
axa.tick_params(
    direction="in",
    which="major",
    length=2,
    bottom=True,
    top=True,
    left=True,
    right=True,
)
axa.set_xticks(torch.arange(f1, f2 + 1, 5))
axa.set_yticks([0, 0.5, 1])
axa.set_xlim(f1 - 0.5, f2 - 0.5)
axa.set_ylim(-0.15, 1.15)
axa.set_xlabel("Time (frame)")
axa.set_ylabel(r"$p(\mathsf{specific})$")

n = 0
f1, f2 = 220, 241
frames = range(f1, f2)
vmin, vmax = 250, 600
gs1 = gsb[0, 0].subgridspec(1, 21)
for i, f in enumerate(frames):
    ax = fig.add_subplot(gs1[0, i])
    plt.imshow(
        model.data.images[n, f, model.cdx].numpy(), vmin=vmin, vmax=vmax, cmap="gray"
    )
    ax.set_title(rf"${f}$", fontsize=8)
    ax.axis("off")

    if i == 0:
        ax.text(-25, -7, r"\textbf{B}")
        ax.text(-23, 8, r"$P = 14$", color="C2")

# pspecific
axb = fig.add_subplot(gsb[3, 0])
axb.plot(
    torch.arange(f1, f2),
    model.params["z_probs"][n, f1:f2],
    "o-",
    ms=2,
    lw=0.5,
    color="C2",
)
plt.minorticks_on()
axb.tick_params(
    direction="in",
    which="minor",
    length=1,
    bottom=True,
    top=True,
    left=True,
    right=True,
)
axb.tick_params(
    direction="in",
    which="major",
    length=2,
    bottom=True,
    top=True,
    left=True,
    right=True,
)
axb.set_xticks(torch.arange(f1, f2 + 1, 5))
axb.set_yticks([0, 0.5, 1])
axb.set_xlim(f1 - 0.5, f2 - 0.5)
axb.set_ylim(-0.15, 1.15)
axb.set_xlabel("Time (frame)")
axb.set_ylabel(r"$p(\mathsf{specific})$")

# P = 10
path_data = Path("experimental/P10DatasetA")
model = Cosmos()
model.load(path_data, data_only=False)

n = 163
f1, f2 = 625, 646
frames = range(f1, f2)
vmin, vmax = 340, 635
gs1 = gsa[1, 0].subgridspec(1, 21)
for i, f in enumerate(frames):
    ax = fig.add_subplot(gs1[0, i])
    plt.imshow(
        model.data.images[n, f, model.cdx].numpy(), vmin=vmin, vmax=vmax, cmap="gray"
    )
    ax.axis("off")
    if i == 0:
        ax.text(-20, 5, r"$P = 10$", color="C3")

# pspecific
axa.plot(
    torch.arange(f1, f2),
    model.params["z_probs"][n, f1:f2],
    "o-",
    ms=2,
    lw=0.5,
    color="C3",
)

n = 0
f1, f2 = 220, 241
frames = range(f1, f2)
vmin, vmax = 250, 600
gs1 = gsb[1, 0].subgridspec(1, 21)
for i, f in enumerate(frames):
    ax = fig.add_subplot(gs1[0, i])
    plt.imshow(
        model.data.images[n, f, model.cdx].numpy(), vmin=vmin, vmax=vmax, cmap="gray"
    )
    ax.axis("off")
    if i == 0:
        ax.text(-20, 5, r"$P = 10$", color="C3")

# pspecific
axb.plot(
    torch.arange(f1, f2),
    model.params["z_probs"][n, f1:f2],
    "o-",
    ms=2,
    lw=0.5,
    color="C3",
)

path_data = Path("experimental/P6DatasetA")
model = Cosmos()
model.load(path_data, data_only=False)

n = 163
f1, f2 = 625, 646
frames = range(f1, f2)
vmin, vmax = 340, 635
gs1 = gsa[2, 0].subgridspec(1, 21)
for i, f in enumerate(frames):
    ax = fig.add_subplot(gs1[0, i])
    plt.imshow(
        model.data.images[n, f, model.cdx].numpy(), vmin=vmin, vmax=vmax, cmap="gray"
    )
    ax.axis("off")
    if i == 0:
        ax.text(-19, 3, r"$P = 6$", color="C4")

# pspecific
axa.plot(
    torch.arange(f1, f2),
    model.params["z_probs"][n, f1:f2],
    "o-",
    ms=2,
    lw=0.5,
    color="C4",
)

n = 0
f1, f2 = 220, 241
frames = range(f1, f2)
vmin, vmax = 250, 600
gs1 = gsb[2, 0].subgridspec(1, 21)
for i, f in enumerate(frames):
    ax = fig.add_subplot(gs1[0, i])
    plt.imshow(
        model.data.images[n, f, model.cdx].numpy(), vmin=vmin, vmax=vmax, cmap="gray"
    )
    ax.axis("off")
    if i == 0:
        ax.text(-19, 3, r"$P = 6$", color="C4")

# pspecific
axb.plot(
    torch.arange(f1, f2),
    model.params["z_probs"][n, f1:f2],
    "o-",
    ms=2,
    lw=0.5,
    color="C4",
)

plt.savefig("figures/tapqir_analysis_size.png", dpi=900)
