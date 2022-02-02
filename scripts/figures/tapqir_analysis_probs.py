"""
Figure 3-Figure supplement 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculated spot probabilities.

Image file: ``figures/tapqir_analysis_probs.png``

To generate the image file, run::

  python scripts/figures/tapqir_analysis_probs.py

Input data:

* ``simulations/lamda0.5`` (panel A)
* ``experimental/DatasetA`` (panel B)
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle, Polygon
from tapqir.models import Cosmos

mpl.rc("text", usetex=True)
mpl.rcParams.update({"font.size": 8})

fig = plt.figure(figsize=(7.2, 3.8), constrained_layout=False)

# panel a
path_data = "simulations/lamda0.5"
model = Cosmos()
model.load(path_data, data_only=False)

n = 0
frames = [100, 103, 105, 108, 110, 113, 115, 118, 120]
f1, f2 = 100, 121
vmin, vmax = model.data.vmin, model.data.vmax
gs = fig.add_gridspec(
    nrows=6,
    ncols=1,
    top=0.94,
    bottom=0.12,
    left=0.12,
    right=0.54,
    height_ratios=[1.5, 1.0, 1.0, 1.0, 1.0, 1.0],
)

gs1 = gs[0].subgridspec(2, 9)

for i, f in enumerate(frames):
    ax = fig.add_subplot(gs1[0, i])
    plt.imshow(
        model.data.images[n, f, model.cdx].numpy(), vmin=vmin, vmax=vmax, cmap="gray"
    )
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1.2)
        ax.spines[axis].set_color("#AA3377")
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    if i == 0:
        ax.text(-35, -5, r"\textbf{A}")
        ax.text(
            -15,
            11,
            "AOI\nimages",
            horizontalalignment="center",
        )

for i, f in enumerate(frames):
    ax = fig.add_subplot(gs1[1, i])
    ax.imshow(torch.ones((model.data.P, model.data.P)), vmin=0, vmax=1, cmap="gray")
    # add patch
    for k in range(2):
        if model.params["m_probs"][k, n, f].item() > 0.5:
            fill = model.params["theta_probs"][k, n, f].item() > 0.5
            ax.add_patch(
                Circle(
                    (
                        model.data.x[n, f, model.cdx]
                        + model.params["x"]["Mean"][k, n, f].item(),
                        model.data.y[n, f, model.cdx]
                        + model.params["y"]["Mean"][k, n, f].item(),
                    ),
                    1.5,
                    color=f"C{k}",
                    fill=fill,
                )
            )
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(1.2)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    if i == 0:
        ax.text(
            -15,
            11,
            "Spot\ndetection",
            horizontalalignment="center",
        )

# pspecific
ax = fig.add_subplot(gs[1, :])
ax.plot(
    torch.arange(f1, f2),
    model.params["z_probs"][n, f1:f2],
    "o-",
    ms=2,
    lw=0.5,
    color="C2",
)
plt.minorticks_on()
ax.tick_params(
    direction="in",
    which="minor",
    length=1,
    bottom=True,
    top=True,
    left=True,
    right=True,
)
ax.tick_params(
    direction="in",
    which="major",
    length=2,
    bottom=True,
    top=True,
    left=True,
    right=True,
)
ax.set_xticks(torch.arange(f1, f2 + 1, 5))
ax.set_yticks([0, 0.5, 1])
ax.set_xticklabels([])
ax.set_yticklabels([r"$0$", r"$0.5$", r"$1$"])
ax.set_xlim(f1 - 0.5, f2 - 0.5)
ax.set_ylim(-0.15, 1.15)
ax.set_ylabel(r"$p(\mathsf{specific})$")

for k in range(2):
    # p(theta)
    ax = fig.add_subplot(gs[2 + k * 2, :])
    ax.plot(
        torch.arange(f1, f2),
        model.params["theta_probs"][k, n, f1:f2],
        "o-",
        ms=2,
        lw=0.5,
        color=f"C{k}",
    )
    plt.minorticks_on()
    ax.tick_params(
        direction="in",
        which="minor",
        length=1,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax.tick_params(
        direction="in",
        which="major",
        length=2,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([r"$0$", r"$0.5$", r"$1$"])
    ax.set_xlim(f1 - 0.5, f2 - 0.5)
    ax.set_ylim(-0.15, 1.15)
    ax.set_ylabel(fr"$p(\theta={k+1})$")

    # p(m)
    ax = fig.add_subplot(gs[3 + k * 2, :])
    ax.plot(
        torch.arange(f1, f2),
        model.params["m_probs"][k, n, f1:f2],
        "o-",
        ms=2,
        lw=0.5,
        color=f"C{k}",
    )
    plt.minorticks_on()
    ax.tick_params(
        direction="in",
        which="minor",
        length=1,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax.tick_params(
        direction="in",
        which="major",
        length=2,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([r"$0$", r"$0.5$", r"$1$"])
    ax.set_xlim(f1 - 0.5, f2 - 0.5)
    ax.set_ylim(-0.15, 1.15)
    ax.set_ylabel(r"$p(m=1)$")
    if k == 1:
        ax.set_xticks(torch.arange(f1, f2 + 1, 5))
        ax.set_xlabel("Time (frame)")
    else:
        ax.set_xticklabels([])
    if k == 0:
        ax.text(
            f1 - 5,
            1,
            "spot 1",
            color="C0",
            rotation=90,
            va="center",
        )
    elif k == 1:
        ax.text(
            f1 - 5,
            1,
            "spot 2",
            color="C1",
            rotation=90,
            va="center",
        )
    ax.arrow(
        f1 - 4,
        -0.2,
        0,
        2.9,
        length_includes_head=True,
        head_width=0.0,
        head_length=0.0,
        color="k",
        clip_on=False,
    )
    ax.arrow(
        f1 - 4,
        2.7,
        0.5,
        0,
        length_includes_head=True,
        head_width=0.0,
        head_length=0.0,
        color="k",
        clip_on=False,
    )
    ax.arrow(
        f1 - 4,
        -0.2,
        0.5,
        0,
        length_includes_head=True,
        head_width=0.0,
        head_length=0.0,
        color="k",
        clip_on=False,
    )


# stripes
for i, f in enumerate(frames):
    x = 1.85
    h = (21 - 9 * x) / 8
    ax.add_patch(
        Polygon(
            [
                (f - 0.2, -0.15),
                (f + 0.2, -0.15),
                (f + 0.2, 5.73 * 1.3),
                (f1 - 0.5 + 0.5 * x + (h + x) * i + x / 2, 5.95 * 1.3),
                (f1 - 0.5 + 0.5 * x + (h + x) * i - x / 2, 5.95 * 1.3),
                (f - 0.2, 5.73 * 1.3),
            ],
            clip_on=False,
            zorder=100,
            alpha=0.25,
            color="gray",
        )
    )

# panel b
path_data = Path("experimental/DatasetA")
model = Cosmos()
model.load(path_data, data_only=False)

n = 163
f1, f2 = 625, 646
frames = [625, 628, 630, 633, 635, 638, 640, 643, 645]
vmin, vmax = 340, 635

gs = fig.add_gridspec(
    nrows=6,
    ncols=1,
    top=0.94,
    bottom=0.12,
    left=0.57,
    right=0.99,
    width_ratios=[9],
    height_ratios=[1.5, 1.0, 1.0, 1.0, 1.0, 1.0],
)
gs1 = gs[0].subgridspec(2, 9)

for i, f in enumerate(frames):
    ax = fig.add_subplot(gs1[0, i])
    plt.imshow(
        model.data.images[n, f, model.cdx].numpy(), vmin=vmin, vmax=vmax, cmap="gray"
    )
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1.2)
        ax.spines[axis].set_color("#AA3377")
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    if i == 0:
        ax.text(-5, -5, r"\textbf{B}")

for i, f in enumerate(frames):
    ax = fig.add_subplot(gs1[1, i])
    ax.imshow(torch.ones((model.data.P, model.data.P)), vmin=0, vmax=1, cmap="gray")
    # add patch
    for k in range(2):
        if model.params["m_probs"][k, n, f].item() > 0.5:
            fill = model.params["theta_probs"][k, n, f].item() > 0.5
            ax.add_patch(
                Circle(
                    (
                        model.data.x[n, f, model.cdx]
                        + model.params["x"]["Mean"][k, n, f].item(),
                        model.data.y[n, f, model.cdx]
                        + model.params["y"]["Mean"][k, n, f].item(),
                    ),
                    1.5,
                    color=f"C{k}",
                    fill=fill,
                )
            )
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(1.2)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

# pspecific
ax = fig.add_subplot(gs[1, :])
ax.plot(
    torch.arange(f1, f2),
    model.params["z_probs"][n, f1:f2],
    "o-",
    ms=2,
    lw=0.5,
    color="C2",
)
plt.minorticks_on()
ax.tick_params(
    direction="in",
    which="minor",
    length=1,
    bottom=True,
    top=True,
    left=True,
    right=True,
)
ax.tick_params(
    direction="in",
    which="major",
    length=2,
    bottom=True,
    top=True,
    left=True,
    right=True,
)
ax.set_xticks(torch.arange(f1, f2 + 1, 5))
ax.set_yticks([0, 0.5, 1])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlim(f1 - 0.5, f2 - 0.5)
ax.set_ylim(-0.15, 1.15)

for k in range(2):
    # p(theta)
    ax = fig.add_subplot(gs[2 + k * 2, :])
    ax.plot(
        torch.arange(f1, f2),
        model.params["theta_probs"][k, n, f1:f2],
        "o-",
        ms=2,
        lw=0.5,
        color=f"C{k}",
    )
    plt.minorticks_on()
    ax.tick_params(
        direction="in",
        which="minor",
        length=1,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax.tick_params(
        direction="in",
        which="major",
        length=2,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(f1 - 0.5, f2 - 0.5)
    ax.set_ylim(-0.15, 1.15)

    # p(m)
    ax = fig.add_subplot(gs[3 + k * 2, :])
    ax.plot(
        torch.arange(f1, f2),
        model.params["m_probs"][k, n, f1:f2],
        "o-",
        ms=2,
        lw=0.5,
        color=f"C{k}",
    )
    plt.minorticks_on()
    ax.tick_params(
        direction="in",
        which="minor",
        length=1,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax.tick_params(
        direction="in",
        which="major",
        length=2,
        bottom=True,
        top=True,
        left=True,
        right=True,
    )
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([])
    ax.set_xlim(f1 - 0.5, f2 - 0.5)
    ax.set_ylim(-0.15, 1.15)
    if k == 1:
        ax.set_xticks(torch.arange(f1, f2 + 1, 5))
        ax.set_xlabel("Time (frame)")
    else:
        ax.set_xticklabels([])

# stripes
for i, f in enumerate(frames):
    x = 1.85
    h = (21 - 9 * x) / 8
    ax.add_patch(
        Polygon(
            [
                (f - 0.2, -0.15),
                (f + 0.2, -0.15),
                (f + 0.2, 5.73 * 1.3),
                (f1 - 0.5 + 0.5 * x + (h + x) * i + x / 2, 5.95 * 1.3),
                (f1 - 0.5 + 0.5 * x + (h + x) * i - x / 2, 5.95 * 1.3),
                (f - 0.2, 5.73 * 1.3),
            ],
            clip_on=False,
            zorder=100,
            alpha=0.25,
            color="gray",
        )
    )

plt.savefig("figures/tapqir_analysis_probs.png", dpi=900)
