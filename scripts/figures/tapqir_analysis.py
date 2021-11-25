from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle, Polygon
from tapqir.models import Cosmos

mpl.rc("text", usetex=True)
mpl.rcParams.update({"font.size": 8})

fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=False)

# panel a
path_data = "simulations/lamda0.5"
model = Cosmos()
model.load(path_data, data_only=False)

n = 0
frames = [100, 103, 105, 108, 110, 113, 115, 118, 120]
f1, f2 = 100, 121
vmin, vmax = model.data.vmin, model.data.vmax
gs = fig.add_gridspec(
    nrows=9,
    ncols=1,
    top=0.95,
    bottom=0.08,
    left=0.12,
    right=0.54,
    height_ratios=[1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
        ax.text(-40, -5, r"\textbf{A}")
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
    model.params["p(specific)"][n, f1:f2],
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

# height
for k in range(2):
    ax = fig.add_subplot(gs[2 + k * 3, :])
    mask = model.params["m_probs"][k, n, f1:f2] > 0.5
    ax.errorbar(
        x=torch.arange(f1, f2)[mask],
        y=model.params["height"]["Mean"][k, n, f1:f2][mask],
        yerr=torch.stack(
            (
                model.params["height"]["Mean"][k, n, f1:f2][mask]
                - model.params["height"]["LL"][k, n, f1:f2][mask],
                model.params["height"]["UL"][k, n, f1:f2][mask]
                - model.params["height"]["Mean"][k, n, f1:f2][mask],
            ),
            0,
        ),
        fmt="o",
        ms=2,
        color=f"C{k}",
        capsize=2,
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
    ax.set_yticks([0, 2000, 4000])
    ax.set_xticklabels([])
    ax.set_xlim(f1 - 0.5, f2 - 0.5)
    ax.set_ylim(-500, 5500)
    ax.set_ylabel(r"$h$")

    # x
    ax = fig.add_subplot(gs[3 + k * 3, :])
    mask = model.params["m_probs"][k, n, f1:f2] > 0.5
    ax.errorbar(
        x=torch.arange(f1, f2)[mask],
        y=model.params["x"]["Mean"][k, n, f1:f2][mask],
        yerr=torch.stack(
            (
                model.params["x"]["Mean"][k, n, f1:f2][mask]
                - model.params["x"]["LL"][k, n, f1:f2][mask],
                model.params["x"]["UL"][k, n, f1:f2][mask]
                - model.params["x"]["Mean"][k, n, f1:f2][mask],
            ),
            0,
        ),
        fmt="o",
        ms=2,
        color=f"C{k}",
        capsize=2,
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
    ax.set_yticks([-7, 0, 7])
    ax.set_xticklabels([])
    ax.set_xlim(f1 - 0.5, f2 - 0.5)
    ax.set_ylim(-9, 9)
    ax.set_ylabel(r"$x$")
    if k == 0:
        ax.text(
            f1 - 5,
            0,
            "spot 1",
            color="C0",
            rotation=90,
            va="center",
        )
    elif k == 1:
        ax.text(
            f1 - 5,
            0,
            "spot 2",
            color="C1",
            rotation=90,
            va="center",
        )
    ax.arrow(
        f1 - 4,
        -30,
        0,
        60,
        length_includes_head=True,
        head_width=0.0,
        head_length=0.0,
        color="k",
        clip_on=False,
    )
    ax.arrow(
        f1 - 4,
        30,
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
        -30,
        0.5,
        0,
        length_includes_head=True,
        head_width=0.0,
        head_length=0.0,
        color="k",
        clip_on=False,
    )

    # y
    ax = fig.add_subplot(gs[4 + k * 3, :])
    mask = model.params["m_probs"][k, n, f1:f2] > 0.5
    ax.errorbar(
        x=torch.arange(f1, f2)[mask],
        y=model.params["y"]["Mean"][k, n, f1:f2][mask],
        yerr=torch.stack(
            (
                model.params["y"]["Mean"][k, n, f1:f2][mask]
                - model.params["y"]["LL"][k, n, f1:f2][mask],
                model.params["y"]["UL"][k, n, f1:f2][mask]
                - model.params["y"]["Mean"][k, n, f1:f2][mask],
            ),
            0,
        ),
        fmt="o",
        ms=2,
        color=f"C{k}",
        capsize=2,
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
    ax.set_yticks([-7, 0, 7])
    ax.set_xticklabels([])
    ax.set_xlim(f1 - 0.5, f2 - 0.5)
    ax.set_ylim(-9, 9)
    ax.set_ylabel(r"$y$")

# background
ax = fig.add_subplot(gs[8, :])
ax.errorbar(
    x=torch.arange(f1, f2),
    y=model.params["background"]["Mean"][n, f1:f2],
    yerr=torch.stack(
        (
            model.params["background"]["Mean"][n, f1:f2]
            - model.params["background"]["LL"][n, f1:f2],
            model.params["background"]["UL"][n, f1:f2]
            - model.params["background"]["Mean"][n, f1:f2],
        ),
        0,
    ),
    fmt="o",
    ms=2,
    color="k",
    capsize=2,
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
ax.set_yticks([0, 200, 400])
ax.set_xlim(f1 - 0.5, f2 - 0.5)
ax.set_ylim(0, 500)
ax.set_ylabel(r"$b$")
ax.set_xlabel("Time (frame)")

# stripes
for i, f in enumerate(frames):
    x = 1.85
    h = (21 - 9 * x) / 8
    ax.add_patch(
        Polygon(
            [
                (f - 0.2, 0),
                (f + 0.2, 0),
                (f + 0.2, 9.48 * 500),
                (f1 - 0.5 + 0.5 * x + (h + x) * i + x / 2, 9.66 * 500),
                (f1 - 0.5 + 0.5 * x + (h + x) * i - x / 2, 9.66 * 500),
                (f - 0.2, 9.48 * 500),
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
    nrows=9,
    ncols=1,
    top=0.95,
    bottom=0.08,
    left=0.57,
    right=0.99,
    width_ratios=[9],
    height_ratios=[1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
    model.params["p(specific)"][n, f1:f2],
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

# height
for k in range(2):
    ax = fig.add_subplot(gs[2 + k * 3, :])
    mask = model.params["m_probs"][k, n, f1:f2] > 0.5
    ax.errorbar(
        x=torch.arange(f1, f2)[mask],
        y=model.params["height"]["Mean"][k, n, f1:f2][mask],
        yerr=torch.stack(
            (
                model.params["height"]["Mean"][k, n, f1:f2][mask]
                - model.params["height"]["LL"][k, n, f1:f2][mask],
                model.params["height"]["UL"][k, n, f1:f2][mask]
                - model.params["height"]["Mean"][k, n, f1:f2][mask],
            ),
            0,
        ),
        fmt="o",
        ms=2,
        color=f"C{k}",
        capsize=2,
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
    ax.set_yticks([0, 2000, 4000])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(f1 - 0.5, f2 - 0.5)
    ax.set_ylim(-500, 5500)

    # x
    ax = fig.add_subplot(gs[3 + k * 3, :])
    mask = model.params["m_probs"][k, n, f1:f2] > 0.5
    ax.errorbar(
        x=torch.arange(f1, f2)[mask],
        y=model.params["x"]["Mean"][k, n, f1:f2][mask],
        yerr=torch.stack(
            (
                model.params["x"]["Mean"][k, n, f1:f2][mask]
                - model.params["x"]["LL"][k, n, f1:f2][mask],
                model.params["x"]["UL"][k, n, f1:f2][mask]
                - model.params["x"]["Mean"][k, n, f1:f2][mask],
            ),
            0,
        ),
        fmt="o",
        ms=2,
        color=f"C{k}",
        capsize=2,
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
    ax.set_yticks([-7, 0, 7])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(f1 - 0.5, f2 - 0.5)
    ax.set_ylim(-9, 9)

    # y
    ax = fig.add_subplot(gs[4 + k * 3, :])
    mask = model.params["m_probs"][k, n, f1:f2] > 0.5
    ax.errorbar(
        x=torch.arange(f1, f2)[mask],
        y=model.params["y"]["Mean"][k, n, f1:f2][mask],
        yerr=torch.stack(
            (
                model.params["y"]["Mean"][k, n, f1:f2][mask]
                - model.params["y"]["LL"][k, n, f1:f2][mask],
                model.params["y"]["UL"][k, n, f1:f2][mask]
                - model.params["y"]["Mean"][k, n, f1:f2][mask],
            ),
            0,
        ),
        fmt="o",
        ms=2,
        color=f"C{k}",
        capsize=2,
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
    ax.set_yticks([-7, 0, 7])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(f1 - 0.5, f2 - 0.5)
    ax.set_ylim(-9, 9)

# background
ax = fig.add_subplot(gs[8, :])
ax.errorbar(
    x=torch.arange(f1, f2),
    y=model.params["background"]["Mean"][n, f1:f2],
    yerr=torch.stack(
        (
            model.params["background"]["Mean"][n, f1:f2]
            - model.params["background"]["LL"][n, f1:f2],
            model.params["background"]["UL"][n, f1:f2]
            - model.params["background"]["Mean"][n, f1:f2],
        ),
        0,
    ),
    fmt="o",
    ms=2,
    color="k",
    capsize=2,
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
ax.set_yticks([0, 200, 400])
ax.set_yticklabels([])
ax.set_xlim(f1 - 0.5, f2 - 0.5)
ax.set_ylim(0, 500)
ax.set_xlabel("Time (frame)")

# stripes
for i, f in enumerate(frames):
    x = 1.85
    h = (21 - 9 * x) / 8
    ax.add_patch(
        Polygon(
            [
                (f - 0.2, 0),
                (f + 0.2, 0),
                (f + 0.2, 9.48 * 500),
                (f1 - 0.5 + 0.5 * x + (h + x) * i + x / 2, 9.66 * 500),
                (f1 - 0.5 + 0.5 * x + (h + x) * i - x / 2, 9.66 * 500),
                (f - 0.2, 9.48 * 500),
            ],
            clip_on=False,
            zorder=100,
            alpha=0.25,
            color="gray",
        )
    )

plt.savefig("figures/tapqir_analysis.png", dpi=600)
