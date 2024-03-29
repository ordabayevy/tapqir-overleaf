"""
Figure 2
--------

Depiction of the cosmos probabilistic image model and model parameters.

Image file: ``figures/graphical_model.png``

To generate panels A, B, and C in the image, run (outpus ``figures/graphical_model.svg`` vector image)::

  python scripts/figures/graphical_model.py

Input data:

* ``experimental/DatasetA``

Graphical model in panel D is located at ``figures/graphical_model.pdf``.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from tapqir.distributions.util import gaussian_spots
from tapqir.models import Cosmos

mpl.rc("text", usetex=True)
mpl.rcParams.update({"font.size": 8})

fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=False)

gs = fig.add_gridspec(
    nrows=3,
    ncols=2,
    top=0.99,
    bottom=0.03,
    left=0.04,
    right=0.98,
    hspace=0.1,
    width_ratios=[2, 3],
    height_ratios=[1, 2, 0.6],
)

# panel a
gsa = gs[0, 0].subgridspec(1, 3, width_ratios=[2, 1, 3])

path_data = Path("experimental/DatasetA")
model = Cosmos()
model.load(path_data, data_only=False)

# 2-D image
ax = fig.add_subplot(gsa[0, 0])
ax.text(-8, -8, r"\textbf{A}")
n = 163
f = 640
vmin, vmax = 340, 635
ax.imshow(model.data.images[n, f, model.cdx].numpy(), vmin=vmin, vmax=vmax, cmap="gray")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-3, 16)
ax.set_ylim(16, -3)
ax.set_xlabel(r"$x$", labelpad=-6)
ax.set_ylabel(r"$y$", labelpad=-5)
ax.set_title("2-D image", fontsize=8, y=1, pad=-2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
# make arrows
ax.arrow(
    -2.0,
    9.0,
    0.0,
    4,
    length_includes_head=True,
    head_width=0.4,
    head_length=0.6,
    color="k",
)
ax.arrow(
    9.0,
    15.0,
    4.0,
    0,
    length_includes_head=True,
    head_width=0.4,
    head_length=0.6,
    color="k",
)

# <-> arrow
ax = fig.add_subplot(gsa[0, 1])
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.arrow(
    0.0,
    0.0,
    0.5,
    0,
    length_includes_head=True,
    width=0.005,
    head_width=0.03,
    head_length=0.15,
    color="k",
    clip_on=False,
)
ax.arrow(
    0.0,
    0.0,
    -0.5,
    0,
    length_includes_head=True,
    width=0.005,
    head_width=0.03,
    head_length=0.15,
    color="k",
    clip_on=False,
)
ax.axis("off")

# 3-D plot
vmin3d = model.params["background"]["Mean"][n, f] + model.data.offset.mean
vmax3d = vmin3d + 150
ax = fig.add_subplot(gsa[0, 2], projection="3d")
P_range = torch.arange(model.data.P)
X, Y = torch.meshgrid(P_range, P_range, indexing="xy")
Z = model.data.images[n, f, model.cdx].numpy()
ax.invert_yaxis()
# Plot the surface.
surf = ax.plot_surface(
    X,
    Y,
    Z,
    cmap=mpl.cm.coolwarm,
    vmin=vmin3d,
    vmax=vmax3d,
    linewidth=0,
    antialiased=False,
)

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
ax.set_xlabel(r"$x$", labelpad=-15)
ax.set_ylabel(r"$y$", labelpad=-15)

ax.view_init(elev=15.0, azim=125)
ax.set_zlim(vmin3d - 150, vmax3d + 100)
ax.set_title("3-D plot", y=1, pad=2, fontsize=8)

# panel b
gsb = gs[1, 0].subgridspec(2, 2, hspace=0.0)
vmin3d = model.params["background"]["Mean"][n, f]
vmax3d = vmin3d + 120
for m1, m2 in ((0, 0), (0, 1), (1, 0), (1, 1)):
    ax = fig.add_subplot(gsb[m1, m2], projection="3d")

    img_ideal = model.params["background"]["Mean"][n, f : f + 1, None, None]
    gaussian = gaussian_spots(
        model.params["height"]["Mean"][:, n, f : f + 1],
        model.params["width"]["Mean"][:, n, f : f + 1],
        model.params["x"]["Mean"][:, n, f : f + 1],
        model.params["y"]["Mean"][:, n, f : f + 1],
        model.data.xy[n, f : f + 1, model.cdx],
        model.data.P,
        torch.tensor([[m1], [m2]]),
    )
    img_ideal = img_ideal + gaussian.sum(-4)
    Z = img_ideal[0].numpy()

    # Plot the surface.
    ax.invert_yaxis()
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=mpl.cm.coolwarm,
        vmin=vmin3d,
        vmax=vmax3d,
        linewidth=0,
        antialiased=False,
    )
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.view_init(elev=15.0, azim=125)
    ax.set_zlim(vmin3d - 150, vmax3d + 100)

    if m1 == 0 and m2 == 0:
        ax.set_title(
            r"$m_{\mathsf{spot}(1)}=0$" + "\n" + r"$m_{\mathsf{spot}(2)}=0$",
            y=1,
            pad=2,
            fontsize=8,
        )
        ax.text2D(-0.125, 0.12, s=r"\textbf{B}")
    elif m1 == 0 and m2 == 1:
        ax.set_title(
            r"$m_{\mathsf{spot}(1)}=0$" + "\n" + r"$m_{\mathsf{spot}(2)}=1$",
            y=1,
            pad=2,
            fontsize=8,
        )
        ax.text2D(-0.05, 0.03, s=r"$\mathbf{2}$", color="C1")
    elif m1 == 1 and m2 == 0:
        ax.set_title(
            r"$m_{\mathsf{spot}(1)}=1$" + "\n" + r"$m_{\mathsf{spot}(2)}=0$",
            y=1,
            pad=2,
            fontsize=8,
        )
        ax.text2D(0.01, 0.035, s=r"$\mathbf{1}$", color="C0")
    elif m1 == 1 and m2 == 1:
        ax.set_title(
            r"$m_{\mathsf{spot}(1)}=1$" + "\n" + r"$m_{\mathsf{spot}(2)}=1$",
            y=1,
            pad=2,
            fontsize=8,
        )
        ax.text2D(0.01, 0.035, s=r"$\mathbf{1}$", color="C0")
        ax.text2D(-0.05, 0.03, s=r"$\mathbf{2}$", color="C1")

# panel c
gsc = gs[2, 0].subgridspec(1, 3)

ax = fig.add_subplot(gsc[0, 0])
theta0 = (
    gaussian_spots(
        height=torch.tensor([3000, 3000]),
        width=torch.tensor([1.5, 1.5]),
        x=torch.tensor([-4.5, 3.5]),
        y=torch.tensor([2.5, -1.5]),
        target_locs=torch.tensor([6.5, 6.5]),
        P=14,
    ).sum(-3)
    + 10
)
ax.imshow(theta0.numpy(), vmin=0, vmax=200, cmap="gray")
ax.axis("off")
ax.set_title(r"$z = 0, \: \theta = 0$", fontsize=8)
ax.text(-4, -3, r"\textbf{C}")
ax.text(-4.5 + 6, 2.5 + 7, s=r"$\mathbf{1}$", color="C0")
ax.text(3.5 + 6, -1.5 + 7, s=r"$\mathbf{2}$", color="C1")

ax = fig.add_subplot(gsc[0, 1])
theta1 = (
    gaussian_spots(
        height=torch.tensor([3000, 3000]),
        width=torch.tensor([1.5, 1.5]),
        x=torch.tensor([0, 3.5]),
        y=torch.tensor([0, -3.5]),
        target_locs=torch.tensor([6.5, 6.5]),
        P=14,
    ).sum(-3)
    + 10
)
ax.imshow(theta1.numpy(), vmin=0, vmax=200, cmap="gray")
ax.axis("off")
ax.set_title(r"$z = 1, \: \theta = 1$", fontsize=8)
ax.text(0 + 6, 0 + 7, s=r"$\mathbf{1}$", color="C0")
ax.text(3.5 + 6, -3.5 + 7, s=r"$\mathbf{2}$", color="C1")

ax = fig.add_subplot(gsc[0, 2])
theta2 = (
    gaussian_spots(
        height=torch.tensor([3000, 3000]),
        width=torch.tensor([1.5, 1.5]),
        x=torch.tensor([0.5, 0]),
        y=torch.tensor([5.5, 0]),
        target_locs=torch.tensor([6.5, 6.5]),
        P=14,
    ).sum(-3)
    + 10
)
ax.imshow(theta2.numpy(), vmin=0, vmax=200, cmap="gray")
ax.axis("off")
ax.set_title(r"$z = 1, \: \theta = 2$", fontsize=8)
ax.text(0.5 + 6, 5.5 + 7, s=r"$\mathbf{1}$", color="C0")
ax.text(0 + 6, 0 + 7, s=r"$\mathbf{2}$", color="C1")

# panel d
ax = fig.add_subplot(gs[:, 1])
ax.axis("off")
ax.text(-0.1, 0.95, r"\textbf{D}")

plt.savefig("figures/graphical_model.svg", dpi=900)
