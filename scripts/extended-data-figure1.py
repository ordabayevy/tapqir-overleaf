from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tapqir.distributions import AffineBeta

mpl.rc("text", usetex=True)
mpl.rcParams.update({"font.size": 8})

fig = plt.figure(figsize=(7.2, 8), constrained_layout=False)
gs = fig.add_gridspec(
    nrows=1,
    ncols=1,
    top=0.98,
    bottom=0.25,
    left=0.02,
    right=0.98,
)

# panel a
ax = fig.add_subplot(gs[0])
ax.text(0, 1, r"\textbf{a}")
ax.axis("off")

gs = fig.add_gridspec(
    nrows=1,
    ncols=1,
    top=0.23,
    bottom=0.05,
    left=0.4,
    right=0.6,
)
# panel b
ax = fig.add_subplot(gs[0])
ax.text(-14, 1.2, r"\textbf{b}")

x = torch.arange(-7.5, 7.5, 0.1)
d1 = AffineBeta(0, 2, -7.5, 7.5)
d2 = AffineBeta(0, 230, -7.5, 7.5)
ax.plot(x, d2.log_prob(x).exp(), color="C2", label="specific")
ax.plot(x, d1.log_prob(x).exp(), color="C3", label="non-specific")
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
ax.set_xlabel(r"$x$ or $y$")
ax.set_ylabel("Probability density")
ax.set_xticks([-6, 0, 6])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xlim(-8, 8)
ax.set_ylim(-0.03, 1.2)
ax.legend(frameon=False)

plt.savefig("extended-data/figure1.svg", dpi=600)
