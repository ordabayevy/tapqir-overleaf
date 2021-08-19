from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import torch
from pyro import distributions as dist
from pyro.ops.stats import hpdi, resample
from tapqir.models import Cosmos
from tapqir.utils.imscroll import time_to_first_binding

mpl.rc("text", usetex=True)
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update({"font.size": 8})

# load model & parameters
path_data = Path("/shared/centaur/final/sigma54RNAPCy3-598P2993")
model = Cosmos(verbose=False)
model.load(path_data, data_only=False)

fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=False)
gs = fig.add_gridspec(
    nrows=2,
    ncols=2,
    top=0.95,
    bottom=0.1,
    left=0.08,
    right=0.98,
    hspace=0.4,
    wspace=0.3,
    width_ratios=[2, 1],
)

# panel a
ax = fig.add_subplot(gs[0, :])
ax.text(
    -30,
    -10,
    r"\textbf{a}",
)

# sorted on-target
ttfb = time_to_first_binding(model.params["z_map"])
# sort ttfb
sdx = torch.argsort(ttfb, descending=True)

norm = mpl.colors.Normalize(vmin=0, vmax=1)
im = ax.imshow(
    model.params["p(specific)"][sdx][:, ::10],
    norm=norm,
    aspect="equal",
    interpolation="none",
)
ax.set_xlabel("Time (frame)")
ax.set_ylabel("AOI")
cbar = fig.colorbar(im, ax=ax, aspect=8, shrink=0.9)
cbar.set_label(label=r"$p(\mathsf{specific})$")

gsb = gs[1, 0].subgridspec(1, 2, wspace=0.4)
# panel b (Tapqir)
ax = fig.add_subplot(gsb[0, 0])
ax.text(
    -0.38 * model.data.ontarget.F,
    1.1,
    r"\textbf{b}",
)
ax.text(
    model.data.ontarget.F,
    1.1,
    "Tapqir",
    horizontalalignment="right",
)

results = pd.read_csv("scripts/edfig5.csv", index_col=0)
# prepare data
Tmax = model.data.ontarget.F
torch.manual_seed(0)
z = dist.Bernoulli(model.params["p(specific)"]).sample((2000,))
data = time_to_first_binding(z)

nz = (data == 0).sum(1, keepdim=True)
n = (data == Tmax).sum(1, keepdim=True)
N = data.shape[1]

fraction_bound = (data.unsqueeze(-1) < torch.arange(Tmax)).float().mean(1)
fb_ll, fb_ul = hpdi(fraction_bound, 0.95, dim=0)

x = torch.arange(Tmax)

ax.fill_between(torch.arange(Tmax), fb_ll, fb_ul, alpha=0.3, color="C2")
ax.plot(torch.arange(Tmax), fraction_bound.mean(0), color="C2")

ax.plot(
    torch.arange(Tmax),
    (
        nz / N
        + (1 - nz / N)
        * (
            results.loc["Af", "Mean"]
            * (
                1
                - torch.exp(
                    -(results.loc["ka", "Mean"] + results.loc["kns", "Mean"])
                    * torch.arange(Tmax)
                )
            )
            + (1 - results.loc["Af", "Mean"])
            * (1 - torch.exp(-results.loc["kns", "Mean"] * torch.arange(Tmax)))
        )
    ).mean(0),
    color="k",
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
ax.set_xticks([0, 1500, 3000])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticklabels([r"$0$", r"$0.2$", r"$0.4$", r"$0.6$", r"$0.8$", r"$1$"])
ax.set_xlabel("Time (frame)")
ax.set_ylabel("Fraction bound")
ax.set_ylim(-0.05, 1.05)

# panel c (Spotpicker)
ax = fig.add_subplot(gsb[0, 1])
ax.text(
    -0.38 * model.data.ontarget.F,
    1.1,
    r"\textbf{c}",
)
ax.text(
    model.data.ontarget.F,
    1.1,
    "Spot-picker",
    horizontalalignment="right",
)

spotpicker_data = time_to_first_binding(model.data.ontarget.labels["z"])
spotpicker_data = torch.tensor(spotpicker_data, dtype=torch.float)
spotpicker_control = time_to_first_binding(model.data.offtarget.labels["z"])
spotpicker_control = torch.tensor(spotpicker_control, dtype=torch.float)

torch.manual_seed(0)
bootstrap_data = torch.stack(
    [
        resample(spotpicker_data, num_samples=len(spotpicker_data), replacement=True)
        for _ in range(2000)
    ],
    dim=0,
)
bootstrap_control = torch.stack(
    [
        resample(
            spotpicker_control, num_samples=len(spotpicker_control), replacement=True
        )
        for _ in range(2000)
    ],
    dim=0,
)

nz = (bootstrap_data == 0).sum(1, keepdim=True)
n = (bootstrap_data == Tmax).sum(1, keepdim=True)
N = bootstrap_data.shape[1]

nzc = (bootstrap_control == 0).sum(1, keepdim=True)
nc = (bootstrap_control == Tmax).sum(1, keepdim=True)
Nc = bootstrap_control.shape[1]

fraction_bound = (bootstrap_data.unsqueeze(-1) < torch.arange(Tmax)).float().mean(1)
fb_ll, fb_ul = hpdi(fraction_bound, 0.95, dim=0)

fraction_boundc = (bootstrap_control.unsqueeze(-1) < torch.arange(Tmax)).float().mean(1)
fbc_ll, fbc_ul = hpdi(fraction_boundc, 0.95, dim=0)

x = torch.arange(Tmax)

ax.fill_between(torch.arange(Tmax), fb_ll, fb_ul, alpha=0.3, color="#AA3377")
ax.plot(torch.arange(Tmax), fraction_bound.mean(0), color="#AA3377")

ax.plot(
    torch.arange(Tmax),
    (
        nz / N
        + (1 - nz / N)
        * (
            results.loc["Af_sp", "Mean"]
            * (
                1
                - torch.exp(
                    -(results.loc["ka_sp", "Mean"] + results.loc["kns_sp", "Mean"])
                    * torch.arange(Tmax)
                )
            )
            + (1 - results.loc["Af_sp", "Mean"])
            * (1 - torch.exp(-results.loc["kns_sp", "Mean"] * torch.arange(Tmax)))
        )
    ).mean(0),
    color="k",
)

ax.fill_between(torch.arange(Tmax), fbc_ll, fbc_ul, alpha=0.3, color="#CCBB44")
ax.plot(torch.arange(Tmax), fraction_boundc.mean(0), color="#CCBB44")

ax.plot(
    torch.arange(Tmax),
    (
        nzc / Nc
        + (1 - nzc / Nc)
        * (1 - torch.exp(-results.loc["kns_sp", "Mean"] * torch.arange(Tmax)))
    ).mean(0),
    color="k",
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
ax.set_xticks([0, 1500, 3000])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticklabels([r"$0$", r"$0.2$", r"$0.4$", r"$0.6$", r"$0.8$", r"$1$"])
ax.set_xlabel("Time (frame)")
ax.set_ylabel("Fraction bound")
ax.set_ylim(-0.05, 1.05)

# panel d
gsd = gs[1, 1].subgridspec(3, 1, hspace=1.5)
# ka
ax = fig.add_subplot(gsd[0])
ax.text(
    -0.38 * 0.006,
    2,
    r"\textbf{d}",
)
ax.barh(
    [0, 1],
    width=results.loc[["ka_sp", "ka"], "Mean"],
    height=0.6,
    tick_label=["Spot-picker", "Tapqir"],
    xerr=(
        results.loc[["ka_sp", "ka"], "Mean"] - results.loc[["ka_sp", "ka"], "95% LL"],
        results.loc[["ka_sp", "ka"], "95% UL"] - results.loc[["ka_sp", "ka"], "Mean"],
    ),
    color="gray",
)
plt.minorticks_on()
ax.tick_params(
    direction="in",
    which="minor",
    length=1,
    bottom=True,
    top=True,
    left=False,
    right=False,
)
ax.tick_params(
    direction="in",
    which="major",
    length=2,
    bottom=True,
    top=True,
    left=False,
    right=False,
)
ax.set_xticks([0, 2.5e-3, 5e-3])
ax.set_xticklabels([r"$0$", r"$2.5 \times 10^{-3}$", r"$5 \times 10^{-3}$"])
ax.set_xlabel(r"$k_\mathsf{a}$ (s$^{-1}$)")
ax.set_xlim(0, 6e-3)
ax.set_ylim(-0.6, 1.6)

# kns
ax = fig.add_subplot(gsd[1])
ax.barh(
    [0, 1],
    width=results.loc[["kns_sp", "kns"], "Mean"],
    height=0.6,
    tick_label=["Spot-picker", "Tapqir"],
    xerr=(
        results.loc[["kns_sp", "kns"], "Mean"]
        - results.loc[["kns_sp", "kns"], "95% LL"],
        results.loc[["kns_sp", "kns"], "95% UL"]
        - results.loc[["kns_sp", "kns"], "Mean"],
    ),
    color="gray",
)
plt.minorticks_on()
ax.tick_params(
    direction="in",
    which="minor",
    length=1,
    bottom=True,
    top=True,
    left=False,
    right=False,
)
ax.tick_params(
    direction="in",
    which="major",
    length=2,
    bottom=True,
    top=True,
    left=False,
    right=False,
)
ax.set_xticks([0, 2.5e-3, 5e-3])
ax.set_xticklabels([r"$0$", r"$2.5 \times 10^{-3}$", r"$5 \times 10^{-3}$"])
ax.set_xlabel(r"$k_\mathsf{ns}$ (s$^{-1}$)")
ax.set_xlim(0, 6e-3)
ax.set_ylim(-0.6, 1.6)

# Af
ax = fig.add_subplot(gsd[2])
ax.barh(
    [0, 1],
    width=results.loc[["Af_sp", "Af"], "Mean"],
    height=0.6,
    align="center",
    tick_label=["Spot-picker", "Tapqir"],
    xerr=(
        results.loc[["Af_sp", "Af"], "Mean"] - results.loc[["Af_sp", "Af"], "95% LL"],
        results.loc[["Af_sp", "Af"], "95% UL"] - results.loc[["Af_sp", "Af"], "Mean"],
    ),
    color="gray",
)
plt.minorticks_on()
ax.tick_params(
    direction="in",
    which="minor",
    length=1,
    bottom=True,
    top=True,
    left=False,
    right=False,
)
ax.tick_params(
    direction="in",
    which="major",
    length=2,
    bottom=True,
    top=True,
    left=False,
    right=False,
)
ax.set_xticks([0, 0.5, 1])
ax.set_xticklabels([r"$0$", r"$0.5$", r"$1$"])
ax.set_xlabel(r"$A_\mathsf{f}$")
ax.set_xlim(0, 1)
ax.set_ylim(-0.6, 1.6)

plt.savefig("extended-data/figure5.png", dpi=600)
