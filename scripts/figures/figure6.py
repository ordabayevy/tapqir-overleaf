from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import torch
from pyro import distributions as dist
from pyro.ops.stats import pi
from tapqir.models import Cosmos
from tapqir.utils.imscroll import association_rate, dissociation_rate

mpl.rc("text", usetex=True)
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams.update({"font.size": 8})

# path to simulated data
SIMULATIONS_DIR = Path("simulations")

truth = {}
fit = {}

for data_path in SIMULATIONS_DIR.iterdir():
    if data_path.is_dir() and data_path.name.startswith("kon"):
        # load results
        truth[data_path.name] = pd.read_csv(
            data_path / "simulated_params.csv", squeeze=True, index_col=0
        ).rename(data_path.name)

        model = Cosmos(verbose=False)
        model.load(data_path, data_only=False)

        fit[data_path.name] = model.statistics.drop("trained").astype(float)
        for p in ("gain", "proximity", "lamda", "SNR", "kon", "koff"):
            fit[data_path.name].loc[p, "True"] = truth[data_path.name][p]

        z_samples = dist.Bernoulli(model.params["p(specific)"]).sample((500,))
        # kon distribtion (MLE fit)
        kon_samples = association_rate(z_samples)
        fit[data_path.name].loc["kon", "Mean"] = kon_samples.mean().item()
        hdp = pi(kon_samples, 0.95, dim=0)
        fit[data_path.name].loc["kon", "95% LL"] = hdp[0].item()
        fit[data_path.name].loc["kon", "95% UL"] = hdp[1].item()
        # koff distribution (MLE fit)
        koff_samples = dissociation_rate(z_samples)
        hdp = pi(koff_samples, 0.95, dim=0)
        fit[data_path.name].loc["koff", "Mean"] = koff_samples.mean().item()
        fit[data_path.name].loc["koff", "95% LL"] = hdp[0].item()
        fit[data_path.name].loc["koff", "95% UL"] = hdp[1].item()

truth_df = pd.concat(truth.values(), axis=1).T.astype(float)
truth_df["Keq"] = truth_df["kon"] / truth_df["koff"]
truth_df = truth_df.sort_values(by=["kon", "lamda"])

fig = plt.figure(figsize=(7.2, 3.4), constrained_layout=False)
gs = fig.add_gridspec(
    nrows=6,
    ncols=1,
    top=0.98,
    bottom=0.1,
    left=0.15,
    right=0.5,
    hspace=0.1,
    height_ratios=[1, 1, 1, 1, 0.3, 1],
)

# panel a

model = Cosmos(verbose=False)
model.load(SIMULATIONS_DIR / "kon0.02lamda1", data_only=False)
torch.manual_seed(0)
z_samples = dist.Bernoulli(model.params["p(specific)"]).sample((2,))
n = 4
f1 = 0
f2 = 300

# target-binder reaction
ax = fig.add_subplot(gs[0])
ax.text(-0.37, 0.9, r"\textbf{a}")
ax.text(
    0.27,
    0.5,
    "target + binder",
    horizontalalignment="center",
)
ax.text(
    0.7,
    0.5,
    "target·binder",
    horizontalalignment="center",
)
ax.text(
    0.5,
    0.7,
    r"$k_\mathsf{on}$",
    horizontalalignment="center",
)
ax.text(
    0.5,
    0.28,
    r"$k_\mathsf{off}$",
    horizontalalignment="center",
)
ax.arrow(
    0.45,
    0.6,
    0.1,
    0,
    length_includes_head=True,
    head_width=0.05,
    head_length=0.02,
    color="k",
)
ax.arrow(
    0.55,
    0.5,
    -0.1,
    0,
    length_includes_head=True,
    head_width=0.05,
    head_length=0.02,
    color="k",
)
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# panel b

# simulation
ax = fig.add_subplot(gs[1])
ax.text(-0.45 * 300, 1.1, r"\textbf{b}")
ax.plot(
    torch.arange(f1, f2),
    model.data.ontarget.labels["z"][n, f1:f2],
    "-",
    lw=1,
    color="C0",
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
    left=True,
    right=True,
)
ax.set_xticks([0, 100, 200])
ax.set_yticks([0, 1])
ax.set_xticklabels([])
ax.set_xlim(f1 - 10, f2 + 10)
ax.set_ylim(-0.15, 1.15)
ax.text(
    -90,
    0.2,
    "simulated\ntarget-specific\nspot presence",
    horizontalalignment="center",
)

# pspecific
ax = fig.add_subplot(gs[2])
ax.plot(
    torch.arange(f1, f2),
    model.params["p(specific)"][n, f1:f2],
    "o-",
    ms=1,
    lw=1,
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
ax.set_xticks([0, 100, 200])
ax.set_yticks([0, 0.5, 1])
ax.set_xticklabels([])
ax.set_yticklabels([r"$0$", r"$0.5$", r"$1$"])
ax.set_xlim(f1 - 10, f2 + 10)
ax.set_ylim(-0.15, 1.15)
ax.text(
    -90,
    0.55,
    "Tapqir",
    horizontalalignment="center",
)
ax.text(
    -90,
    0.25,
    r"$p(\mathsf{specific})$",
    horizontalalignment="center",
)

# sample 1
ax = fig.add_subplot(gs[3])
ax.plot(torch.arange(f1, f2), z_samples[0, n, f1:f2], "-", lw=1, color="k")
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
    left=True,
    right=True,
)
ax.set_xticks([0, 100, 200])
ax.set_yticks([0, 1])
ax.set_xticklabels([])
ax.set_xlim(f1 - 10, f2 + 10)
ax.set_ylim(-0.15, 1.15)

# dots
ax = fig.add_subplot(gs[4])
ax.text(
    0.5,
    0,
    r"\textbf{...}",
    rotation=90,
    va="center",
)
ax.set_xlim(0, 1)
ax.set_ylim(-0.5, 0.5)
ax.axis("off")

# bracket
ax.arrow(
    -0.25,
    4.75,
    0,
    -3.0,
    length_includes_head=True,
    head_width=0.01,
    head_length=0.2,
    color="k",
    clip_on=False,
)
ax.arrow(
    -0.1,
    -4.25,
    0,
    8.5,
    length_includes_head=True,
    head_width=0.0,
    head_length=0.0,
    color="k",
    clip_on=False,
)
ax.arrow(
    -0.1,
    -4.25,
    0.02,
    0,
    length_includes_head=True,
    head_width=0.0,
    head_length=0.0,
    color="k",
    clip_on=False,
)
ax.arrow(
    -0.1,
    4.25,
    0.02,
    0,
    length_includes_head=True,
    head_width=0.0,
    head_length=0.0,
    color="k",
    clip_on=False,
)

# sample 2
ax = fig.add_subplot(gs[5])
ax.plot(torch.arange(f1, f2), z_samples[1, n, f1:f2], "-", lw=1, color="k")
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
    left=True,
    right=True,
)
ax.set_xticks([0, 100, 200])
ax.set_yticks([0, 1])
ax.set_xlabel("Time (frame)")
ax.set_xlim(f1 - 10, f2 + 10)
ax.set_ylim(-0.15, 1.15)
# dton and dtoff
ax.arrow(
    50,
    0.5,
    57,
    0,
    length_includes_head=True,
    width=0.01,
    head_width=0.08,
    head_length=4,
    color="k",
)
ax.arrow(
    50,
    0.5,
    -23,
    0,
    length_includes_head=True,
    width=0.01,
    head_width=0.08,
    head_length=4,
    color="k",
)
ax.text(
    70,
    0.65,
    r"$\Delta t_\mathsf{on}$",
    horizontalalignment="center",
)
ax.arrow(
    178,
    0.5,
    7,
    0,
    length_includes_head=True,
    width=0.01,
    head_width=0.08,
    head_length=4,
    color="k",
)
ax.arrow(
    207,
    0.5,
    -7,
    0,
    length_includes_head=True,
    width=0.01,
    head_width=0.08,
    head_length=4,
    color="k",
)
ax.text(
    220,
    0.65,
    r"$\Delta t_\mathsf{off}$",
    horizontalalignment="center",
)
ax.text(
    -90,
    1,
    "posterior\nsamples\nfrom",
    horizontalalignment="center",
)
ax.text(
    -90,
    0.7,
    r"$p(\mathsf{specific})$",
    horizontalalignment="center",
)


# panel c
gs = fig.add_gridspec(
    nrows=3,
    ncols=3,
    top=0.95,
    bottom=0.1,
    left=0.58,
    right=0.99,
    wspace=0.05,
    hspace=0.1,
)

for i, kon in enumerate([0.01, 0.02, 0.03]):
    ax = fig.add_subplot(gs[0, i])
    mask = truth_df["kon"] == kon
    ax.scatter(
        truth_df.loc[mask, "lamda"],
        truth_df.loc[mask, "kon"],
        s=15,
        marker="x",
        label="simulation",
    )
    ax.errorbar(
        truth_df.loc[mask, "lamda"],
        [
            fit[i].loc["kon", "Mean"]
            for i in truth_df.index
            if i.startswith(f"kon{kon}")
        ],
        yerr=torch.tensor(
            [
                abs(
                    fit[i].loc["kon", ["95% LL", "95% UL"]].values
                    - fit[i].loc["kon", "Mean"]
                )
                for i in truth_df.index
                if i.startswith(f"kon{kon}")
            ]
        ).T,
        fmt="o",
        ms=2,
        color="k",
        capsize=2,
        label="inferred from Tapqir analysis",
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

    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.02, 0.04])
    ax.set_xticklabels([])
    if i == 0:
        ax.set_yticklabels([r"$0$", r"$0.02$", r"$0.04$"])
        ax.set_ylabel(r"$k_\mathsf{on}$ (s$^{-1}$)")
        ax.text(-0.7, 1.05 * 0.05, r"\textbf{c}")
        ax.legend(bbox_to_anchor=(0, 1.25), loc="upper left", frameon=False, ncol=2)
    else:
        ax.set_yticklabels([])
    ax.set_ylim(-0.002, 0.05)
    ax.set_xlim(-0.1, 1.1)

# panel d
for i, kon in enumerate([0.01, 0.02, 0.03]):
    ax = fig.add_subplot(gs[1, i])
    mask = truth_df["kon"] == kon
    ax.scatter(
        truth_df.loc[mask, "lamda"], truth_df.loc[mask, "koff"], s=15, marker="x"
    )
    ax.errorbar(
        truth_df.loc[mask, "lamda"],
        [
            fit[i].loc["koff", "Mean"]
            for i in truth_df.index
            if i.startswith(f"kon{kon}")
        ],
        yerr=torch.tensor(
            [
                abs(
                    fit[i].loc["koff", ["95% LL", "95% UL"]].values
                    - fit[i].loc["koff", "Mean"]
                )
                for i in truth_df.index
                if i.startswith(f"kon{kon}")
            ]
        ).T,
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

    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.2, 0.4])
    ax.set_xticklabels([])
    if i == 0:
        ax.set_yticklabels([r"$0$", r"$0.2$", r"$0.4$"])
        ax.set_ylabel(r"$k_\mathsf{off}$ (s$^{-1}$)")
        ax.text(-0.7, 1.05 * 0.5, r"\textbf{d}")
    else:
        ax.set_yticklabels([])
    ax.set_ylim(-0.02, 0.5)
    ax.set_xlim(-0.1, 1.1)

# panel e
for i, kon in enumerate([0.01, 0.02, 0.03]):
    ax = fig.add_subplot(gs[2, i])
    mask = truth_df["kon"] == kon
    ax.scatter(truth_df.loc[mask, "lamda"], truth_df.loc[mask, "Keq"], s=15, marker="x")
    ax.errorbar(
        truth_df.loc[mask, "lamda"],
        [
            fit[i].loc["Keq", "Mean"]
            for i in truth_df.index
            if i.startswith(f"kon{kon}")
        ],
        yerr=torch.tensor(
            [
                abs(
                    fit[i].loc["Keq", ["95% LL", "95% UL"]].values
                    - fit[i].loc["Keq", "Mean"]
                )
                for i in truth_df.index
                if i.startswith(f"kon{kon}")
            ]
        ).T,
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
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.08, 0.16])
    ax.set_xticklabels([r"$0$", r"$0.5$", r"$1$"])
    if i == 0:
        ax.set_yticklabels([r"$0$", r"$0.08$", r"$0.16$"])
        ax.set_ylabel(r"$K_\mathsf{eq}$")
        ax.text(-0.7, 1.05 * 0.2, r"\textbf{e}")
    else:
        ax.set_yticklabels([])
    if i == 1:
        ax.set_xlabel(r"$\lambda$")
    ax.set_ylim(-0.01, 0.2)
    ax.set_xlim(-0.1, 1.1)

plt.savefig("figures/figure6.png", dpi=600)
