from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tapqir.models import Cosmos
from tapqir.utils.dataset import load

mpl.rc("text", usetex=True)
mpl.rcParams.update({"font.size": 8})
# mpl.rcParams["savefig.facecolor"] = "0.8"

# path to simulated data
SIMULATIONS_DIR = Path("simulations")

truth = {}
fit = {}
predictions = defaultdict(dict)

# load results
for data_path in SIMULATIONS_DIR.iterdir():
    if data_path.is_dir() and data_path.name.startswith("height"):

        truth[data_path.name] = pd.read_csv(
            data_path / "simulated_params.csv", squeeze=True, index_col=0
        ).rename(data_path.name)

        model = Cosmos(verbose=False)
        model.load(data_path, data_only=False)

        fit[data_path.name] = model.statistics.drop("trained").astype(float)
        for p in ("gain", "proximity", "pi", "lamda", "SNR"):
            fit[data_path.name].loc[p, "True"] = truth[data_path.name][p]

        mask = torch.from_numpy(model.data.ontarget.labels["z"])
        samples = torch.masked_select(model.params["p(specific)"], mask)
        predictions[data_path.name]["z_masked"] = samples
        predictions[data_path.name]["z_all"] = model.params["p(specific)"].flatten()

truth_df = pd.concat(truth.values(), axis=1).T.astype(float)
truth_df = truth_df.sort_values(by="height")

fig = plt.figure(figsize=(7.2, 5.9), constrained_layout=False)
gsa = fig.add_gridspec(
    nrows=1,
    ncols=8,
    top=0.94,
    bottom=0.87,
    left=0.2,
    right=0.8,
    wspace=0.03,
)

# panel a
for i, name in enumerate(truth_df.index):
    ax = fig.add_subplot(gsa[i])
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title(fr"${truth_df.loc[name, 'SNR']:.2f}$", fontsize=8)
    data = load(SIMULATIONS_DIR / name)
    ax.imshow(data.ontarget.images[3, 222].numpy(), vmin=190, vmax=380, cmap="gray")
    if i == 0:
        ax.text(
            -12,
            -3.5,
            "SNR:",
        )
        ax.text(
            -20,
            -5.5,
            r"\textbf{a}",
        )

gs = fig.add_gridspec(
    nrows=3,
    ncols=3,
    top=0.83,
    bottom=0.08,
    left=0.08,
    right=0.98,
    wspace=0.35,
    hspace=0.47,
    width_ratios=[1, 3, 1],
    height_ratios=[1, 1, 1],
)

# panel b
ax = fig.add_subplot(gs[0, 0])
ax.text(
    -2,
    1.1,
    r"\textbf{b}",
)
ax.errorbar(
    truth_df["SNR"],
    [fit[i].loc["p(specific)", "Mean"] for i in truth_df.index],
    yerr=torch.tensor(
        [
            abs(
                fit[i].loc["p(specific)", ["95% LL", "95% UL"]].values
                - fit[i].loc["p(specific)", "Mean"]
            )
            for i in truth_df.index
        ]
    ).T,
    fmt="o-",
    ms=3,
    color="k",
    mfc="C2",
    mec="C2",
    ecolor="C2",
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
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels([r"$0$", r"$0.5$", r"$1$"])
ax.set_xlim(-0.1, 4.1)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("SNR")
ax.set_ylabel(r"$\langle p(\mathsf{specific}) \rangle $")

# panel c
gsc = gs[0, 1].subgridspec(1, 3, wspace=0.03)
for i, name in enumerate(["height300", "height750", "height3000"]):
    ax = fig.add_subplot(gsc[0, i])
    ax.hist(
        predictions[name]["z_all"].numpy(),
        bins=torch.arange(0, 1.05, 0.05),
        histtype="bar",
        lw=1.0,
        color="#dddddd",
        edgecolor="k",
        label=f"{truth_df.loc[name, 'SNR']:.2f} SNR",
    )
    ax.hist(
        predictions[name]["z_masked"].numpy(),
        bins=torch.arange(0, 1.05, 0.05),
        histtype="bar",
        lw=1.0,
        color="C2",
        edgecolor="k",
        alpha=0.6,
        label=f"{truth_df.loc[name, 'SNR']:.2f} SNR",
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
    ax.set_yscale("log")
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([r"$0$", r"$0.5$", r"$1$"])
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1e-1, 5e3)
    if name == "height300":
        ax.set_ylabel("Counts")
        ax.text(
            -0.6,
            7e3,
            r"\textbf{c}",
        )
        ax.text(
            0.2,
            1e3,
            r"$\mathsf{SNR} = 0.38$",
        )
    elif name == "height750":
        ax.set_xlabel(r"$p(\mathsf{specific})$")
        ax.set_yticklabels([])
        ax.text(
            0.2,
            1e3,
            r"$\mathsf{SNR} = 0.94$",
        )
    else:
        ax.set_yticklabels([])
        ax.text(
            0.2,
            1e3,
            r"$\mathsf{SNR} = 3.76$",
        )

# panel d
ax = fig.add_subplot(gs[0, 2])
ax.text(
    -2,
    1.1,
    r"\textbf{d}",
)
ax.plot(
    truth_df["SNR"],
    [fit[i].loc["Recall", "Mean"] for i in truth_df.index],
    "o-",
    color="k",
    ms=3,
    label="Recall",
)
ax.plot(
    truth_df["SNR"][1:],
    [fit[i].loc["Precision", "Mean"] for i in truth_df.index][1:],
    "o-",
    color="C3",
    ms=3,
    label="Precision",
)
ax.plot(
    truth_df["SNR"],
    [fit[i].loc["MCC", "Mean"] for i in truth_df.index],
    "o-",
    color="C0",
    ms=3,
    label="MCC",
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
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels([r"$0$", r"$0.5$", r"$1$"])
ax.set_xlim(-0.1, 4.1)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("SNR")
ax.set_ylabel("Accuracy")
ax.legend(bbox_to_anchor=(1.08, 0.0), loc="lower right", frameon=False, ncol=1)

# load results
truth = {}
fit = {}
predictions = defaultdict(dict)
for data_path in SIMULATIONS_DIR.iterdir():
    if data_path.is_dir() and data_path.name.startswith("lamda"):

        truth[data_path.name] = pd.read_csv(
            data_path / "simulated_params.csv", squeeze=True, index_col=0
        ).rename(data_path.name)

        model = Cosmos(verbose=False)
        model.load(data_path, data_only=False)

        fit[data_path.name] = model.statistics.drop("trained").astype(float)
        for p in ("gain", "proximity", "pi", "lamda", "SNR"):
            fit[data_path.name].loc[p, "True"] = truth[data_path.name][p]

        mask = torch.from_numpy(model.data.ontarget.labels["z"])
        samples = torch.masked_select(model.params["p(specific)"], mask)
        predictions[data_path.name]["z_masked"] = samples
        predictions[data_path.name]["z_all"] = model.params["p(specific)"].flatten()

truth_df = pd.concat(truth.values(), axis=1).T.astype(float)
truth_df = truth_df.sort_values(by="lamda")

# panel e
ax = fig.add_subplot(gs[1, 0])
ax.text(
    -0.5,
    1.1,
    r"\textbf{e}",
)
ax.errorbar(
    truth_df["lamda"],
    [fit[i].loc["p(specific)", "Mean"] for i in truth_df.index],
    yerr=torch.tensor(
        [
            abs(
                fit[i].loc["p(specific)", ["95% LL", "95% UL"]].values
                - fit[i].loc["p(specific)", "Mean"]
            )
            for i in truth_df.index
        ]
    ).T,
    fmt="o-",
    ms=3,
    color="k",
    mfc="C2",
    mec="C2",
    ecolor="C2",
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
ax.set_yticks([0, 0.5, 1])
ax.set_xticklabels([r"$0$", r"$0.5$", r"$1$"])
ax.set_yticklabels([r"$0$", r"$0.5$", r"$1$"])
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\langle p(\mathsf{specific}) \rangle $")

# panel f
gsg = gs[1, 1].subgridspec(1, 3, wspace=0.03)
for i, name in enumerate(["lamda0.01", "lamda0.15", "lamda1"]):
    ax = fig.add_subplot(gsg[0, i])
    ax.hist(
        predictions[name]["z_all"].numpy(),
        bins=torch.arange(0, 1.05, 0.05),
        histtype="bar",
        lw=1.0,
        color="#dddddd",
        edgecolor="k",
        label=f"{truth_df.loc[name, 'SNR']:.2f} SNR",
    )
    ax.hist(
        predictions[name]["z_masked"].numpy(),
        bins=torch.arange(0, 1.05, 0.05),
        histtype="bar",
        lw=1.0,
        color="C2",
        edgecolor="k",
        alpha=0.6,
        label=f"{truth_df.loc[name, 'SNR']:.2f} SNR",
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
    ax.set_yscale("log")
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([r"$0$", r"$0.5$", r"$1$"])
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1e-1, 5e3)
    if name == "lamda0.01":
        ax.set_ylabel("Counts")
        ax.text(
            -0.6,
            7e3,
            r"\textbf{f}",
        )
        ax.text(
            0.3,
            1e3,
            r"$\lambda = 0.01$",
        )
    elif name == "lamda0.15":
        ax.set_xlabel(r"$p(\mathsf{specific})$")
        ax.set_yticklabels([])
        ax.text(
            0.3,
            1e3,
            r"$\lambda = 0.15$",
        )
    else:
        ax.set_yticklabels([])
        ax.text(
            0.35,
            1e3,
            r"$\lambda = 1$",
        )

# panel g
ax = fig.add_subplot(gs[1, 2])
ax.text(
    -0.5,
    1.1,
    r"\textbf{g}",
)
ax.plot(
    truth_df["lamda"],
    [fit[i].loc["Recall", "Mean"] for i in truth_df.index],
    "o-",
    color="k",
    ms=3,
    label="Recall",
)
ax.plot(
    truth_df["lamda"],
    [fit[i].loc["Precision", "Mean"] for i in truth_df.index],
    "o-",
    color="C3",
    ms=3,
    label="Precision",
)
ax.plot(
    truth_df["lamda"],
    [fit[i].loc["MCC", "Mean"] for i in truth_df.index],
    "o-",
    color="C0",
    ms=3,
    label="MCC",
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
ax.set_yticks([0, 0.5, 1])
ax.set_xticklabels([r"$0$", r"$0.5$", r"$1$"])
ax.set_yticklabels([r"$0$", r"$0.5$", r"$1$"])
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\langle p(\mathsf{specific}) \rangle $")
ax.set_ylabel("Accuracy")
ax.legend(bbox_to_anchor=(1.08, 0.0), loc="lower right", frameon=False, ncol=1)

# load results
truth = {}
fit = {}
predictions = defaultdict(dict)
for data_path in SIMULATIONS_DIR.iterdir():
    if data_path.is_dir() and data_path.name.startswith("negative"):

        truth[data_path.name] = pd.read_csv(
            data_path / "simulated_params.csv", squeeze=True, index_col=0
        ).rename(data_path.name)

        model = Cosmos(verbose=False)
        model.load(data_path, data_only=False)

        fit[data_path.name] = model.statistics.drop("trained").astype(float)
        for p in ("gain", "proximity", "pi", "lamda", "SNR"):
            fit[data_path.name].loc[p, "True"] = truth[data_path.name][p]

        mask = torch.from_numpy(model.data.ontarget.labels["z"])
        samples = torch.masked_select(model.params["p(specific)"], mask)
        predictions[data_path.name]["z_masked"] = samples
        predictions[data_path.name]["z_all"] = model.params["p(specific)"].flatten()

truth_df = pd.concat(truth.values(), axis=1).T.astype(float)
truth_df = truth_df.sort_values(by="lamda")

# panel h
gsi = gs[2, 1].subgridspec(1, 3, wspace=0.03)
for i, name in enumerate(["negative0.01", "negative0.15", "negative1"]):
    ax = fig.add_subplot(gsi[0, i])
    ax.hist(
        predictions[name]["z_all"].numpy(),
        bins=torch.arange(0, 1.05, 0.05),
        histtype="bar",
        lw=1.0,
        color="#dddddd",
        edgecolor="k",
        label=f"{truth_df.loc[name, 'SNR']:.2f} SNR",
    )
    ax.hist(
        predictions[name]["z_masked"].numpy(),
        bins=torch.arange(0, 1.05, 0.05),
        histtype="bar",
        lw=1.0,
        color="C2",
        edgecolor="k",
        alpha=0.6,
        label=f"{truth_df.loc[name, 'SNR']:.2f} SNR",
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
    ax.set_yscale("log")
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([r"$0$", r"$0.5$", r"$1$"])
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1e-1, 5e3)
    if name == "negative0.01":
        ax.set_ylabel("Counts")
        ax.text(
            -0.6,
            7e3,
            r"\textbf{h}",
        )
        ax.text(
            0.3,
            1e3,
            r"$\lambda = 0.01$",
        )
    elif name == "negative0.15":
        ax.set_xlabel(r"$p(\mathsf{specific})$")
        ax.set_yticklabels([])
        ax.text(
            0.3,
            1e3,
            r"$\lambda = 0.15$",
        )
    else:
        ax.set_yticklabels([])
        ax.text(
            0.35,
            1e3,
            r"$\lambda = 1$",
        )
    ax.text(
        0.35,
        3e2,
        r"$\pi = 0$",
    )

plt.savefig("figures/figure5.png", dpi=600)
