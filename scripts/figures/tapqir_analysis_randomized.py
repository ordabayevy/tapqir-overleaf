from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rc("text", usetex=True)
mpl.rcParams.update({"font.size": 8})

fig = plt.figure(figsize=(7.2, 1.9), constrained_layout=False)
gs = fig.add_gridspec(
    nrows=1,
    ncols=4,
    top=0.9,
    bottom=0.25,
    left=0.06,
    right=0.98,
    wspace=0.4,
)

# path to simulated data
SIMULATIONS_DIR = Path("simulations")

truth = {}
fit = {}

for data_path in SIMULATIONS_DIR.iterdir():
    if data_path.is_dir() and data_path.name.startswith("seed"):
        # load results
        truth[data_path.name] = pd.read_csv(
            data_path / "simulated_params.csv", squeeze=True, index_col=0
        ).rename(data_path.name)
        truth[data_path.name]["seed"] = data_path.name[4:]

        statistics = pd.read_csv(data_path / "cosmos-channel0-summary.csv", index_col=0)

        fit[data_path.name] = statistics.astype(float)
        for p in ("gain", "proximity", "pi", "lamda", "SNR"):
            fit[data_path.name].loc[p, "True"] = truth[data_path.name][p]

truth_df = pd.concat(truth.values(), axis=1).T.astype(float).sort_values(by="seed")
truth_df = truth_df.drop(columns=["seed"])

# panel a
ax = fig.add_subplot(gs[0])
ax.text(-0.28 * 25 - 2, 1.05 * 25 - 2, r"\textbf{A}")
ax.plot(truth_df["gain"].sort_values(), truth_df["gain"].sort_values(), "k--")
ax.errorbar(
    truth_df["gain"],
    [fit[i].loc["gain", "Mean"] for i in truth_df.index],
    yerr=np.array(
        [
            abs(
                fit[i].loc["gain", ["95% LL", "95% UL"]].values
                - fit[i].loc["gain", "Mean"]
            )
            for i in truth_df.index
        ]
    ).T,
    fmt="o",
    ms=2,
    color="C0",
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
ax.set_xlim(-2, 23)
ax.set_ylim(-2, 23)
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_xlabel(r"$g$ (true)")
ax.set_ylabel(r"$g$ (fit)")

# panel b
ax = fig.add_subplot(gs[1])
ax.text(-0.28 * 0.5 - 0.05, 1.05 * 0.5 - 0.05, r"\textbf{B}")
ax.plot(truth_df["pi"].sort_values(), truth_df["pi"].sort_values(), "k--")
ax.errorbar(
    truth_df["pi"],
    [fit[i].loc["pi", "Mean"] for i in truth_df.index],
    yerr=np.array(
        [
            abs(
                fit[i].loc["pi", ["95% LL", "95% UL"]].values - fit[i].loc["pi", "Mean"]
            )
            for i in truth_df.index
        ]
    ).T,
    fmt="o",
    ms=2,
    color="C0",
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
ax.set_xlim(-0.05, 0.45)
ax.set_ylim(-0.05, 0.45)
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
ax.set_xticklabels([r"$0$", r"$0.1$", r"$0.2$", r"$0.3$", r"$0.4$"])
ax.set_yticklabels([r"$0$", r"$0.1$", r"$0.2$", r"$0.3$", r"$0.4$"])
ax.set_xlabel(r"$\pi$ (true)")
ax.set_ylabel(r"$\pi$ (fit)")

# panel c
ax = fig.add_subplot(gs[2])
ax.text(-0.28 * 1.2 - 0.1, 1.05 * 1.2 - 0.1, r"\textbf{C}")
ax.plot(truth_df["lamda"].sort_values(), truth_df["lamda"].sort_values(), "k--")
ax.errorbar(
    truth_df["lamda"],
    [fit[i].loc["lamda", "Mean"] for i in truth_df.index],
    yerr=np.array(
        [
            abs(
                fit[i].loc["lamda", ["95% LL", "95% UL"]].values
                - fit[i].loc["lamda", "Mean"]
            )
            for i in truth_df.index
        ]
    ).T,
    fmt="o",
    ms=2,
    color="C0",
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
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xticklabels([r"$0$", r"$0.2$", r"$0.4$", r"$0.6$", r"$0.8$", r"$1$"])
ax.set_yticklabels([r"$0$", r"$0.2$", r"$0.4$", r"$0.6$", r"$0.8$", r"$1$"])
ax.set_xlabel(r"$\lambda$ (true)")
ax.set_ylabel(r"$\lambda$ (fit)")

# panel d
ax = fig.add_subplot(gs[3])
ax.text(-0.28 * 0.55 + 0.15, 1.05 * 0.55 + 0.15, r"\textbf{D}")
ax.plot(truth_df["proximity"].sort_values(), truth_df["proximity"].sort_values(), "k--")
ax.errorbar(
    truth_df["proximity"].sort_values(),
    [
        fit[i].loc["proximity", "Mean"]
        for i in truth_df.sort_values(by="proximity").index
    ],
    yerr=np.array(
        [
            abs(
                fit[i].loc["proximity", ["95% LL", "95% UL"]].values
                - fit[i].loc["proximity", "Mean"]
            )
            for i in truth_df.sort_values(by="proximity").index
        ]
    ).T,
    fmt="o",
    ms=2,
    color="C0",
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
ax.set_xlim(0.15, 0.7)
ax.set_ylim(0.15, 0.7)
ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_xticklabels([r"$0.2$", r"$0.3$", r"$0.4$", r"$0.5$", r"$0.6$"])
ax.set_yticklabels([r"$0.2$", r"$0.3$", r"$0.4$", r"$0.5$", r"$0.6$"])
ax.set_xlabel(r"$\sigma^{xy}$ (true)")
ax.set_ylabel(r"$\sigma^{xy}$ (fit)")

plt.savefig("figures/tapqir_analysis_randomized.png", dpi=600)
