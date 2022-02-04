"""
Supplementary File 5
--------------------

Kinetic simulation parameters and corresponding fit values

To generate source image file ``supplementary/data5.xlsx``, run::

  python scripts/supplementary/data5.py

Input data:

* ``simulations/kon*``
"""

from pathlib import Path

import pandas as pd
import torch.distributions as dist
from pyro.ops.stats import pi
from tapqir.models import Cosmos
from tapqir.utils.imscroll import association_rate, dissociation_rate
from utils import description, resize_columns

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

        statistics = pd.read_csv(data_path / "cosmos-channel0-summary.csv", index_col=0)

        fit[data_path.name] = statistics.astype(float)
        for p in ("gain", "proximity", "lamda", "SNR"):
            fit[data_path.name].loc[p, "True"] = truth[data_path.name][p]

        model = Cosmos()
        model.load(data_path, data_only=False)

        z_samples = dist.Bernoulli(model.params["z_probs"]).sample((500,))
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

truth_df = pd.concat(truth.values(), axis=1).T.astype(float).drop(columns=["Fc"])
truth_df["Keq"] = truth_df["kon"] / truth_df["koff"]
truth_df = truth_df.sort_values(by=["kon", "lamda"])
description.name = (
    "Supplementary File 5: Kinetic simulation parameters and corresponding fit values"
)

with pd.ExcelWriter("supplementary/data5.xlsx", engine="xlsxwriter") as writer:
    description.to_excel(writer, sheet_name="Description")
    worksheet = writer.sheets["Description"]
    worksheet.set_column(0, 0, 15)
    worksheet.set_column(1, 1, 150)

    truth_df.drop(columns="SNR").to_excel(
        writer, float_format="%.4f", sheet_name="Simulation inputs"
    )
    resize_columns(writer, truth_df, "Simulation inputs")

    for key in truth_df.index:
        fit[key].to_excel(writer, float_format="%.4f", sheet_name=key)
        resize_columns(writer, fit[key], key)
