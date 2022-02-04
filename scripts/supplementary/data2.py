"""
Supplementary File 2
--------------------

Randomized simulation parameters and corresponding fit values

To generate source image file ``supplementary/data2.xlsx``, run::

  python scripts/supplementary/data2.py

Input data:

* ``simulations/seed*``
"""

from pathlib import Path

import pandas as pd
from utils import description, resize_columns

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
description.name = (
    "Supplementary File 2: Randomized simulation parameters "
    "and corresponding fit values"
)

with pd.ExcelWriter("supplementary/data2.xlsx", engine="xlsxwriter") as writer:
    description.to_excel(writer, sheet_name="Description")
    worksheet = writer.sheets["Description"]
    worksheet.set_column(0, 0, 15)
    worksheet.set_column(1, 1, 90)

    truth_df.drop(columns="SNR").to_excel(
        writer, float_format="%.4f", sheet_name="Simulation inputs"
    )
    resize_columns(writer, truth_df, "Simulation inputs")

    for key in truth_df.index:
        fit[key].to_excel(writer, float_format="%.4f", sheet_name=key)
        resize_columns(writer, fit[key], key)
