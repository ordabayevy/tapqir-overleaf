from pathlib import Path

import pandas as pd
import pyro
import torch
from pyro import distributions as dist
from pyro.ops.stats import hpdi, resample
from tapqir.models import Cosmos
from tapqir.utils.imscroll import time_to_first_binding
from tapqir.utils.mle_analysis import train, ttfb_guide, ttfb_model

# load model & parameters
path_data = Path("experimental/Rpb1SNAP549")
model = Cosmos(verbose=False)
model.load(path_data, data_only=False)

# prepare data
Tmax = model.data.ontarget.F
control = None
torch.manual_seed(0)
z = dist.Bernoulli(model.params["p(specific)"]).sample((2000,))
data = time_to_first_binding(z)

# use cuda
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Tapqir fit
train(
    ttfb_model,
    ttfb_guide,
    lr=5e-3,
    n_steps=15000,
    data=data.cuda(),
    control=control,
    Tmax=Tmax,
    jit=False,
)

results = pd.DataFrame(columns=["Mean", "95% LL", "95% UL"])

results.loc["ka", "Mean"] = pyro.param("ka").mean().item()
ll, ul = hpdi(pyro.param("ka").data.squeeze(), 0.95, dim=0)
results.loc["ka", "95% LL"], results.loc["ka", "95% UL"] = ll.item(), ul.item()

results.loc["kns", "Mean"] = pyro.param("kns").mean().item()
ll, ul = hpdi(pyro.param("kns").data.squeeze(), 0.95, dim=0)
results.loc["kns", "95% LL"], results.loc["kns", "95% UL"] = ll.item(), ul.item()

results.loc["Af", "Mean"] = pyro.param("Af").mean().item()
ll, ul = hpdi(pyro.param("Af").data.squeeze(), 0.95, dim=0)
results.loc["Af", "95% LL"], results.loc["Af", "95% UL"] = ll.item(), ul.item()


# Spotpicker fit
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

torch.set_default_tensor_type(torch.cuda.FloatTensor)

train(
    ttfb_model,
    ttfb_guide,
    lr=5e-3,
    n_steps=15000,
    data=bootstrap_data.cuda(),
    control=bootstrap_control.cuda(),
    Tmax=Tmax,
    jit=False,
)

results.loc["ka_sp", "Mean"] = pyro.param("ka").mean().item()
ll, ul = hpdi(pyro.param("ka").data.squeeze(), 0.95, dim=0)
results.loc["ka_sp", "95% LL"], results.loc["ka_sp", "95% UL"] = ll.item(), ul.item()

results.loc["kns_sp", "Mean"] = pyro.param("kns").mean().item()
ll, ul = hpdi(pyro.param("kns").data.squeeze(), 0.95, dim=0)
results.loc["kns_sp", "95% LL"], results.loc["kns_sp", "95% UL"] = ll.item(), ul.item()

results.loc["Af_sp", "Mean"] = pyro.param("Af").mean().item()
ll, ul = hpdi(pyro.param("Af").data.squeeze(), 0.95, dim=0)
results.loc["Af_sp", "95% LL"], results.loc["Af_sp", "95% UL"] = ll.item(), ul.item()

results.to_csv("scripts/extended-data/figure4.csv")
