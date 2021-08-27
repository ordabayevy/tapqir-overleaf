from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pyro
import torch
from pyro.infer import Predictive
from tapqir.models import Cosmos

# path_data = Path("experimental/Rpb1SNAP549")
path_data = Path("experimental/sigma54RNAPCy3-597P255")
model = Cosmos(verbose=False)
model.load(path_data, data_only=False)

model.load_checkpoint(param_only=True)
# model.data.offtarget.images = None
model.data.offtarget = model.data.offtarget._replace(images=None)
predictive = Predictive(
    pyro.poutine.uncondition(model.model), guide=model.guide, num_samples=200
)

summary = {}
summary["mean"] = torch.zeros(200, model.data.ontarget.N, model.data.ontarget.F)
summary["var"] = torch.zeros(200, model.data.ontarget.N, model.data.ontarget.F)
summary["skewness"] = torch.zeros(200, model.data.ontarget.N, model.data.ontarget.F)
summary["kurtosis"] = torch.zeros(200, model.data.ontarget.N, model.data.ontarget.F)

for f in range(model.data.ontarget.F):
    print(f"{f} / {model.data.ontarget.F}")
    model.f = torch.tensor([f])
    pyro.set_rng_seed(0)
    samples = predictive()

    array = samples["d/data"].squeeze()
    mean = torch.mean(array, (-2, -1), keepdim=True)
    diffs = array - mean
    var = torch.mean(torch.pow(diffs, 2.0), (-2, -1), keepdim=True)
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skewness = torch.mean(torch.pow(zscores, 3.0), (-2, -1))
    kurtosis = torch.mean(torch.pow(zscores, 4.0), (-2, -1)) - 3.0

    summary["mean"][..., f] = mean.squeeze()
    summary["var"][..., f] = var.squeeze()
    summary["skewness"][..., f] = skewness.squeeze()
    summary["kurtosis"][..., f] = kurtosis.squeeze()

# torch.save(summary, "experimental/DatasetA_summary.pt")
torch.save(summary, "experimental/DatasetB_summary.pt")
