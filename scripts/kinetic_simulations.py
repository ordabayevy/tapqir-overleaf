# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from pyroapi import distributions as dist
from pyroapi import pyro, pyro_backend
from tapqir.models import HMM
from tapqir.utils.dataset import save
from tapqir.utils.simulate import simulate


def main(args):
    if args.cuda:
        device = "cuda"
    else:
        device = "cpu"
    pyro.set_rng_seed(args.seed)
    params = {}
    params["width"] = 1.4
    params["gain"] = 7
    params["kon"] = args.kon
    params["koff"] = 0.2
    params["lamda"] = args.lamda
    params["proximity"] = 0.2
    params["offset"] = 90
    params["height"] = 3000
    params["background"] = 150

    model = HMM(1, 2, device=device)
    C = 1
    model.vectorized = False
    data = simulate(
        model,
        args.N,
        args.F,
        C,
        args.P,
        seed=args.seed,
        params=params,
    )
    # save data
    args.path.mkdir(exist_ok=True)
    save(data, args.path)
    # calculate snr
    rv = dist.MultivariateNormal(
        torch.tensor([(args.P - 1) / 2, (args.P - 1) / 2]),
        scale_tril=torch.eye(2) * params["width"],
    )
    P_range = torch.arange(args.P, dtype=model.dtype)
    i_pixel, j_pixel = torch.meshgrid(P_range, P_range, indexing="xy")
    ij_pixel = torch.stack((i_pixel, j_pixel), dim=-1)
    weights = rv.log_prob(ij_pixel).exp()
    signal = (weights**2 * params["height"]).sum()
    noise = math.sqrt((params["background"] * params["gain"]))
    params["SNR"] = float(signal / noise)
    params["N"] = args.N
    params["F"] = args.F
    params["P"] = args.P

    pd.Series(params).to_csv(Path(args.path) / "simulated_params.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM Simulations")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--kon", default=1e-3, type=float)
    parser.add_argument("--lamda", default=0.15, type=float)
    parser.add_argument("-N", default=5, type=int)
    parser.add_argument("-F", default=500, type=int)
    parser.add_argument("-P", default=14, type=int)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--funsor", action="store_true")
    args = parser.parse_args()

    if args.funsor:
        import funsor

        funsor.set_backend("torch")
        import pyro.contrib.funsor

        PYRO_BACKEND = "contrib.funsor"
    else:
        PYRO_BACKEND = "pyro"

    with pyro_backend(PYRO_BACKEND):
        main(args)
