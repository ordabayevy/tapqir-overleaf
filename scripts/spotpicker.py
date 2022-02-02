# Copyright Contributors to the Tapqir project.
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import math

import torch
from pyroapi import distributions as dist
from scipy.io import savemat
from tapqir.models import Cosmos
from tapqir.utils.simulate import simulate


def main(args):
    device = "cuda"
    N, F, P = 10, 500, 14
    params = {}
    params["width"] = 1.4
    params["gain"] = 7
    params["pi"] = 0.15
    params["lamda"] = 1
    params["proximity"] = 0.2
    params["offset"] = 90
    params["height"] = 3000
    params["background"] = 150

    model = Cosmos(1, 2, device=device)
    C = 1
    samples = simulate(
        model,
        N,
        F,
        C,
        P,
        seed=args.seed,
        params=params,
    )
    # create meshgrid of PxP pixel positions
    i_range = torch.arange(P * 11)
    j_range = torch.arange(P * 5)
    j_pixel, i_pixel = torch.meshgrid(j_range, i_range)
    ij_pixel = torch.stack((i_pixel, j_pixel), dim=-1)

    # Ideal 2D gaussian spots
    target_locs = torch.zeros(2 * N, F, 2)
    for n in range(N):
        target_locs[n, :, 0] = (1 + 2 * (n % N)) * P + (P - 1) / 2
        target_locs[n, :, 1] = P + (P - 1) / 2
        target_locs[N + n, :, 0] = (1 + 2 * (n % N)) * P + (P - 1) / 2
        target_locs[N + n, :, 1] = 3 * P + (P - 1) / 2

    gain = params["gain"]
    background = torch.full((F,), params["background"])
    height = torch.full((2 * N, F, 2), params["height"])
    width = torch.full((2 * N, F, 2), params["width"])
    x = torch.stack([samples[f"x_{k}"][0] for k in range(2)], -1)
    xc = torch.stack([samples[f"x_{k}"][0] for k in range(2)], -1)
    x = torch.cat([x, xc], dim=0)
    y = torch.stack([samples[f"y_{k}"][0] for k in range(2)], -1)
    yc = torch.stack([samples[f"y_{k}"][0] for k in range(2)], -1)
    y = torch.cat([y, yc], dim=0)
    m = torch.stack([samples[f"m_{k}"][0] for k in range(2)], -1)
    mc = torch.stack([samples[f"m_{k}"][0] for k in range(2)], -1)
    m = torch.cat([m, mc], dim=0)
    spot_locs = target_locs.unsqueeze(-2) + torch.stack((x, y), -1)
    scale = width[..., None, None, None]
    loc = spot_locs[..., None, None, :]
    var = scale**2
    normalized_gaussian = torch.exp(
        (
            -((ij_pixel - loc) ** 2) / (2 * var)
            - scale.log()
            - math.log(math.sqrt(2 * math.pi))
        ).sum(-1)
    )
    if m is not None:
        height = m * height
    gaussian_spots = height[..., None, None] * normalized_gaussian
    image = background[..., None, None] + gaussian_spots.sum((-5, -3))
    image_samples = dist.Gamma(image / gain, 1 / gain).sample() + params["offset"]
    data = {}
    data["images"] = image_samples.long().cpu().numpy()
    data["aoiinfo"] = target_locs[:, 0, :].cpu().numpy() + 1  # adjust to matlab
    savemat("simulations/lamda1.mat", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lamda Simulation for Spot-picker")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dtype", default="double", type=str)
    args = parser.parse_args()
    main(args)
