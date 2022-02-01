"""
Figure 4-Figure supplement 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

False negative spot misidentifications by Tapqir and spot-picker method.

To generate source image file ``figures/tapqir_performance_fn.svg``, run::

  python scripts/figures/tapqir_performance_fn.py

Input data:

* ``simulations/lamda1``
* ``simulations/spotpicker_result.mat`` (spot-picker analysis output)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from tapqir.models import Cosmos

mpl.rc("text", usetex=True)
mpl.rcParams.update({"font.size": 8})

# load data
model = Cosmos()
model.load("simulations/lamda1", data_only=False)

spotpicker = loadmat("simulations/spotpicker_result.mat")

aois, frames = np.nonzero(
    model.data.labels["z"][: model.data.N, :, model.cdx]
    & (
        ~model.params["z_map"][: model.data.N].numpy()
        | ~spotpicker["binary_default1p5"][:5].astype(bool)
    )
)

fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=False)
gs = fig.add_gridspec(
    nrows=4,
    ncols=10,
    top=0.94,
    bottom=0.01,
    left=0.1,
    right=0.99,
    wspace=0.1,
)

for i, (n, f) in enumerate(zip(aois, frames)):
    ax = fig.add_subplot(gs[i // 10, i % 10])
    ax.imshow(model.data.images[n, f, model.cdx], cmap="gray")
    tsign = "+" if model.params["z_map"][n, f] else "-"
    ssign = "+" if spotpicker["binary_default1p5"][n, f] else "-"
    ax.set_title(
        fr"${n}$; ${f}$" + "\n" + fr"${tsign}$" + "\n" + fr"${ssign}$", fontsize=8
    )
    if not i % 10:
        ax.text(
            -10,
            -2.8,
            "AOI; Frame\nTapqir\nSpot-picker",
            ha="center",
        )
    ax.axis("off")

plt.savefig("figures/tapqir_performance_fn.png", dpi=900)
