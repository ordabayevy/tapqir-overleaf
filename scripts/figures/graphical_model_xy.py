import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from tapqir.distributions import AffineBeta

mpl.rc("text", usetex=True)
mpl.rcParams.update({"font.size": 8})

fig = plt.figure(figsize=(2.5, 2.5), constrained_layout=False)

ax = fig.add_subplot()

x = torch.arange(-7.5, 7.5, 0.1)
d1 = AffineBeta(0, 2, -7.5, 7.5)
d2 = AffineBeta(0, 230, -7.5, 7.5)
ax.plot(x, d2.log_prob(x).exp(), color="C2", label="specific")
ax.plot(x, d1.log_prob(x).exp(), color="C3", label="non-specific")
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
ax.set_xlabel(r"$x$ or $y$")
ax.set_ylabel("Probability density")
ax.set_xticks([-6, 0, 6])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xlim(-8, 8)
ax.set_ylim(-0.03, 1.2)
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("figures/graphical_model_xy.png", dpi=600)
