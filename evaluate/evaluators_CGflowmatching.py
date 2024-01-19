"""
Code provided by Yaoyi Chen, shared first author of:
Köhler, Jonas, et al. "Flow-matching: Efficient coarse-graining of molecular dynamics without forces." 
Journal of Chemical Theory and Computation 19.3 (2023): 942-952.
"""

import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from matplotlib import cm

K_B = 1.380650324e-23  # J/K
T = 300  # K
PER_MOL = 6.02214076e23  # /mol
J_PER_CAL = 4.184  # J/cal
K_BT_IN_KCAL_PER_MOL = K_B * T * PER_MOL / 1000 / J_PER_CAL


def mse(density1, density2):
    """
    Compute the mse of free energy between discrete prob. distribution
    """
    with np.errstate(divide="ignore"):
        U1 = K_BT_IN_KCAL_PER_MOL * np.log(density1)
        U2 = K_BT_IN_KCAL_PER_MOL * np.log(density2)
    U1 = np.where(np.isinf(U1), np.NaN, U1)
    U2 = np.where(np.isinf(U2), np.NaN, U2)
    count = np.sum(np.isfinite(U1 - U2))
    return np.nansum(np.square(U1 - U2)) / count


def get_torsions(coords, topology):
    """
    Computes phi and psi angles for inputting coordinates
    Coords shape: [N, 5, 3]
    """
    t = md.Trajectory(coords, topology)
    return md.compute_dihedrals(t, [[0, 1, 2, 3], [1, 2, 3, 4]])


def get_prob(tors_data, n_bins=61):
    """
    Compute a histogram over phi-psi space
    Approximates the equilibrium probability distributions
    """
    bin_edges = np.linspace(-np.pi, np.pi, n_bins)
    hist, _, _ = np.histogram2d(
        tors_data[:, 0], tors_data[:, 1], bins=bin_edges, density=True
    )
    prob = hist / hist.sum()
    return prob


def kl_div(density1, density2):
    """
    Compute a kl divergence between discrete prob. distribution
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = density2 / density1
    ratio[density1 == 0] = 1  # so that the log value will be 0
    ratio[density2 == 0] = 1  # so that the log value will be 0
    return -np.nansum(density1 * np.log(ratio))


def plot_free_E_2d(prob, ax, unit_conv=1.0, cax=None, title=None, n_bins=61):
    """
    Plot free energy in a 2D plot
    """
    with np.errstate(divide="ignore"):
        ys = -np.log(prob) * unit_conv
    ys -= ys.min()
    bin_edges = np.linspace(-np.pi, np.pi, n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cc = ax.contourf(
        bin_centers,
        bin_centers,
        ys.T,
        vmax=5,
        levels=np.linspace(0.0, 5.5, 12),
        extend="max",
        antialiased=False,
        cmap="magma",
    )
    cbar = plt.colorbar(cc, cax=cax, ax=ax)
    cbar.set_label("Free energy / kcal$\cdot$mol$^{-1}$")
    cbar.ax.tick_params()
    line_colors = []
    for i, j in enumerate(np.linspace(0, 1, 12)):
        if i % 2 == 0 and i < 9:
            line_colors.append(cm.binary(j))
        else:
            line_colors.append(
                (0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 0.0)
            )  # transparent
    ax.contour(cc, colors=line_colors, linewidths=1.5, antialiased=True)
    ax.set_title(title)
