"""Snapshot figure for the plug-flow hypothesis article: final particle
positions after 6,000 steps (120 time units) in the straight channel,
no attraction vs strong attraction, same initial positions.
Grey = single particle, colour = in an aggregate (cutoff 2a + 0.5/kappa).

Usage:
    python3 examples/snapshot_figure.py [--out FILE]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bloodflow import core
from bloodflow.core import Params

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def cluster_colors(pos, p, cutoff):
    """Vectorised clustering; singletons grey, clusters tab colours."""
    n = pos.shape[0]
    dx = pos[:, 0][:, None] - pos[:, 0][None, :]
    dx -= p.L * np.round(dx / p.L)  # minimum image in x
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    adj = (dx * dx + dy * dy) < cutoff * cutoff
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    iu = np.triu_indices(n, k=1)
    for i, j in zip(*iu):
        if adj[i, j]:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[ri] = rj
    labels = np.array([find(i) for i in range(n)])
    _, counts = np.unique(labels, return_counts=True)
    size_of = {lab: int(c) for lab, c in zip(np.unique(labels), counts)}
    ranked = sorted(
        (lab for lab in np.unique(labels) if size_of[lab] >= 2),
        key=lambda lab: -size_of[lab],
    )
    palette = list(mcolors.TABLEAU_COLORS.values()) + list(
        mcolors.CSS4_COLORS.values()
    )
    col_of = {}
    for k, lab in enumerate(ranked):
        col_of[lab] = mcolors.to_rgba(palette[k % len(palette)])
    grey = (0.72, 0.72, 0.72, 1.0)
    return np.array([col_of.get(lab, grey) for lab in labels])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="snapshot-plug-vs-poiseuille.png")
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()

    a, H, L, phi = 1.0, 50.0, 100.0, 0.25
    n = int(phi * L * H / (np.pi * a * a))
    base = dict(N=n, a=a, H=H, L=L, eta=1.0, Umax=2.0,
                kappa=2.0, kT=0.0, dt=0.02, seed=7)
    p_off = Params(eps=0.0, **base)
    p_on = Params(eps=1.28, **base)
    rng = np.random.default_rng(7)
    pos_off = core.init_positions(p_off, rng)
    pos_on = pos_off.copy()
    rng_off = np.random.default_rng(7)
    rng_on = np.random.default_rng(7)
    for _ in range(args.steps):
        core._step(pos_off, p_off, rng_off, 0.0)
        core._step(pos_on, p_on, rng_on, 0.0)

    cutoff = 2.0 * a + 0.5 / 2.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), dpi=130)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.12,
                        wspace=0.08)
    for ax, pos, p, title in zip(
        axes, (pos_off, pos_on), (p_off, p_on),
        ("(a) No attraction (Mason 0)",
         "(b) Strong attraction (Mason 16)"),
    ):
        ax.set_xlim(0, L)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x (flow direction)")
        ax.scatter(pos[:, 0], pos[:, 1],
                   c=cluster_colors(pos, p, cutoff),
                   s=18, edgecolors="none")
    axes[0].set_ylabel("y (channel width)")
    axes[1].set_yticklabels([])
    fig.suptitle(f"Particle positions at t = {args.steps * 0.02:.0f} "
                 f"time units (grey = single, colour = in an aggregate)",
                 fontsize=12)
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out, os.path.getsize(args.out), "bytes")


if __name__ == "__main__":
    main()
