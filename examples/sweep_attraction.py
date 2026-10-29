# Sweep over attraction strength at fixed flow.
#
# For each Morse depth eps, run the toy model and print the transit-time
# summary. The hypothesis under test: as attraction grows, particles get
# captured into aggregates that move as a plug, so the long tail of the
# transit-time distribution (slow near-wall particles) shrinks.
#
# Usage (from the repository root):
#   python examples/sweep_attraction.py

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from bloodflow import Params, cluster_stats, simulate, transit_summary


def main():
    base = Params(
        N=60, a=1.0, H=16.0, L=40.0, eta=1.0, Umax=1.0,
        kappa=2.0, eps_rep=1.0, kT=0.0, dt=0.02, seed=1,
    )
    phi = base.N * np.pi * base.a ** 2 / (base.L * base.H)
    print(f"channel H={base.H}, L={base.L}, a={base.a} (a/H={base.a / base.H:.3f}), "
          f"area fraction phi={phi:.2f}, Umax={base.Umax}, "
          f"wall shear={base.gamma_w:.3f}, dt={base.dt}")
    print(f"{'eps':>5} {'Mason':>7} {'mean':>7} {'median':>7} {'p95':>7} "
          f"{'frac>3x':>8} {'compl':>6} {'wall_t':>7} {'clust':>6} {'in_cl':>6}")
    for eps in (0.0, 0.5, 2.0):
        p = replace(base, eps=eps)
        res = simulate(p, n_steps=8000, equil_steps=500)
        s = transit_summary(res.transit, res.wall_time)
        # Tight cutoff (just past contact): bound Morse pairs sit near
        # r = 2a, while a generous cutoff percolates at this density and
        # reports one giant cluster for every eps.
        c = cluster_stats(res.final_pos, cutoff=2.0 * p.a + 0.5 / p.kappa)
        print(f"{eps:5.1f} {p.mason_number():7.1f} {s['mean']:7.1f} "
              f"{s['median']:7.1f} {s['p95']:7.1f} "
              f"{s['frac_slower_than_3x_mean']:8.3f} {s['frac_completed']:6.2f} "
              f"{s['mean_wall_residence']:7.1f} {c['mean_cluster_size']:6.2f} "
              f"{c['frac_in_clusters']:6.2f}")


if __name__ == "__main__":
    main()
