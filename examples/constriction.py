"""Constricted-channel simulation: Poiseuille vs aggregated flow through
a stenosis-like narrowing.

Geometry: main channel width 50, throat width 10 (= 5 particle
diameters for a = 1), flat throat core 20 long with cosine ramps of 15
on each side, then back to full width. Flow is flux-conserving
(lubrication approximation), so the centreline velocity in the throat is
5x the main-channel value -- and the local wall shear 25x, which drops
the local Mason number from 16 to 0.64: aggregates that survive the main
channel should break up in the throat and may reform downstream.

Outputs:
  - CSV: case, particle, transit, throat_passage, wall_time, completed
  - PNG: final snapshot of both cases over the channel outline
  - Printed summary: transit stats, throat-passage stats, wall residence,
    clustering overall and per region (upstream / throat / downstream)

Usage:
    python3 examples/constriction.py [--out-prefix PREFIX]
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bloodflow import analysis, core
from bloodflow.constriction import (
    Constriction,
    init_positions_constriction,
    simulate_constriction,
)
from bloodflow.core import Params


def stats(vals):
    done = vals[~np.isnan(vals)]
    if done.size == 0:
        return {"frac_completed": 0.0}
    mean = float(np.mean(done))
    return {
        "frac_completed": float(done.size / vals.size),
        "mean": mean,
        "std": float(np.std(done)),
        "cv": float(np.std(done) / mean),
        "median": float(np.median(done)),
        "p95": float(np.percentile(done, 95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="constriction")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--equil-steps", type=int, default=3000)
    ap.add_argument("--L", type=float, default=100.0)
    ap.add_argument("--xc", type=float, default=50.0)
    args = ap.parse_args()

    a, H_main, H_throat, L, phi = 1.0, 50.0, 10.0, args.L, 0.25
    constr = Constriction(H_main=H_main, H_throat=H_throat, L=L,
                          xc=args.xc, core_half=10.0, ramp=15.0)
    n = int(phi * L * H_main / (np.pi * a * a))
    xs_grid = np.linspace(0, L, 2001)
    h_avg = float(np.mean(constr.width(xs_grid)))
    base = dict(N=n, a=a, H=H_main, L=L, eta=1.0, Umax=2.0,
                kappa=2.0, kT=0.0, dt=0.02, seed=7)
    p_off = Params(eps=0.0, **base)
    p_on = Params(eps=1.28, **base)
    mason_main = p_on.mason_number()
    mason_throat = mason_main * (H_throat / H_main) ** 2
    print(f"N={n}  main width={H_main} throat width={H_throat} "
          f"(= 5 particle diameters)  mean width={h_avg:.1f} "
          f"effective phi={phi * H_main / h_avg:.2f}")
    print(f"Mason: main channel {mason_main:.1f}, throat {mason_throat:.2f}")
    print(f"steps={args.steps} equil={args.equil_steps}")

    rng = np.random.default_rng(7)
    pos0 = init_positions_constriction(p_off, constr, rng)

    results = {}
    for name, p in (("no-attraction", p_off), ("attraction", p_on)):
        res = simulate_constriction(p, constr, args.steps,
                                    equil_steps=args.equil_steps, pos0=pos0)
        results[name] = res
        print(f"\n[{name}] transit:")
        for k, v in stats(res.transit).items():
            print(f"  {k}: {v:.4g}")
        print(f"[{name}] throat passage:")
        for k, v in stats(res.passage).items():
            print(f"  {k}: {v:.4g}")
        print(f"[{name}] mean wall residence: "
              f"{np.mean(res.wall_time):.4g}")
        print(f"[{name}] safety-valve re-injections: {res.n_repairs}")
        labels = analysis.cluster_labels(
            res.final_pos, 2.0 * a + 0.5 / p.kappa
        )
        _, counts = np.unique(labels, return_counts=True)
        size_of = {lab: int(c) for lab, c in zip(np.unique(labels), counts)}
        in_cl = np.array([size_of[lab] >= 2 for lab in labels])
        print(f"[{name}] frac in clusters (final snapshot): "
              f"{np.mean(in_cl):.3f}")
        x1, x2 = constr.xc - constr.core_half, constr.xc + constr.core_half
        for region, mask in (
            ("upstream", res.final_pos[:, 0] < x1),
            ("throat", (res.final_pos[:, 0] >= x1)
             & (res.final_pos[:, 0] <= x2)),
            ("downstream", res.final_pos[:, 0] > x2),
        ):
            if mask.sum():
                print(f"[{name}]   {region}: n={int(mask.sum())} "
                      f"frac in clusters={np.mean(in_cl[mask]):.3f}")

    csv_path = f"{args.out_prefix}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "particle", "transit", "throat_passage",
                    "wall_time", "completed"])
        for name, res in results.items():
            for i in range(res.params.N):
                t = res.transit[i]
                pg = res.passage[i]
                w.writerow([name, i,
                            "" if np.isnan(t) else f"{t:.4f}",
                            "" if np.isnan(pg) else f"{pg:.4f}",
                            f"{res.wall_time[i]:.4f}",
                            int(not np.isnan(t))])
    print("\nwrote", csv_path)

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
        xs = np.linspace(0, L, 400)
        for ax, (name, res) in zip(axes, results.items()):
            ax.fill_between(xs, constr.y_lo(xs), constr.y_hi(xs),
                            color="#dbe7f3", zorder=0)
            ax.plot(xs, constr.y_lo(xs), color="#4a7bb5")
            ax.plot(xs, constr.y_hi(xs), color="#4a7bb5")
            labels = analysis.cluster_labels(
                res.final_pos, 2.0 * a + 0.5 / res.params.kappa
            )
            _, counts = np.unique(labels, return_counts=True)
            size_of = {lab: int(c)
                       for lab, c in zip(np.unique(labels), counts)}
            ranked = sorted(
                (lab for lab in np.unique(labels) if size_of[lab] >= 2),
                key=lambda lab: -size_of[lab],
            )
            palette = list(mcolors.TABLEAU_COLORS.values())
            col_of = {lab: mcolors.to_rgba(palette[k % len(palette)])
                      for k, lab in enumerate(ranked)}
            grey = (0.72, 0.72, 0.72, 1.0)
            cols = np.array([col_of.get(lab, grey) for lab in labels])
            ax.scatter(res.final_pos[:, 0], res.final_pos[:, 1],
                       c=cols, s=18, edgecolors="none", zorder=3)
            ax.set_xlim(0, L)
            ax.set_ylim(0, H_main)
            ax.set_aspect("equal")
            ax.set_title(name)
            ax.set_xlabel("x (flow direction)")
        axes[0].set_ylabel("y")
        fig.suptitle("Final snapshot in the constricted channel "
                     "(grey = single, colour = in an aggregate)")
        png_path = f"{args.out_prefix}.png"
        fig.savefig(png_path, dpi=130, bbox_inches="tight")
        print("wrote", png_path)


if __name__ == "__main__":
    main()
