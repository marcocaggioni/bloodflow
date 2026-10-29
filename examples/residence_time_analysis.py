"""Residence-time distribution analysis: Poiseuille vs aggregated flow.

Runs the toy model twice from identical initial positions (no attraction
vs strong attraction) and tracks every particle's residence time in the
field of view, defined as the time for its unwrapped x-displacement to
reach one channel length L (the simulate() transit time). Particles that
never cross within the run keep NaN (censored) and are counted separately.

Hypothesis under test: the Poiseuille case has a BROADER residence-time
distribution (long near-wall tail) than the aggregated case, where
particles travel together as a plug and the distribution narrows.

Outputs:
  - CSV with one row per particle: case, particle, transit, wall_time,
    completed
  - PNG: overlaid transit-time histograms (log x-axis) with medians
  - Printed summary: n, fraction completed, mean, std, CV (std/mean),
    median, IQR, p95, fraction slower than 3x mean, mean wall residence

Usage:
    python3 examples/residence_time_analysis.py [--out-prefix PREFIX]
Requires matplotlib (for the PNG only; the CSV and summary work without it
if --no-plot is given).
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bloodflow import core
from bloodflow.core import Params


def summarize(transit, wall_time):
    done = transit[~np.isnan(transit)]
    out = {
        "n": int(transit.size),
        "frac_completed": float(done.size / transit.size),
    }
    if done.size:
        mean = float(np.mean(done))
        out["mean"] = mean
        out["std"] = float(np.std(done))
        out["cv"] = float(np.std(done) / mean)
        out["median"] = float(np.median(done))
        q25, q75 = np.percentile(done, [25, 75])
        out["iqr"] = float(q75 - q25)
        out["iqr_over_median"] = float((q75 - q25) / np.median(done))
        out["p95"] = float(np.percentile(done, 95))
        out["frac_slower_than_3x_mean"] = float(np.mean(done > 3.0 * mean))
    if wall_time is not None:
        out["mean_wall_residence"] = float(np.mean(wall_time))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="residence-time")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--equil-steps", type=int, default=3000)
    args = ap.parse_args()

    a, H, L, phi = 1.0, 50.0, 100.0, 0.25
    n = int(phi * L * H / (np.pi * a * a))
    base = dict(N=n, a=a, H=H, L=L, eta=1.0, Umax=2.0,
                kappa=2.0, kT=0.0, dt=0.02, seed=7)
    p_off = Params(eps=0.0, **base)
    p_on = Params(eps=1.28, **base)  # Mason ~ 16 at the wall shear rate
    print(f"N={n}  H={H} (a/H={a / H})  Mason(on)={p_on.mason_number():.1f}  "
          f"steps={args.steps} equil={args.equil_steps}")

    rng = np.random.default_rng(7)
    pos0 = core.init_positions(p_off, rng)

    results = {}
    for name, p in (("no-attraction", p_off), ("attraction", p_on)):
        res = core.simulate(p, args.steps,
                            equil_steps=args.equil_steps, pos0=pos0)
        results[name] = res
        s = summarize(res.transit, res.wall_time)
        print(f"\n[{name}]")
        for k, v in s.items():
            print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")

    csv_path = f"{args.out_prefix}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "particle", "transit", "wall_time", "completed"])
        for name, res in results.items():
            for i in range(res.params.N):
                t = res.transit[i]
                w.writerow([name, i, "" if np.isnan(t) else f"{t:.4f}",
                            f"{res.wall_time[i]:.4f}",
                            int(not np.isnan(t))])
    print("\nwrote", csv_path)

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for name, color in (("no-attraction", "#4a7bb5"),
                            ("attraction", "#d95f2b")):
            done = results[name].transit
            done = done[~np.isnan(done)]
            ax.hist(done, bins=np.logspace(np.log10(max(done.min(), 1.0)),
                                           np.log10(done.max()), 40),
                    density=True, alpha=0.55, color=color, label=name)
            ax.axvline(np.median(done), color=color, linestyle="--",
                       label=f"{name} median")
        ax.set_xscale("log")
        ax.set_xlabel("Residence time in field of view (time units, log scale)")
        ax.set_ylabel("Density")
        ax.set_title("Residence-time distribution: Poiseuille vs aggregated flow")
        ax.legend(fontsize=9)
        png_path = f"{args.out_prefix}.png"
        fig.savefig(png_path, dpi=130, bbox_inches="tight")
        print("wrote", png_path)


if __name__ == "__main__":
    main()
