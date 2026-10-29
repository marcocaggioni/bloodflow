"""Particle-age analysis: how long has each particle been in the field of
view at the end of the run?

Definition: the channel is periodic in x, so a particle "enters" the field
of view each time it wraps around (exits right, re-enters left). A
particle's age at the end of the measurement phase is the time since its
most recent wrap; particles that never wrapped during the measurement
phase have age = the full measurement duration (they have been in view
the whole time -- the censored/old end of the distribution).

This complements the transit-time analysis: transit time measures how
long a crossing takes, while age measures, at one snapshot, how long the
particles currently in view have already been there. The hypothesis is
the same: Poiseuille flow should show many very old (near-wall) particles
and a broad age distribution, while aggregated plug flow should show a
narrower, more uniform age distribution.

Outputs:
  - CSV with one row per particle: case, particle, age, n_wraps
  - PNG: overlaid age histograms with medians
  - Printed summary: mean, std, CV, median, IQR, p95, fraction at max age

Usage:
    python3 examples/particle_age_analysis.py [--out-prefix PREFIX]
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bloodflow import core
from bloodflow.core import Params


def run_age(p, n_steps, equil_steps, pos0):
    """Run and return (age, n_wraps) at the end of the measurement phase."""
    rng = np.random.default_rng(p.seed)
    pos = pos0.copy()
    for _ in range(equil_steps):
        core._step(pos, p, rng, 0.0)
    # Measurement phase: age clock starts now for every particle.
    last_wrap = np.zeros(p.N, dtype=int)
    n_wraps = np.zeros(p.N, dtype=int)
    for step in range(1, n_steps + 1):
        x_old = pos[:, 0].copy()
        dx = core._step(pos, p, rng, 0.0)
        wrapped = (x_old + dx >= p.L) | (x_old + dx < 0.0)
        last_wrap[wrapped] = step
        n_wraps[wrapped] += 1
    age = (n_steps - last_wrap) * p.dt
    return age, n_wraps


def summarize(age):
    mean = float(np.mean(age))
    q25, q75 = np.percentile(age, [25, 75])
    return {
        "mean": mean,
        "std": float(np.std(age)),
        "cv": float(np.std(age) / mean),
        "median": float(np.median(age)),
        "iqr": float(q75 - q25),
        "p95": float(np.percentile(age, 95)),
        "frac_at_max_age": float(np.mean(age >= age.max() - 1e-9)),
        "max": float(age.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="particle-age")
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
          f"steps={args.steps} equil={args.equil_steps}  "
          f"measurement duration={args.steps * 0.02:.0f} time units")

    rng = np.random.default_rng(7)
    pos0 = core.init_positions(p_off, rng)

    results = {}
    for name, p in (("no-attraction", p_off), ("attraction", p_on)):
        age, n_wraps = run_age(p, args.steps, args.equil_steps, pos0)
        results[name] = (age, n_wraps)
        s = summarize(age)
        print(f"\n[{name}]")
        for k, v in s.items():
            print(f"  {k}: {v:.4g}")

    csv_path = f"{args.out_prefix}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "particle", "age", "n_wraps"])
        for name, (age, n_wraps) in results.items():
            for i in range(n):
                w.writerow([name, i, f"{age[i]:.4f}", int(n_wraps[i])])
    print("\nwrote", csv_path)

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for name, color in (("no-attraction", "#4a7bb5"),
                            ("attraction", "#d95f2b")):
            age = results[name][0]
            ax.hist(age, bins=40, density=True, alpha=0.55,
                    color=color, label=name)
            ax.axvline(np.median(age), color=color, linestyle="--",
                       label=f"{name} median")
        ax.set_xlabel("Particle age in field of view at end of run "
                      "(time units)")
        ax.set_ylabel("Density")
        ax.set_title("Particle-age distribution: Poiseuille vs aggregated flow")
        ax.legend(fontsize=9)
        png_path = f"{args.out_prefix}.png"
        fig.savefig(png_path, dpi=130, bbox_inches="tight")
        print("wrote", png_path)


if __name__ == "__main__":
    main()
