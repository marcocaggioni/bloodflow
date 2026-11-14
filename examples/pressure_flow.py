"""Pressure-driven flow sweep: velocity profiles and plug widths.

Sweeps the pressure gradient G for eps in {0, 0.64, 1.28} at the
Couette scale (N=397, H=50, L=100, phi=0.25). For each point reports
the time-averaged particle and solvent velocity profiles, the plug
half-width (profile-based at two thresholds and stress-based), the
total-stress linearity check, and transit statistics.

Couette-measured (Bingham) yield stresses used for the plug
prediction: tau_y = 0.0315 (eps=0.64), 0.072 (eps=1.28); eps=0 has
no yield (tau_y_pred = 0).

Usage: python3 examples/pressure_flow.py [--out-prefix pressure-flow]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bloodflow import pressure
from bloodflow.core import Params

TAU_Y = {0.0: 0.0, 0.64: 0.0315, 1.28: 0.072}
G_VALUES = [0.001, 0.002, 0.004, 0.008]
EPS_VALUES = [0.0, 0.64, 1.28]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="pressure-flow")
    ap.add_argument("--equil-steps", type=int, default=20000)
    ap.add_argument("--meas-steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    base = dict(N=397, H=50.0, L=100.0, eta=1.0, dt=0.02, seed=args.seed)
    rows = []
    profiles = {}
    for eps in EPS_VALUES:
        for G in G_VALUES:
            p = Params(eps=eps, **base)
            r = pressure.run_point(
                p, G, equil_steps=args.equil_steps,
                meas_steps=args.meas_steps, tau_y_pred=TAU_Y[eps])
            ts = r["transit_summary"]
            f = r["stress_fit"]
            print(
                f"eps={eps:.2f} G={G:.3f} tau_wall={r['tau_wall']:.3f} "
                f"Umax_part={r['Umax_particles']:.3f} "
                f"Umax_solv={r['Umax_solvent']:.3f} "
                f"eta_eff={r['eta_eff']:.3f} "
                f"plug05={r['plug_half_width_05']:.1f} "
                f"plug10={r['plug_half_width_10']:.1f} "
                f"plug_stress={r['plug_half_width_stress']:.1f} "
                f"plug_pred={r['plug_half_width_pred']:.1f} "
                f"slope/G={f['slope_over_G']:.3f} stressR2={f['r2']:.4f} "
                f"centreV {r['centre_v_first']:.3f}/{r['centre_v_second']:.3f} "
                f"transit p95={ts.get('p95', float('nan')):.1f} "
                f"completed={ts['frac_completed']:.2f}",
                flush=True)
            rows.append({
                "eps": eps, "G": G, "tau_wall": r["tau_wall"],
                "Umax_particles": r["Umax_particles"],
                "Umax_solvent": r["Umax_solvent"],
                "eta_eff": r["eta_eff"],
                "plug_half_width_05": r["plug_half_width_05"],
                "plug_half_width_10": r["plug_half_width_10"],
                "plug_half_width_stress": r["plug_half_width_stress"],
                "plug_half_width_pred": r["plug_half_width_pred"],
                "stress_slope_over_G": f["slope_over_G"],
                "stress_intercept": f["intercept"],
                "stress_intercept_expected": f["intercept_expected"],
                "stress_r2": f["r2"],
                "sigma_xy": r["sigma_xy"],
                "sigma_xy_first": r["sigma_xy_first"],
                "sigma_xy_second": r["sigma_xy_second"],
                "centre_v_first": r["centre_v_first"],
                "centre_v_second": r["centre_v_second"],
                "transit_p95": ts.get("p95", float("nan")),
                "transit_median": ts.get("median", float("nan")),
                "transit_frac_completed": ts["frac_completed"],
                "mean_wall_residence": ts.get("mean_wall_residence", float("nan")),
                "fp_mean_integral": r["fp_mean_integral"],
                "n_samples": r["n_samples"],
                "equil_steps": r["equil_steps"],
                "meas_steps": r["meas_steps"],
            })
            profiles[(eps, G)] = r

    pd.DataFrame(rows).to_csv(f"{args.out_prefix}.csv", index=False)

    # Profiles CSV: one row per (eps, G, y-bin) with v and u.
    prof_rows = []
    for (eps, G), r in profiles.items():
        for y, v in zip(r["y_vbins"], r["v_profile"]):
            prof_rows.append({"eps": eps, "G": G, "y": y, "v_particles": v,
                              "u_solvent": float(np.interp(y, r["y_nodes"], r["u_profile"]))})
    pd.DataFrame(prof_rows).to_csv(f"{args.out_prefix}-profiles.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: particle velocity profiles, one panel per eps.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    colors = {0.001: "tab:blue", 0.002: "tab:green", 0.004: "tab:orange",
              0.008: "tab:red"}
    for ax, eps in zip(axes, EPS_VALUES):
        for G in G_VALUES:
            r = profiles[(eps, G)]
            ax.plot(r["y_vbins"], r["v_profile"], "o-",
                    color=colors[G], ms=3,
                    label=f"G={G} ($\\tau_w$={r['tau_wall']:.2f})")
            # Plug prediction as a shaded band.
            hw = r["plug_half_width_pred"]
            if hw > 0:
                ax.axvspan(25 - hw, 25 + hw, color=colors[G], alpha=0.07)
        ax.set_xlabel("height y")
        ax.set_title(f"eps = {eps} ($\\tau_y$={TAU_Y[eps]})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("particle velocity v(y)")
    fig.suptitle("Pressure-driven flow: velocity profiles "
                 "(shaded: plug predicted from Couette $\\tau_y/G$)")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}-profiles.png", dpi=150)

    # Figure 2: total stress profiles (linearity check).
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, eps in zip(axes, EPS_VALUES):
        for G in G_VALUES:
            r = profiles[(eps, G)]
            ax.plot(r["plane_heights"], r["tau_total"], "o-",
                    color=colors[G], ms=3, label=f"G={G}")
            yline = np.linspace(0, 50, 50)
            ax.plot(yline, G * (25 - yline), "--", color=colors[G], lw=0.8)
        ax.set_xlabel("height y")
        ax.set_title(f"eps = {eps}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("total stress $\\tau(y)$")
    fig.suptitle("Total stress $\\eta u' + \\sigma^p_{xy}$ vs y "
                 "(dashed: $G(H/2-y)$)")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}-stress.png", dpi=150)
    print(f"wrote {args.out_prefix}.csv, {args.out_prefix}-profiles.csv, "
          f"{args.out_prefix}-profiles.png, {args.out_prefix}-stress.png")


if __name__ == "__main__":
    main()
