# Flow curve of the Bloodflow toy model in planar Couette shear.
#
# For each Morse depth eps, impose a shear rate gamma_dot (linear
# Couette field, U(y) = gamma_dot * (y - H/2)) and measure the shear
# stress the particles exert in response. The result is tau(gamma_dot)
# — the material's flow curve — plus the Casson check: for a Casson
# fluid sqrt(tau) = sqrt(tau_y) + sqrt(eta_p * gamma_dot), so a linear
# fit of sqrt(tau) against sqrt(gamma_dot) gives the yield stress
# tau_y (intercept squared) and the plastic viscosity eta_p (slope
# squared). eps = 0 must come out Newtonian (tau_y ~ 0); with
# attraction tau_y must be finite and grow with eps.
#
# Stress reported two ways (they must agree in steady state): the
# particle virial sigma_xy (tau = eta*gamma_dot + sigma_xy) and the
# method-of-planes stress tau_planes measured across planes spanning
# the channel, including planes adjacent to the walls. The run also
# prints the first/second-half means of the averaging window, so a
# point that has not reached steady state can be recognised (at low
# rates with strong attraction the aggregates are long-lived; a
# drifting point is flagged, not trusted).
#
# Usage (from the repository root):
#   python examples/flow_curve.py [--rates ...] [--eps-list ...]
#       [--H 50] [--L 100] [--phi 0.25] [--out flow-curve.csv]
#       [--plot flow-curve.png]
# Requires numpy (and matplotlib only for --plot).

import argparse
import csv
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from bloodflow import Params, cluster_stats
from bloodflow.couette import run_point


def casson_fit(gamma_dot, tau):
    """Linear fit of sqrt(tau) vs sqrt(gamma_dot).

    Returns (tau_y, eta_p, r2, intercept): tau_y = intercept^2,
    eta_p = slope^2. A negative intercept means no yield stress within
    the fit (tau_y is reported as 0 in that case, with the raw
    intercept kept in the printed line).
    """
    x = np.sqrt(np.asarray(gamma_dot, dtype=float))
    y = np.sqrt(np.asarray(tau, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    yfit = slope * x + intercept
    ss_res = float(np.sum((y - yfit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    tau_y = intercept ** 2 if intercept > 0 else 0.0
    return tau_y, slope ** 2, r2, intercept


def bingham_fit(gamma_dot, tau):
    """Linear fit of tau vs gamma_dot: tau = tau_y + eta_p * gamma_dot.

    Returns (tau_y, eta_p, r2). Reported alongside the Casson fit
    because the toy's particle stress is roughly rate-independent at
    low rates (a constant offset), which the Bingham form captures
    directly: its intercept tracks the low-rate sigma_xy plateau,
    while the Casson intercept comes out several times smaller (the
    Casson excess over the viscous part grows as sqrt(gamma_dot),
    but the toy's excess collapses as the aggregates break).
    """
    x = np.asarray(gamma_dot, dtype=float)
    y = np.asarray(tau, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    yfit = slope * x + intercept
    ss_res = float(np.sum((y - yfit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return intercept, slope, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+",
                    default=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
                    help="shear rates to sweep (default: 0.05..2.0)")
    ap.add_argument("--eps-list", type=float, nargs="+",
                    default=[0.0, 0.32, 0.64, 1.28],
                    help="Morse attraction depths (default: 0 0.32 0.64 1.28)")
    ap.add_argument("--H", type=float, default=50.0)
    ap.add_argument("--L", type=float, default=100.0)
    ap.add_argument("--phi", type=float, default=0.25)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--equil-strain", type=float, default=10.0,
                    help="equilibration length in strain units")
    ap.add_argument("--meas-strain", type=float, default=10.0,
                    help="measurement window in strain units")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="flow-curve.csv")
    ap.add_argument("--plot", default=None,
                    help="optional PNG of the flow curve + Casson fits")
    args = ap.parse_args()

    a = 1.0
    n = int(args.phi * args.L * args.H / (np.pi * a * a))
    base = Params(N=n, a=a, H=args.H, L=args.L, eta=1.0, Umax=1.0,
                  kappa=args.kappa, eps_rep=1.0, kT=0.0, dt=args.dt,
                  seed=args.seed)
    print(f"channel H={base.H}, L={base.L}, a={a} (a/H={a / base.H:.3f}), "
          f"N={n}, area fraction phi={args.phi:.2f}, kappa={args.kappa}, "
          f"dt={args.dt}, equil={args.equil_strain} strains, "
          f"meas={args.meas_strain} strains")
    print(f"{'eps':>5} {'gamma':>7} {'Mason':>7} {'sigma_xy':>9} "
          f"{'tau':>8} {'tau_pl':>8} {'1st':>8} {'2nd':>8} {'clust':>6}")

    rows = []
    for eps in args.eps_list:
        gammas, taus = [], []
        for gamma_dot in args.rates:
            # Umax is unused by the Couette step, but setting it to
            # gamma_dot*H/4 makes p.gamma_w == gamma_dot, so
            # mason_number() reports the Couette Mason number.
            p = replace(base, eps=eps, Umax=gamma_dot * args.H / 4.0)
            heights = np.array([0.1, 0.3, 0.5, 0.7, 0.9]) * p.H
            res = run_point(p, gamma_dot,
                            equil_strain=args.equil_strain,
                            meas_strain=args.meas_strain,
                            sample_every=20, plane_heights=heights)
            cutoff = 2.0 * p.a + 0.5 / p.kappa
            c = cluster_stats(res["final_pos"], cutoff=cutoff)
            drift = abs(res["sigma_xy_first"] - res["sigma_xy_second"])
            # Stationarity judged against the TOTAL stress: a drift in
            # the particle stress only matters if it moves tau. (Judging
            # it against sigma_xy itself flags every near-zero point,
            # where the halves differ by noise on a ~0.005 scale.)
            flag = "  <-- NOT STATIONARY" if drift > 0.15 * res["tau"] else ""
            print(f"{eps:5.2f} {gamma_dot:7.3f} {res['mason']:7.1f} "
                  f"{res['sigma_xy']:9.4f} {res['tau']:8.4f} "
                  f"{res['tau_planes']:8.4f} {res['sigma_xy_first']:8.4f} "
                  f"{res['sigma_xy_second']:8.4f} {c['mean_cluster_size']:6.2f}"
                  f"{flag}")
            gammas.append(gamma_dot)
            taus.append(res["tau"])
            rows.append({
                "eps": eps, "gamma_dot": gamma_dot,
                "mason": res["mason"], "sigma_xy": res["sigma_xy"],
                "sigma_xy_std": res["sigma_xy_std"],
                "sigma_xy_first": res["sigma_xy_first"],
                "sigma_xy_second": res["sigma_xy_second"],
                "tau": res["tau"], "tau_planes": res["tau_planes"],
                "tau_planes_std": res["tau_planes_std"],
                "mean_cluster_size": c["mean_cluster_size"],
                "frac_in_clusters": c["frac_in_clusters"],
                "equil_steps": res["equil_steps"],
                "meas_steps": res["meas_steps"],
                "n_samples": res["n_samples"],
            })
        tau_y, eta_p, r2, intercept = casson_fit(gammas, taus)
        print(f"[eps={eps:g}] Casson fit: sqrt(tau) = {intercept:.4f} "
              f"+ {np.sqrt(eta_p):.4f}*sqrt(gamma_dot)  ->  "
              f"tau_y = {tau_y:.4f}, eta_p = {eta_p:.4f}, R^2 = {r2:.4f}")
        tau_y_b, eta_p_b, r2_b = bingham_fit(gammas, taus)
        print(f"[eps={eps:g}] Bingham fit: tau = {tau_y_b:.4f} "
              f"+ {eta_p_b:.4f}*gamma_dot  ->  "
              f"tau_y = {tau_y_b:.4f}, eta_p = {eta_p_b:.4f}, R^2 = {r2_b:.4f}")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for eps in args.eps_list:
            sub = [r for r in rows if r["eps"] == eps]
            g = np.array([r["gamma_dot"] for r in sub])
            t = np.array([r["tau"] for r in sub])
            axes[0].loglog(g, t, "o-", label=f"eps = {eps:g}")
            tau_y, eta_p, _, _ = casson_fit(g, t)
            g_fine = np.logspace(np.log10(g.min()), np.log10(g.max()), 50)
            axes[0].loglog(g_fine, (np.sqrt(tau_y)
                                    + np.sqrt(eta_p * g_fine)) ** 2,
                           "--", lw=1)
            tau_y_b, eta_p_b, _ = bingham_fit(g, t)
            axes[0].loglog(g_fine, tau_y_b + eta_p_b * g_fine,
                           ":", lw=1)
            axes[1].plot(np.sqrt(g), np.sqrt(t), "o-",
                         label=f"eps = {eps:g}")
            axes[1].plot(np.sqrt(g_fine),
                         np.sqrt(tau_y) + np.sqrt(eta_p * g_fine),
                         "--", lw=1)
        axes[0].set_xlabel("shear rate gamma_dot")
        axes[0].set_ylabel("shear stress tau")
        axes[0].set_title("Flow curve (dashed: Casson, dotted: Bingham)")
        axes[0].legend()
        axes[1].set_xlabel("sqrt(gamma_dot)")
        axes[1].set_ylabel("sqrt(tau)")
        axes[1].set_title("Casson linearisation")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
