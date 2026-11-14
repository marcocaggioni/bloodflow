"""Fit the TC model to the Couette flow curve (examples/flow-curve.csv).

TC model (Caggioni, Trappe & Spicer 2020, J. Rheol. 64(2) 413-422):
    sigma = sigma_y + sigma_y * sqrt(gamma_dot / gamma_dot_c)
            + eta_bg * gamma_dot

Fits minimise relative residuals, (model - data) / |data|, so every
decade of stress contributes equally (the low-rate points, where the
yield lives, are not drowned by the high-rate points). Bingham and
Casson are refit with the same objective so the three models can be
compared on equal footing.

What the data say (see the README): the fit drives gamma_dot_c to the
top of its range, i.e. the sqrt(gamma_dot) term switches off and TC
collapses onto Bingham. The yield it finds (sigma_y = 0.032 / 0.072
for eps = 0.64 / 1.28) is confirmed independently by the pressure-driven
plug width, plug half-width = tau_y / G (examples/pressure_flow.py).

Usage (from the repository root):
    python examples/tc_fit.py [--plot tc-fit-loglog.png]
Requires numpy and scipy (matplotlib only for --plot). Reads
examples/flow-curve.csv, the reference output of examples/flow_curve.py.
"""

import argparse
import csv
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "flow-curve.csv")

EPS_COLORS = {0.0: "tab:blue", 0.32: "tab:red", 0.64: "tab:purple",
              1.28: "tab:cyan"}


def tc_model(x, sigma_y, gamma_dot_c, eta_bg):
    return sigma_y + sigma_y * np.sqrt(x / gamma_dot_c) + eta_bg * x


def bingham_model(x, tau_y, eta_p):
    return tau_y + eta_p * x


def casson_model(x, tau_y, eta_p):
    return tau_y + 2.0 * np.sqrt(tau_y * eta_p * x) + eta_p * x


def fit_relative(model, x, y, bounds, n_starts=64, seed=0):
    """Multi-start least squares on relative residuals.

    Returns (params, redchi) with redchi = sum(r^2) / (n - n_params).
    """
    rng = np.random.default_rng(seed)
    lo, hi = np.array(bounds[0], dtype=float), np.array(bounds[1], dtype=float)

    def resid(p):
        return (model(x, *p) - y) / np.abs(y)

    starts = [tuple(np.exp(rng.uniform(np.log(lo), np.log(hi))))
              for _ in range(n_starts)]
    best = None
    for s in starts:
        try:
            sol = least_squares(resid, s, bounds=(lo, hi), method="trf",
                                ftol=1e-14, xtol=1e-14, gtol=1e-14)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    r = resid(best.x)
    redchi = float(np.sum(r ** 2) / max(len(y) - len(best.x), 1))
    return best.x, redchi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--plot", default=None,
                    help="optional PNG of the data with TC fits (log-log)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.data)))
    data = {}
    for r in rows:
        data.setdefault(float(r["eps"]), []).append(
            (float(r["gamma_dot"]), float(r["tau"])))
    fits = {}
    for eps in sorted(data):
        pts = np.array(sorted(data[eps]))
        x, y = pts[:, 0], pts[:, 1]
        p_tc, rc_tc = fit_relative(
            tc_model, x, y,
            ([1e-12, 1e-6, 1e-12], [10.0, 1e12, 10.0]))
        p_b, rc_b = fit_relative(
            bingham_model, x, y,
            ([1e-9, 1e-9], [10.0, 10.0]))
        p_c, rc_c = fit_relative(
            casson_model, x, y,
            ([1e-9, 1e-9], [10.0, 10.0]))
        print(f"eps={eps:g}: TC sigma_y={p_tc[0]:.4g} "
              f"gamma_dot_c={p_tc[1]:.4g} eta_bg={p_tc[2]:.4g} "
              f"redchi={rc_tc:.4g} | Bingham tau_y={p_b[0]:.4g} "
              f"redchi={rc_b:.4g} | Casson tau_y={p_c[0]:.4g} "
              f"redchi={rc_c:.4g}")
        fits[eps] = (x, y, p_tc)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.5, 6))
        xg = np.logspace(np.log10(0.04), np.log10(2.5), 200)
        for eps, (x, y, p) in sorted(fits.items()):
            c = EPS_COLORS.get(eps, "tab:gray")
            ax.plot(x, y, "o", color=c, label=f"eps = {eps:g} data")
            ax.plot(xg, tc_model(xg, *p), "-", color=c,
                    label=f"eps = {eps:g} TC: $\\sigma_y$={p[0]:.3g}, "
                          f"$\\dot\\gamma_c$={p[1]:.3g}, "
                          f"$\\eta_{{bg}}$={p[2]:.3g}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("shear rate $\\dot\\gamma$")
        ax.set_ylabel("shear stress $\\tau$")
        ax.set_title("Couette flow curves with TC model fits\n"
                     "$\\tau = \\sigma_y + \\sigma_y\\sqrt{\\dot\\gamma/"
                     "\\dot\\gamma_c} + \\eta_{bg}\\dot\\gamma$")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
