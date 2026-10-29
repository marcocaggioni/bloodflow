# Validation tests for Bloodflow. Runnable two ways:
#   python tests/test_poiseuille.py   (plain python, no pytest needed)
#   python -m pytest tests/           (pytest-compatible)
#
# Test 1: with attraction off (eps = 0) and no noise, every particle is
# advected at exactly the Poiseuille velocity of its height, so its
# transit time across length L must equal L / U(y).
# Test 2: the cell-list forces must match a direct O(N^2) computation.

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bloodflow import Params, compute_forces, poiseuille_velocity, simulate
from bloodflow import potentials


def test_poiseuille_transit():
    p = Params(N=6, a=1.0, H=16.0, L=30.0, Umax=1.0,
               eps=0.0, kT=0.0, dt=0.01, seed=0)
    # Heights spaced wider than the pair cutoff, so no forces act at all
    # (and no wall forces: all heights are clear of the wall zone).
    # Note: particles at different heights eventually align in x (they
    # move at different speeds), so the *height gap* itself must exceed
    # the cutoff, or the WCA repulsion would nudge them to new heights.
    ys = np.array([2.0, 4.5, 7.0, 9.5, 12.0, 14.5])
    pos0 = np.column_stack([np.linspace(0.0, 25.0, 6), ys])
    res = simulate(p, n_steps=20000, equil_steps=0, pos0=pos0)
    analytic = p.L / poiseuille_velocity(ys, p)
    assert np.all(~np.isnan(res.transit)), "all particles must complete transit"
    err = np.abs(res.transit - analytic)
    tol = np.maximum(2.0 * p.dt, 0.01 * analytic)
    assert np.all(err <= tol), f"transit mismatch: {res.transit} vs {analytic}"
    print(f"poiseuille transit OK: max err {err.max():.4f} "
          f"(tol {tol.max():.4f})")


def _brute_forces(pos, p):
    N = pos.shape[0]
    F = np.zeros((N, 2))
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dx -= p.L * np.round(dx / p.L)
            dy = pos[i, 1] - pos[j, 1]
            r = np.hypot(dx, dy)
            if r <= 0.0:
                continue
            s = 0.0
            if r < p.wca_cut:
                s += potentials.wca_force_scalar(r, p.wca_sigma, p.eps_rep)
            if p.eps != 0.0 and r < p.morse_cut:
                s += potentials.morse_force_scalar(r, p.a, p.eps, p.kappa)
            fx, fy = s * dx / r, s * dy / r
            F[i] += (fx, fy)
            F[j] -= (fx, fy)
    for d, sign in ((pos[:, 1], 1.0), (p.H - pos[:, 1], -1.0)):
        mask = (d < potentials._WCA_C * p.a) & (d > 0.0)
        if mask.any():
            F[mask, 1] += sign * potentials.wall_force_scalar(
                d[mask], p.a, p.eps_rep)
    return F


def test_forces_vs_bruteforce():
    # Non-overlapping positions (as in a real run), so forces are
    # moderate and an absolute tolerance is meaningful. Random
    # overlapping positions would give astronomically large WCA forces
    # and make any tolerance vacuous.
    rng = np.random.default_rng(42)
    p = Params(N=60, a=1.0, H=16.0, L=50.0, eps=2.0, kappa=2.0, seed=0)
    from bloodflow.core import init_positions
    pos = init_positions(p, rng)
    f_cell = compute_forces(pos, p)
    f_brute = _brute_forces(pos, p)
    err = np.max(np.abs(f_cell - f_brute))
    assert err < 1e-8, f"force mismatch: max err {err}"
    print(f"cell-list forces OK: max err {err:.3e}")


if __name__ == "__main__":
    test_poiseuille_transit()
    test_forces_vs_bruteforce()
    print("all tests passed")
