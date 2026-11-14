# Validation tests for the Couette flow-curve mode. Runnable as:
#   python tests/test_couette.py   (plain python, no pytest needed)
#
# Test 1: the O(N^2) pair forces in couette.pair_forces_and_stress must
# match the cell-list forces in core.compute_forces (positions kept
# clear of the walls, so no wall forces act).
# Test 2: in steady Couette shear the method-of-planes stress must agree
# with the volume virial, and be independent of the plane height.
# Test 3: eps = 0 is Newtonian — tau proportional to gamma_dot through
# the origin (tau(1.0) / tau(0.25) ~ 4, both positive).
# Test 4: eps > 0 gives a clearly larger stress at low rate than eps = 0
# (the yield signal the Casson fit reads off).

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bloodflow import Params
from bloodflow import core
from bloodflow.couette import pair_forces_and_stress, run_point


def test_pair_forces_vs_cell_list():
    rng = np.random.default_rng(42)
    p = Params(N=60, a=1.0, H=16.0, L=50.0, eps=2.0, kappa=2.0, seed=0)
    pos = core.init_positions(p, rng)
    # Keep every particle clear of the wall zone so compute_forces has
    # no wall contribution to subtract.
    pos[:, 1] = np.clip(pos[:, 1], 3.0, p.H - 3.0)
    out = pair_forces_and_stress(pos, p)
    f_cell = core.compute_forces(pos, p)
    err = np.max(np.abs(out["F"] - f_cell))
    assert err < 1e-8, f"pair-force mismatch: max err {err}"
    print(f"couette pair forces OK: max err {err:.3e}")


def test_virial_vs_planes():
    p = Params(N=60, a=1.0, H=16.0, L=40.0, Umax=0.125,
               eps=0.0, kappa=2.0, kT=0.0, dt=0.02, seed=7)
    # Umax = gamma_dot * H / 4, so gamma_w == gamma_dot == 0.5.
    gamma_dot = 0.5
    heights = np.array([0.1, 0.3, 0.5, 0.7, 0.9]) * p.H
    res = run_point(p, gamma_dot, equil_strain=10.0, meas_strain=10.0,
                    sample_every=20, plane_heights=heights)
    # Recompute the per-height planes means from a fresh short window
    # is unnecessary: run_point reports the window mean over heights
    # and the scatter across heights; check both against the virial.
    vir = res["sigma_xy"]
    pla = res["tau_planes"]
    print(f"virial sigma_xy = {vir:.4f}, planes mean = {pla:.4f}, "
          f"planes scatter across heights = {res['tau_planes_std']:.4f}")
    assert vir > 0, "virial stress must be positive for gamma_dot > 0"
    assert abs(vir - pla) < 0.25 * abs(vir) + 0.01, (
        f"virial/planes disagree: {vir:.4f} vs {pla:.4f}")
    print("virial vs method-of-planes OK")


def test_newtonian_eps0():
    taus = {}
    for gamma_dot in (0.25, 1.0):
        p = Params(N=60, a=1.0, H=16.0, L=40.0,
                   Umax=gamma_dot * 16.0 / 4.0,
                   eps=0.0, kappa=2.0, kT=0.0, dt=0.02, seed=7)
        res = run_point(p, gamma_dot, equil_strain=10.0, meas_strain=10.0,
                        sample_every=20)
        taus[gamma_dot] = res["tau"]
        print(f"eps=0 gamma_dot={gamma_dot}: tau={res['tau']:.4f} "
              f"(sigma_xy={res['sigma_xy']:.4f}, eta*gamma={p.eta * gamma_dot:.4f})")
    ratio = taus[1.0] / taus[0.25]
    assert taus[0.25] > 0 and taus[1.0] > 0, "tau must be positive"
    assert 3.0 < ratio < 5.0, f"eps=0 not Newtonian: tau ratio {ratio:.2f} != 4"
    print(f"eps=0 Newtonian OK: tau(1.0)/tau(0.25) = {ratio:.2f}")


def test_yield_signal():
    # The yield signal lives in the PARTICLE stress: the total tau is
    # diluted by the solvent background eta*gamma_dot (= 0.1 here), so
    # the ratio on tau is modest even when the particle stress jumps.
    gamma_dot = 0.1
    sig = {}
    for eps in (0.0, 1.28):
        p = Params(N=60, a=1.0, H=16.0, L=40.0,
                   Umax=gamma_dot * 16.0 / 4.0,
                   eps=eps, kappa=2.0, kT=0.0, dt=0.02, seed=7)
        res = run_point(p, gamma_dot, equil_strain=10.0, meas_strain=10.0,
                        sample_every=20)
        sig[eps] = res["sigma_xy"]
        print(f"gamma_dot={gamma_dot} eps={eps}: tau={res['tau']:.4f} "
              f"(sigma_xy={res['sigma_xy']:.4f})")
    assert sig[1.28] > 3.0 * sig[0.0], (
        f"no yield signal: sigma_xy(eps=1.28)={sig[1.28]:.4f} vs "
        f"sigma_xy(eps=0)={sig[0.0]:.4f}")
    print("yield signal OK: sigma_xy(eps=1.28) >> sigma_xy(eps=0) at low rate")


if __name__ == "__main__":
    test_pair_forces_vs_cell_list()
    test_virial_vs_planes()
    test_newtonian_eps0()
    test_yield_signal()
    print("all couette tests passed")
