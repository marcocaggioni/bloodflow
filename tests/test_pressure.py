"""Validation tests for the pressure-driven flow mode.

Run: python3 tests/test_pressure.py   (from the work dir)
Covers: Stokes solve exactness, CIC force conservation, eps=0
parabola recovery, momentum-balance linearity, plug-width
prediction vs the Couette tau_y, and the below-yield no-flow case.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from bloodflow import pressure, core
from bloodflow.core import Params

BASE = dict(N=397, H=50.0, L=100.0, dt=0.02, seed=7)
TAU_Y = {0.0: 0.0, 0.64: 0.0315, 1.28: 0.072}  # Couette Bingham tau_y


def test_solve_stokes_parabola():
    H, eta, G, Ny = 50.0, 1.0, 0.004, 81
    y, u = pressure.solve_stokes(G, np.zeros(Ny), H, eta)
    exact = G * y * (H - y) / (2 * eta)
    err = np.max(np.abs(u - exact))
    assert err < 1e-10, f"parabola error {err}"
    assert abs(u.max() - G * H * H / (8 * eta)) < 1e-10
    # Constant drag density shifts the parabola by a known amount.
    fp = np.full(Ny, 0.002)
    y, u = pressure.solve_stokes(G, fp, H, eta)
    exact = (G + 0.002) * y * (H - y) / (2 * eta)
    assert np.max(np.abs(u - exact)) < 1e-10
    print("test_solve_stokes_parabola OK")


def test_spread_conserves_force():
    rng = np.random.default_rng(0)
    p = Params(N=50, H=50.0, L=100.0, eps=1.28, seed=0)
    pos = core.init_positions(p, rng)
    F = core.compute_forces(pos, p)
    fp = pressure.spread_forces_cic(pos, F[:, 0], p.L, p.H, 81)
    total = fp.sum() * (p.H / 80) * p.L
    assert abs(total - F[:, 0].sum()) < 1e-12, f"{total} vs {F[:, 0].sum()}"
    print("test_spread_conserves_force OK")


def test_eps0_parabola():
    p = Params(eps=0.0, **BASE)
    r = pressure.run_point(p, 0.004, equil_steps=5000, meas_steps=5000,
                           sample_every=20)
    y, v = r["y_vbins"], r["v_profile"]
    # Quadratic fit of the particle profile.
    c = np.polyfit(y, v, 2)
    pred = np.polyval(c, y)
    ss_res = float(np.sum((v - pred) ** 2))
    ss_tot = float(np.sum((v - np.mean(v)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.99, f"parabola R2 {r2}"
    # Particle and solvent centreline velocities agree.
    assert abs(r["Umax_particles"] - r["Umax_solvent"]) / r["Umax_solvent"] < 0.05
    # Effective viscosity: Couette gives eta ~ 1.04 at high rate and
    # ~1.16 at the lowest rate; the pressure parabola weights the
    # low-shear centre, so eta_eff must sit in that range.
    assert 1.0 < r["eta_eff"] < 1.30, f"eta_eff {r['eta_eff']}"
    print(f"test_eps0_parabola OK (R2 {r2:.4f}, eta_eff {r['eta_eff']:.3f}, "
          f"Umax {r['Umax_particles']:.3f})")


def test_stress_linearity():
    p = Params(eps=1.28, **BASE)
    r = pressure.run_point(p, 0.004, equil_steps=10000, meas_steps=10000,
                           sample_every=20, tau_y_pred=TAU_Y[1.28])
    f = r["stress_fit"]
    assert abs(f["slope_over_G"] + 1.0) < 0.10, f"slope/G {f['slope_over_G']}"
    assert f["r2"] > 0.98, f"stress R2 {f['r2']}"
    assert abs(f["intercept"] - f["intercept_expected"]) / f["intercept_expected"] < 0.15
    print(f"test_stress_linearity OK (slope/G {f['slope_over_G']:.3f}, "
          f"R2 {f['r2']:.4f})")


def test_plug_prediction():
    p = Params(eps=1.28, **BASE)
    r = pressure.run_point(p, 0.004, equil_steps=10000, meas_steps=10000,
                           sample_every=20, tau_y_pred=TAU_Y[1.28])
    pred = r["plug_half_width_pred"]
    # Stress-based width is the robust one (linear stress profile +
    # Couette tau_y); profile-based widths are threshold-sensitive
    # and scatter ~30-50% run to run (see diag_plug_sharpness.py).
    assert abs(r["plug_half_width_stress"] - pred) / pred < 0.25, (
        f"stress width {r['plug_half_width_stress']} vs {pred}")
    assert abs(r["plug_half_width_10"] - pred) / pred < 0.40, (
        f"hw10 {r['plug_half_width_10']} vs {pred}")
    assert abs(r["plug_half_width_05"] - pred) / pred < 0.50, (
        f"hw5 {r['plug_half_width_05']} vs {pred}")
    print(f"test_plug_prediction OK (stress {r['plug_half_width_stress']:.1f}, "
          f"hw10 {r['plug_half_width_10']:.1f}, "
          f"hw5 {r['plug_half_width_05']:.1f} vs predicted {pred:.1f}")


def test_no_flow():
    p = Params(eps=1.28, **BASE)
    r = pressure.run_point(p, 0.001, equil_steps=10000, meas_steps=10000,
                           sample_every=20, tau_y_pred=TAU_Y[1.28])
    # tau_wall = 0.025 < tau_y = 0.072: the stress criterion says the
    # plug spans the whole channel.
    assert abs(r["plug_half_width_stress"] - p.H / 2.0) / (p.H / 2.0) < 0.10, (
        f"stress plug half-width {r['plug_half_width_stress']} vs H/2 {p.H / 2.0}")
    # The profile creeps (frictionless walls cannot hold a static
    # plug) but is strongly flattened vs the Newtonian parabola at
    # the same G (Umax ~ G*H^2/(8*1.17) = 0.27; wall bins carry
    # >40% of the centreline velocity, vs 0% for a parabola).
    v = r["v_profile"]
    v_centre = float(np.max(v))
    u_newt = 0.001 * p.H ** 2 / (8.0 * 1.17)
    assert v_centre < 0.45 * u_newt, (
        f"creep velocity {v_centre:.3f} too large vs Newtonian {u_newt:.3f}")
    assert v[0] / v_centre > 0.4 and v[-1] / v_centre > 0.4, (
        f"profile not flattened: {np.round(v, 3)}")
    print(f"test_no_flow OK (stress plug hw {r['plug_half_width_stress']:.1f}, "
          f"creep v {v_centre:.3f} vs Newtonian {u_newt:.3f})")


def test_core_unchanged():
    import hashlib
    for name in ("core.py", "potentials.py", "analysis.py", "__init__.py"):
        a = hashlib.md5(open(f"bloodflow/{name}", "rb").read()).hexdigest()
        b = hashlib.md5(
            open(f"/home/hatch/workspace/bloodflow-couette/bloodflow/{name}",
                 "rb").read()).hexdigest()
        assert a == b, f"{name} differs from the Couette-verified copy"
    print("test_core_unchanged OK")


if __name__ == "__main__":
    test_solve_stokes_parabola()
    test_spread_conserves_force()
    test_eps0_parabola()
    test_stress_linearity()
    test_plug_prediction()
    test_no_flow()
    test_core_unchanged()
    print("ALL PRESSURE TESTS PASSED")
