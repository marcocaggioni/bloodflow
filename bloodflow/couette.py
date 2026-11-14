# Planar Couette shear rheometry for the Bloodflow toy model.
#
# Why this module exists: in the Poiseuille setup the velocity profile
# is IMPOSED, so the material never gets to decide whether to flow —
# the defining feature of a yield-stress fluid (no flow below the yield
# stress) cannot appear, and there is no stress on the other side of
# the equation to measure a flow curve against. Couette shear fixes
# that: the shear rate is imposed (a linear velocity field) and the
# shear stress is a genuine response of the particles, measured two
# independent ways (below). Plotting that stress against the imposed
# rate gives tau(gamma_dot), and the Casson test is whether
# sqrt(tau) is linear in sqrt(gamma_dot) with a finite intercept.
#
# Setup: planar Couette, U(y) = gamma_dot * (y - H/2), zero at the
# centre so there is no net drift, periodic in x, the same WCA walls
# as the Poiseuille model (the advection at the walls is +/-gamma_dot
# * H/2, i.e. no-slip: the fluid at the wall moves with the wall).
# The integrator is the same overdamped step as core._step, only the
# advection field changes; core.py itself is untouched.
#
# Stress, two ways (they must agree in steady state):
#
# 1. Virial (volume average). The particle-phase shear stress from the
#    pair forces, Cauchy sign (positive for gamma_dot > 0):
#        sigma_xy = -(1/A) * sum_{i<j} r_ij,x * F_ij,y
#    with r_ij = r_i - r_j and F_ij the force on i due to j. The minus
#    sign is the standard virial-theorem sign: the virial sum gives the
#    pressure tensor, and the Cauchy stress is its negative. (The
#    kinetic term sum m v v vanishes: in overdamped dynamics there is
#    no inertia and no momentum-carrying velocity.) The wall forces
#    are NOT included: they are purely normal, and their virial
#    contribution averages to zero by symmetry (uniform x distribution,
#    zero net wall force on average).
#
# 2. Method of planes (boundary measurement). The shear force
#    transmitted across a horizontal plane at height y0, per unit
#    length: tau(y0) = (1/L) * sum over pairs straddling y0 of the
#    x-force the upper particle exerts on the lower one. The WCA walls
#    are frictionless in x (they exert only normal forces), so the
#    boundary shear stress cannot be read off a wall force directly;
#    the force transmitted across a plane adjacent to the wall IS the
#    shear stress the wall must support. In steady state tau(y0) is
#    independent of y0 and equals the volume virial — that agreement
#    is the validation check.
#
# Total stress: the background (solvent) fluid carries eta*gamma_dot,
# so the stress a rheometer would report is
#     tau = eta * gamma_dot + sigma_xy   (particle virial).
# For eps = 0 the particle part is the WCA-collision viscosity and tau
# stays proportional to gamma_dot through the origin (Newtonian). With
# attraction the aggregates add a rate-independent part: a finite
# tau as gamma_dot -> 0, i.e. a yield stress.
#
# Caveats (read before believing a flow curve from this toy):
# - The shear is still imposed homogeneously: the profile cannot
#   localise into a shear band, so a banded state would show up as a
#   uniform shear with an averaged stress, not as a band.
# - At low rates with strong attraction the aggregates are long-lived
#   and the structure may age instead of reaching a steady state;
#   run_point() reports the first/second-half means so a drifting
#   point can be recognised and not trusted.
# - The stress is the particle virial only; the solvent part eta*gamma_dot
#   is added by the caller (see flow_curve.py).

import numpy as np

from . import core
from .core import Params
from . import potentials


def couette_velocity(y, p, gamma_dot):
    """Imposed Couette profile U(y) = gamma_dot * (y - H/2).

    Zero at the channel centre (no net drift); +/-gamma_dot*H/2 at the
    walls, matching the wall velocities for no-slip.
    """
    return gamma_dot * (y - 0.5 * p.H)


def _step_couette(pos, p, rng, noise_std, gamma_dot):
    """One overdamped step under Couette shear.

    Identical to core._step except the advection is the linear Couette
    field instead of the Poiseuille parabola. Returns the x-displacement
    (as core._step does).
    """
    F = core.compute_forces(pos, p)
    adv = couette_velocity(pos[:, 1], p, gamma_dot)
    dx = (adv + F[:, 0] / p.zeta) * p.dt
    dy = (F[:, 1] / p.zeta) * p.dt
    if noise_std > 0.0:
        dx = dx + noise_std * rng.standard_normal(pos.shape[0])
        dy = dy + noise_std * rng.standard_normal(pos.shape[0])
    pos[:, 0] = (pos[:, 0] + dx) % p.L
    pos[:, 1] = pos[:, 1] + dy
    return dx


def pair_forces_and_stress(pos, p, plane_heights=None):
    """Pair forces, virial shear stress and method-of-planes stress.

    Vectorised O(N^2) over pairs (fine at the N used here; the dynamics
    themselves still use the cell-list forces in core). The pair forces
    use exactly the same WCA + Morse scalars as core.compute_pair_forces
    (via potentials.wca_force_scalar / morse_force_scalar), so the
    returned per-particle forces must match core.compute_forces up to
    the wall forces — see the validation in test_couette.py.

    pos: (N, 2) positions. p: Params. plane_heights: heights y0 at
    which to evaluate the method-of-planes stress (None to skip it).

    Returns a dict with:
      "F": (N, 2) pair forces (no wall forces),
      "sigma_xy": particle virial shear stress (Cauchy sign),
      "tau_planes": array over plane_heights, or None.
    """
    N = pos.shape[0]
    A = p.L * p.H
    dx = pos[:, 0][:, None] - pos[:, 0][None, :]
    dx -= p.L * np.round(dx / p.L)  # minimum image in x
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    r2 = dx * dx + dy * dy
    iu = np.triu_indices(N, k=1)
    dxi = dx[iu]
    dyi = dy[iu]
    r = np.sqrt(r2[iu])
    valid = (r < p.cell) & (r > 0.0)
    s = np.zeros_like(r)
    wca = valid & (r < p.wca_cut)
    if wca.any():
        s[wca] = potentials.wca_force_scalar(r[wca], p.wca_sigma, p.eps_rep)
    if p.eps != 0.0:
        morse = valid & (r < p.morse_cut)
        if morse.any():
            s[morse] += potentials.morse_force_scalar(
                r[morse], p.a, p.eps, p.kappa)
    safe_r = np.where(valid, r, 1.0)
    fx = np.where(valid, s * dxi / safe_r, 0.0)
    fy = np.where(valid, s * dyi / safe_r, 0.0)
    # Virial (Cauchy sign): sigma_xy = -(1/A) sum_{i<j} dx * F_y.
    sigma_xy = -float(np.sum(dxi * fy)) / A
    F = np.zeros((N, 2))
    np.add.at(F[:, 0], iu[0], fx)
    np.add.at(F[:, 0], iu[1], -fx)
    np.add.at(F[:, 1], iu[0], fy)
    np.add.at(F[:, 1], iu[1], -fy)
    tau_planes = None
    if plane_heights is not None:
        y_i = pos[iu[0], 1]
        y_j = pos[iu[1], 1]
        y_lo = np.minimum(y_i, y_j)
        y_hi = np.maximum(y_i, y_j)
        x_lo = np.where(y_i < y_j, pos[iu[0], 0], pos[iu[1], 0])
        x_hi = np.where(y_i < y_j, pos[iu[1], 0], pos[iu[0], 0])
        dx_lu = x_lo - x_hi
        dx_lu -= p.L * np.round(dx_lu / p.L)  # minimum image in x
        # x-force on the lower particle due to the upper one.
        f_x = np.where(valid, s * dx_lu / safe_r, 0.0)
        tau_planes = np.array([
            float(np.sum(f_x[valid & (y_lo < y0) & (y_hi > y0)])) / p.L
            for y0 in plane_heights
        ])
    return {"F": F, "sigma_xy": sigma_xy, "tau_planes": tau_planes}


def run_point(p, gamma_dot, equil_strain=10.0, meas_strain=10.0,
              sample_every=20, seed=None, plane_heights=None,
              min_steps=2000, max_steps=60000):
    """Run one (eps, gamma_dot) flow-curve point.

    The system is sheared for equil_strain strain units (equilibration,
    aggregates form and the structure settles), then the stress is
    sampled every sample_every steps for meas_strain strain units.
    Step counts are clipped to [min_steps, max_steps] so high-rate
    points are not absurdly short and low-rate points stay affordable.
    (Strain is the right clock here: particle meetings are driven by
    the shear, so relaxation times scale as 1/gamma_dot.)

    p: Params (Umax is unused by the Couette step; set it to
    gamma_dot*H/4 so that p.gamma_w == gamma_dot and mason_number()
    reports the Couette Mason number). gamma_dot: imposed shear rate.
    seed: RNG seed (defaults to p.seed).

    Returns a dict with the mean particle virial stress, the total
    stress tau = eta*gamma_dot + sigma_xy, the method-of-planes stress
    (mean over heights and over the window), first/second-half means
    of the virial samples (stationarity check), the sample scatter,
    the final positions, and the step counts used.
    """
    rng = np.random.default_rng(p.seed if seed is None else seed)
    pos = core.init_positions(p, rng)
    noise_std = 0.0
    if p.kT > 0.0:
        D = p.kT / p.zeta
        noise_std = np.sqrt(2.0 * D * p.dt)
    dt = p.dt
    equil_steps = int(np.clip(equil_strain / gamma_dot / dt,
                              min_steps, max_steps))
    meas_steps = int(np.clip(meas_strain / gamma_dot / dt,
                             min_steps, max_steps))
    for _ in range(equil_steps):
        _step_couette(pos, p, rng, noise_std, gamma_dot)
    samples = []
    planes = []
    for step in range(1, meas_steps + 1):
        _step_couette(pos, p, rng, noise_std, gamma_dot)
        if step % sample_every == 0:
            out = pair_forces_and_stress(pos, p, plane_heights)
            samples.append(out["sigma_xy"])
            if out["tau_planes"] is not None:
                planes.append(out["tau_planes"])
    samples = np.array(samples)
    half = len(samples) // 2
    first = samples[:half]
    second = samples[half:]
    planes_arr = np.array(planes) if planes else None
    tau_total = float(p.eta * gamma_dot + np.mean(samples))
    return {
        "gamma_dot": gamma_dot,
        "eps": p.eps,
        "mason": p.mason_number(),
        "sigma_xy": float(np.mean(samples)),
        "sigma_xy_std": float(np.std(samples)),
        "sigma_xy_first": float(np.mean(first)) if first.size else float("nan"),
        "sigma_xy_second": float(np.mean(second)) if second.size else float("nan"),
        "tau": tau_total,
        "tau_planes": (
            float(np.mean(planes_arr)) if planes_arr is not None
            else float("nan")),
        "tau_planes_std": (
            float(np.std(np.mean(planes_arr, axis=0)))
            if planes_arr is not None else float("nan")),
        # Time-averaged method-of-planes stress at each requested
        # height (in the order plane_heights was given).
        "tau_planes_by_height": (
            np.mean(planes_arr, axis=0) if planes_arr is not None
            else None),
        "n_samples": int(len(samples)),
        "equil_steps": equil_steps,
        "meas_steps": meas_steps,
        "final_pos": pos.copy(),
    }
