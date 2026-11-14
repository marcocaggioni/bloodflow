# Pressure-driven flow for the Bloodflow toy model.
#
# Why this module exists: until now the flow profile was IMPOSED
# (Poiseuille parabola, then Couette linear shear), so the material
# never got to decide its own velocity profile. Here a pressure
# gradient G drives the flow and the profile EMERGES from the material
# response: a yield-stress plug forms in the centre where the local
# stress is below the yield stress, and sheared layers form at the
# walls. The Couette flow curve then predicts the plug: its
# half-width is tau_y / G.
#
# Setup: each step, solve the 1D Stokes equation for the solvent
# velocity u(y) on a node grid (Ny nodes across H, no-slip at the
# walls):
#     eta * u''(y) + G + f_p(y) = 0.
# G > 0 is the imposed pressure-gradient force density driving flow
# in +x (pure solvent: u = G*y*(H-y)/(2*eta), Umax = G*H^2/(8*eta)).
# f_p(y) is the two-way coupling: the x-force the particles exert on
# the fluid, spread to the grid (cloud-in-cell) as a force density.
# Particles are advected by the SOLVED u(y), interpolated to each
# particle's height, exactly as core._step advects by the imposed
# profile — same integrator, only the velocity field comes from the
# solve instead of a formula. No body force acts on the particles
# themselves (the pressure acts on the fluid; neutrally buoyant
# particles are carried). core.py itself is untouched.
#
# Coupling sign (derived, then checked by the validation tests):
# the particle update is v_i = u(y_i) + F_i/zeta (+ noise), so the
# drag force the fluid exerts on particle i is zeta*(u - v_i) = -F_i,
# and by Newton the particle exerts +F_i on the fluid. Hence
#     f_p(y) = (1 / (L*dy)) * sum of F_{x,i} over particles near y,
# with F the total non-drag x-force from core.compute_forces (pair
# forces; the wall forces are purely normal and contribute nothing
# in x). Where particles resist the flow (F_x < 0) the fluid is
# slowed and the profile flattens — that is the plug.
#
# Momentum balance (the check that matters): the total stress
#     tau(y) = eta*u'(y) + sigma_xy^p(y)
# (solvent viscous stress + particle-phase stress from the
# method-of-planes measurement) obeys d(tau)/dy = -G, i.e.
#     tau(y) = G*(H/2 - y),
# linear in y, zero at the centre, +/-G*H/2 at the walls. The
# particle-only stress is NOT linear in general (it is linear only
# where u' = 0, i.e. inside the plug); the linearity check is on the
# total. A corollary: the net particle force sums to zero on average
# (integral of f_p over the channel ~ 0) — the particles transmit
# force but neither create nor destroy it.
#
# Caveats (read before believing a profile from this toy):
# - The walls are frictionless in x (WCA normal forces only), so they
#   cannot hold a static plug: below the yield stress everywhere the
#   plug still creeps, dragged by the fluid, at a small velocity set
#   by the no-slip solvent in the particle-free layer next to the
#   wall. The observable for "no flow" is therefore a FLAT profile
#   (no shear anywhere), not exactly zero velocity.
# - The coupling is explicit (one Stokes solve per step from the
#   current forces) and f_p is instantaneously noisy; the solve is
#   linear in f_p, so the time-averaged u(y) is unbiased, but the
#   instantaneous profile wiggles. All reported profiles are
#   time-averaged over the measurement window.
# - The plug has no sharp boundary in this toy: the profile is
#   rounded, not flat-with-a-kink, because the toy's flow curve is
#   smooth and strongly shear-thinning (the Couette stress is
#   roughly rate-independent at low rates but never truly
#   rate-independent, so there is always some creep). Any
#   velocity-based plug width is therefore threshold-dependent; the
#   stress-based width (where |tau| crosses the Couette tau_y on
#   the measured linear stress profile) is the robust one, and the
#   tau_y/G prediction is an upper bound on the flat region, not
#   an exact match for it. The threshold used is absolute (a
#   fraction of the pure-solvent wall shear rate at the same G),
#   stated per point.

import numpy as np

from . import core
from .core import Params
from . import couette
from . import analysis


def solve_stokes(G, fp, H, eta):
    """Solve eta*u'' + G + f_p = 0 on a node grid, no-slip u(0)=u(H)=0.

    fp: particle drag force density at the Ny nodes. Returns
    (y_nodes, u). Pure solvent (fp = 0) gives the parabola
    u = G*y*(H-y)/(2*eta) exactly (up to the grid solve).
    """
    Ny = fp.size
    dy = H / (Ny - 1)
    y = np.linspace(0.0, H, Ny)
    u = np.zeros(Ny)
    if Ny <= 2:
        return y, u
    # Interior: -u_{j-1} + 2*u_j - u_{j+1} = dy^2*(G + fp_j)/eta.
    rhs = (dy * dy / eta) * (G + fp[1:-1])
    n = Ny - 2
    # Thomas algorithm for tridiagonal (-1, 2, -1).
    c = np.zeros(n)
    d = rhs.copy()
    c[0] = -1.0 / 2.0
    d[0] = rhs[0] / 2.0
    for i in range(1, n):
        denom = 2.0 + c[i - 1]
        c[i] = -1.0 / denom if i < n - 1 else 0.0
        d[i] = (rhs[i] + d[i - 1]) / denom
    u_int = np.zeros(n)
    u_int[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        u_int[i] = d[i] - c[i] * u_int[i + 1]
    u[1:-1] = u_int
    return y, u


def spread_forces_cic(pos, Fx, L, H, Ny):
    """Cloud-in-cell spread of particle x-forces to a force density.

    Each particle's F_x is split linearly between its two neighbouring
    nodes; the density is force per area (node area L*dy). Returns
    f_p at the Ny nodes. (Adjoint-consistent with the linear
    interpolation used to advect the particles.)
    """
    dy = H / (Ny - 1)
    s = np.clip(pos[:, 1] / dy, 0.0, Ny - 1.0 - 1e-9)
    j0 = np.floor(s).astype(int)
    w = s - j0
    fp = np.zeros(Ny)
    np.add.at(fp, j0, (1.0 - w) * Fx)
    np.add.at(fp, j0 + 1, w * Fx)
    fp /= (L * dy)
    return fp


def _step_pressure(pos, p, rng, noise_std, G, Ny):
    """One overdamped step under pressure-driven flow.

    Identical to core._step except the advection is the Stokes-solved
    solvent velocity instead of the imposed Poiseuille parabola.
    Returns (dx, y_nodes, u_nodes, fp) — the solved profile and the
    particle drag density are returned so the caller can time-average
    them.
    """
    F = core.compute_forces(pos, p)
    fp = spread_forces_cic(pos, F[:, 0], p.L, p.H, Ny)
    y_nodes, u = solve_stokes(G, fp, p.H, p.eta)
    adv = np.interp(pos[:, 1], y_nodes, u)
    dx = (adv + F[:, 0] / p.zeta) * p.dt
    dy = (F[:, 1] / p.zeta) * p.dt
    if noise_std > 0.0:
        dx = dx + noise_std * rng.standard_normal(pos.shape[0])
        dy = dy + noise_std * rng.standard_normal(pos.shape[0])
    pos[:, 0] = (pos[:, 0] + dx) % p.L
    pos[:, 1] = pos[:, 1] + dy
    return dx, y_nodes, u, fp


def _smooth3(v):
    """One pass of a 3-point binomial smooth (numpy-only)."""
    out = v.copy()
    out[1:-1] = 0.25 * v[:-2] + 0.5 * v[1:-1] + 0.25 * v[2:]
    return out


def _plug_half_width(y, v, thresh):
    """Half-width of the central region where |dv/dy| < thresh.

    thresh is an ABSOLUTE shear-rate threshold (a relative threshold
    breaks down when the whole profile is flat: the max gradient is
    then itself tiny). The profile is binomial-smoothed first: the
    binned gradient is noisy and the contiguous-region test would
    otherwise break on noise spikes. Returns (half_width,
    threshold_used). If the whole profile is below threshold the plug
    spans the channel (half-width H/2).
    """
    vs = _smooth3(np.asarray(v, dtype=float))
    dvdy = np.gradient(vs, y)
    below = np.abs(dvdy) < thresh
    if np.all(below):
        return 0.5 * (y[-1] - y[0]), thresh
    mid = len(y) // 2
    if not below[mid]:
        # Centre is sheared: no plug by this measure.
        return 0.0, thresh
    lo = mid
    while lo > 0 and below[lo - 1]:
        lo -= 1
    hi = mid
    while hi < len(y) - 1 and below[hi + 1]:
        hi += 1
    # Region edges: halfway between the last below-threshold bin and
    # the first above-threshold bin on each side.
    y_lo = y[lo] - (0.5 * (y[1] - y[0]) if lo > 0 else 0.0)
    y_hi = y[hi] + (0.5 * (y[1] - y[0]) if hi < len(y) - 1 else 0.0)
    return 0.5 * (y_hi - y_lo), thresh


def _plug_half_width_stress(y, tau, tau_y, half_channel=None):
    """Half-width of the central region where |tau| < tau_y.

    Crossings found by linear interpolation, scanning outward from
    the centre on each side (first crossing wins). If |tau| < tau_y
    everywhere the plug spans the channel (half_channel, or half the
    sampled range if not given).
    """
    a = np.abs(np.asarray(tau))
    if np.all(a < tau_y):
        return half_channel if half_channel is not None else 0.5 * (y[-1] - y[0])
    mid = len(y) // 2
    y1 = None
    for i in range(mid, -1, -1):
        if a[i] >= tau_y:
            if i + 1 < len(y) and a[i] != a[i + 1]:
                f = (tau_y - a[i + 1]) / (a[i] - a[i + 1])
                y1 = y[i + 1] + f * (y[i] - y[i + 1])
            else:
                y1 = y[i]
            break
    y2 = None
    for i in range(mid, len(y)):
        if a[i] >= tau_y:
            if i - 1 >= 0 and a[i] != a[i - 1]:
                f = (tau_y - a[i - 1]) / (a[i] - a[i - 1])
                y2 = y[i - 1] + f * (y[i] - y[i - 1])
            else:
                y2 = y[i]
            break
    if y1 is None or y2 is None:
        return float("nan")
    return 0.5 * (y2 - y1)


def run_point(p, G, equil_steps=20000, meas_steps=20000, Ny=81,
              n_vbins=25, sample_every=20, plane_heights=None,
              seed=None, tau_y_pred=None):
    """Run one (eps, G) pressure-driven point.

    Equilibrate with the pressure on, then measure for meas_steps:
    time-averaged particle velocity profile (Eulerian bins of the
    actual per-step displacements), time-averaged solvent profile
    u(y), method-of-planes stress at plane_heights, virial samples,
    and transit times (unwrapped x-displacement reaching L).

    p: Params (Umax is unused by the pressure step). G: imposed
    pressure-gradient force density. tau_y_pred: Couette-measured
    yield stress, for the stress-based plug width (optional).

    Returns a dict with the profiles, the plug widths (profile-based
    at two thresholds and stress-based), the total-stress linearity
    fit (slope vs -G), transit summary, stationarity halves, and the
    step counts.
    """
    rng = np.random.default_rng(p.seed if seed is None else seed)
    pos = core.init_positions(p, rng)
    noise_std = 0.0
    if p.kT > 0.0:
        D = p.kT / p.zeta
        noise_std = np.sqrt(2.0 * D * p.dt)
    if plane_heights is None:
        # Close to the walls (0.04H) so the |tau| = tau_y crossing is
        # inside the sampled range even for wide plugs.
        plane_heights = np.linspace(0.04, 0.96, 12) * p.H
    plane_heights = np.asarray(plane_heights, dtype=float)

    for _ in range(equil_steps):
        _step_pressure(pos, p, rng, noise_std, G, Ny)

    # Measurement accumulators.
    v_edges = np.linspace(0.0, p.H, n_vbins + 1)
    v_centres = 0.5 * (v_edges[:-1] + v_edges[1:])
    dv_bin = v_edges[1] - v_edges[0]
    vbin_sum = np.zeros(n_vbins)
    vbin_count = np.zeros(n_vbins)
    u_sum = np.zeros(Ny)
    u_nodes = None
    n_u_samples = 0
    planes = []
    virial = []
    centre_v = []  # per-sample mean particle speed in the central bins
    x0 = pos[:, 0].copy()
    x_unwrapped = pos[:, 0].copy()
    transit = np.full(p.N, np.nan)
    wall_time = np.zeros(p.N)
    fp_mean_sum = np.zeros(Ny)
    for step in range(1, meas_steps + 1):
        dx, y_nodes, u, fp = _step_pressure(pos, p, rng, noise_std, G, Ny)
        u_nodes = y_nodes
        x_unwrapped += dx
        near_wall = (pos[:, 1] < 2.0 * p.a) | (pos[:, 1] > p.H - 2.0 * p.a)
        wall_time[near_wall] += p.dt
        done = np.isnan(transit) & (x_unwrapped - x0 >= p.L)
        transit[done] = step * p.dt
        b = np.clip((pos[:, 1] / dv_bin).astype(int), 0, n_vbins - 1)
        np.add.at(vbin_sum, b, dx)
        np.add.at(vbin_count, b, 1.0)
        if step % sample_every == 0:
            u_sum += u
            n_u_samples += 1
            out = couette.pair_forces_and_stress(pos, p, plane_heights)
            planes.append(out["tau_planes"])
            virial.append(out["sigma_xy"])
            centre_v.append(float(np.mean(dx[(pos[:, 1] > 0.4 * p.H)
                                             & (pos[:, 1] < 0.6 * p.H)])) / p.dt)
            # Net particle force on the fluid (global check: ~ 0).
            fp_mean_sum += fp

    v_profile = vbin_sum / np.maximum(vbin_count, 1.0) / p.dt
    u_profile = u_sum / max(n_u_samples, 1)
    planes_arr = np.array(planes) if planes else None
    tau_planes = np.mean(planes_arr, axis=0) if planes_arr is not None else None
    # Total stress at the plane heights: solvent eta*u' + particle planes.
    du_dy = np.gradient(u_profile, u_nodes)
    eta_uprime = p.eta * np.interp(plane_heights, u_nodes, du_dy)
    tau_total = eta_uprime + tau_planes if tau_planes is not None else None
    # Linearity fit of the total stress: tau = G*(H/2 - y).
    if tau_total is not None:
        A = np.polyfit(plane_heights, tau_total, 1)
        pred = np.polyval(A, plane_heights)
        ss_res = float(np.sum((tau_total - pred) ** 2))
        ss_tot = float(np.sum((tau_total - np.mean(tau_total)) ** 2))
        stress_fit = {
            "slope": float(A[0]),
            "slope_over_G": float(A[0] / G) if G != 0 else float("nan"),
            "intercept": float(A[1]),
            "intercept_expected": float(G * p.H / 2.0),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        }
    else:
        stress_fit = None

    # Plug widths. The threshold is ABSOLUTE: a fraction of the wall
    # shear rate the pure solvent would have at this G
    # (gamma_w = G*H/(2*eta)). A relative threshold (fraction of the
    # measured max gradient) breaks down when the whole profile is
    # flat, because the max gradient is then itself tiny.
    gamma_w = G * p.H / (2.0 * p.eta)
    hw5, _ = _plug_half_width(v_centres, v_profile, 0.05 * gamma_w)
    hw10, _ = _plug_half_width(v_centres, v_profile, 0.10 * gamma_w)
    hw_stress = (
        _plug_half_width_stress(plane_heights, tau_total, tau_y_pred,
                                half_channel=0.5 * p.H)
        if (tau_total is not None and tau_y_pred is not None
            and tau_y_pred > 0.0)
        else (0.0 if tau_y_pred is not None and tau_y_pred <= 0.0
              else float("nan"))
    )
    virial = np.array(virial)
    half = len(virial) // 2
    centre_v = np.array(centre_v)
    chalf = len(centre_v) // 2
    umax_part = float(np.max(v_profile))
    umax_solv = float(np.max(u_profile))
    return {
        "G": G,
        "eps": p.eps,
        "tau_wall": float(G * p.H / 2.0),
        "y_vbins": v_centres,
        "v_profile": v_profile,
        "vbin_count": vbin_count,
        "y_nodes": u_nodes,
        "u_profile": u_profile,
        "plane_heights": plane_heights,
        "tau_planes": tau_planes,
        "tau_planes_std": (
            np.std(np.mean(planes_arr, axis=0)) if planes_arr is not None
            else float("nan")),
        "tau_total": tau_total,
        "stress_fit": stress_fit,
        "plug_half_width_05": float(hw5),
        "plug_half_width_10": float(hw10),
        "plug_half_width_stress": float(hw_stress),
        "plug_half_width_pred": (
            float(tau_y_pred / G) if tau_y_pred is not None and G != 0
            else float("nan")),
        "Umax_particles": umax_part,
        "Umax_solvent": umax_solv,
        "eta_eff": float(G * p.H ** 2 / (8.0 * umax_solv)) if umax_solv > 0 else float("nan"),
        "sigma_xy": float(np.mean(virial)) if virial.size else float("nan"),
        "sigma_xy_first": float(np.mean(virial[:half])) if half else float("nan"),
        "sigma_xy_second": float(np.mean(virial[half:])) if half else float("nan"),
        "centre_v_first": float(np.mean(centre_v[:chalf])) if chalf else float("nan"),
        "centre_v_second": float(np.mean(centre_v[chalf:])) if chalf else float("nan"),
        "transit": transit,
        "transit_summary": analysis.transit_summary(transit, wall_time),
        "fp_mean_integral": float(np.sum(fp_mean_sum / max(n_u_samples, 1)) * (p.H / (Ny - 1)) * p.L),
        "n_samples": int(len(virial)),
        "equil_steps": equil_steps,
        "meas_steps": meas_steps,
        "final_pos": pos.copy(),
    }
