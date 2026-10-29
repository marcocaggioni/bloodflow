# Overdamped Brownian dynamics of rigid disks in a 2D Poiseuille channel.
#
# State: N disks of radius a in a channel of width H (walls at y = 0 and
# y = H), periodic in x with period L.
#
# Equation of motion per particle:
#     dr/dt = U(y) x-hat + (1/zeta) * sum_j F_ij + thermal noise
# with zeta = 6*pi*eta*a (3D Stokes drag for spheres constrained to a
# plane; this avoids the 2D Stokes paradox) and the imposed Poiseuille
# profile U(y) = 4*Umax*(y/H)*(1 - y/H), zero at the walls.
#
# Pair forces: WCA short-range repulsion + Morse attraction, evaluated
# with a linked-cell list (padded grid, vectorised over cell pairs).
# Wall forces: WCA repulsion from both walls.

from dataclasses import dataclass, field

import numpy as np

from . import potentials


@dataclass
class Params:
    N: int = 100          # number of disks
    a: float = 1.0        # particle radius
    H: float = 16.0       # channel width (walls at y=0 and y=H)
    L: float = 50.0       # channel period in x
    eta: float = 1.0      # plasma viscosity
    Umax: float = 1.0     # centreline Poiseuille velocity
    eps: float = 0.0      # Morse attraction depth (0 = no aggregation)
    kappa: float = 2.0    # Morse inverse range (range ~ 1/kappa)
    eps_rep: float = 1.0  # WCA repulsion energy scale
    kT: float = 0.0       # thermal energy (0 = deterministic)
    dt: float = 0.01      # time step
    seed: int = 0

    # Derived quantities, filled in by __post_init__.
    zeta: float = field(init=False)
    wca_sigma: float = field(init=False)
    wca_cut: float = field(init=False)
    morse_cut: float = field(init=False)
    cell: float = field(init=False)
    gamma_w: float = field(init=False)  # wall shear rate 4*Umax/H

    def __post_init__(self):
        self.zeta = 6.0 * np.pi * self.eta * self.a
        self.wca_sigma = 2.0 * self.a
        self.wca_cut = potentials.wca_cutoff(self.wca_sigma)
        self.morse_cut = potentials.morse_cutoff(self.a, self.kappa)
        self.cell = max(self.wca_cut, self.morse_cut)
        self.gamma_w = 4.0 * self.Umax / self.H

    def mason_number(self):
        """Adhesion-vs-shear number: (eps / kappa^-1) / (eta * gamma_w * a^2).

        Attraction force scale ~ eps * kappa over the hydrodynamic force
        scale eta * gamma_w * a^2. >> 1 means aggregates survive the shear.
        """
        return (self.eps * self.kappa) / (
            self.eta * self.gamma_w * self.a ** 2
        )


def poiseuille_velocity(y, p):
    """Imposed Poiseuille profile U(y), zero at both walls."""
    s = y / p.H
    return 4.0 * p.Umax * s * (1.0 - s)


def _padded_cell_grid(pos, p):
    """Bin particles into a linked-cell grid.

    Returns (Pgrid, nx, ny, maxc) where Pgrid[ix, iy, slot] holds the
    particle index in that cell/slot, or -1 if empty. x is periodic, y is
    clipped (walls confine the particles).
    """
    nx = max(1, int(np.ceil(p.L / p.cell)))
    ny = max(1, int(np.ceil(p.H / p.cell)))
    ix = (np.floor(pos[:, 0] / p.cell).astype(int)) % nx
    iy = np.clip(np.floor(pos[:, 1] / p.cell).astype(int), 0, ny - 1)
    # Row-major flat id, consistent with the reshape to (nx, ny, maxc) below.
    cid = ix * ny + iy
    ncell = nx * ny
    order = np.argsort(cid, kind="stable")
    sorted_cid = cid[order]
    counts = np.bincount(sorted_cid, minlength=ncell)
    starts = np.zeros(ncell, dtype=int)
    starts[1:] = np.cumsum(counts)[:-1]
    slot = np.arange(pos.shape[0]) - starts[sorted_cid]
    maxc = int(counts.max()) if counts.size else 0
    P = np.full((ncell, maxc), -1, dtype=int)
    if pos.shape[0]:
        P[sorted_cid, slot] = order
    return P.reshape(nx, ny, maxc), nx, ny, maxc


def compute_pair_forces(pos, p):
    """Pair (WCA + Morse) forces only. Returns an (N, 2) array."""
    N = pos.shape[0]
    F = np.zeros((N, 2))
    if N == 0:
        return F
    Pgrid, nx, ny, maxc = _padded_cell_grid(pos, p)
    if maxc == 0:
        return F
    slots = np.arange(maxc)
    cut2 = p.cell ** 2

    # Half-stencil: each unordered cell pair is visited exactly once.
    # (0, 0) is the intra-cell case, handled with a slot mask.
    for dx_cell, dy_cell in ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 1)):
        rolled = np.roll(Pgrid, shift=-dx_cell, axis=0)
        padded = np.full((nx, ny + 2, maxc), -1, dtype=int)
        padded[:, 1 : ny + 1, :] = rolled
        Q = padded[:, 1 + dy_cell : 1 + dy_cell + ny, :]
        I = Pgrid[:, :, :, None]  # (nx, ny, maxc, 1)
        J = Q[:, :, None, :]     # (nx, ny, 1, maxc)
        valid = (I >= 0) & (J >= 0)
        if dx_cell == 0 and dy_cell == 0:
            valid &= slots[None, None, None, :] > slots[None, None, :, None]
        if not valid.any():
            continue
        i_idx = np.clip(I, 0, N - 1)
        j_idx = np.clip(J, 0, N - 1)
        dx = pos[i_idx, 0] - pos[j_idx, 0]
        dx -= p.L * np.round(dx / p.L)  # minimum image in x
        dy = pos[i_idx, 1] - pos[j_idx, 1]
        r2 = dx * dx + dy * dy
        valid &= (r2 < cut2) & (r2 > 0.0)
        if not valid.any():
            continue
        r = np.sqrt(np.where(valid, r2, 1.0))
        s = np.zeros_like(r)
        wca_mask = valid & (r < p.wca_cut)
        if wca_mask.any():
            rw = r[wca_mask]
            sr = p.wca_sigma / rw
            sr6 = sr ** 6
            s[wca_mask] = (24.0 * p.eps_rep / rw) * (2.0 * sr6 * sr6 - sr6)
        if p.eps != 0.0:
            morse_mask = valid & (r < p.morse_cut)
            if morse_mask.any():
                rm = r[morse_mask]
                x = rm - 2.0 * p.a
                s[morse_mask] += 2.0 * p.kappa * p.eps * (
                    np.exp(-2.0 * p.kappa * x) - np.exp(-p.kappa * x)
                )
        fx = np.where(valid, s * dx / r, 0.0)
        fy = np.where(valid, s * dy / r, 0.0)
        i_flat = np.broadcast_to(I, valid.shape)[valid]
        j_flat = np.broadcast_to(J, valid.shape)[valid]
        np.add.at(F[:, 0], i_flat, fx[valid])
        np.add.at(F[:, 0], j_flat, -fx[valid])
        np.add.at(F[:, 1], i_flat, fy[valid])
        np.add.at(F[:, 1], j_flat, -fy[valid])

    return F


def compute_forces(pos, p):
    """Pair (WCA + Morse) + flat-wall forces. Returns an (N, 2) array."""
    F = compute_pair_forces(pos, p)
    # Wall forces (WCA against y = 0 and y = H).
    for d, sign in ((pos[:, 1], 1.0), (p.H - pos[:, 1], -1.0)):
        mask = (d < potentials._WCA_C * p.a) & (d > 0.0)
        if mask.any():
            dm = d[mask]
            ar = p.a / dm
            ar6 = ar ** 6
            F[mask, 1] += sign * (24.0 * p.eps_rep / dm) * (
                2.0 * ar6 * ar6 - ar6
            )
    return F


def init_positions(p, rng):
    """Random non-overlapping positions, clear of the wall repulsion zone."""
    y_lo = 1.3 * p.a
    y_hi = p.H - 1.3 * p.a
    pos = np.zeros((p.N, 2))
    placed = 0
    tries = 0
    min_sep2 = (2.05 * p.a) ** 2
    while placed < p.N and tries < 200 * p.N:
        tries += 1
        cand = np.array(
            [rng.uniform(0.0, p.L), rng.uniform(y_lo, y_hi)]
        )
        if placed == 0:
            pos[placed] = cand
            placed += 1
            continue
        dx = pos[:placed, 0] - cand[0]
        dx -= p.L * np.round(dx / p.L)  # minimum image in x
        d2 = dx * dx + (pos[:placed, 1] - cand[1]) ** 2
        if np.all(d2 > min_sep2):
            pos[placed] = cand
            placed += 1
    if placed < p.N:  # fall back: fill the rest randomly (WCA sorts it out)
        pos[placed:, 0] = rng.uniform(0.0, p.L, size=p.N - placed)
        pos[placed:, 1] = rng.uniform(y_lo, y_hi, size=p.N - placed)
    return pos


def _step(pos, p, rng, noise_std):
    F = compute_forces(pos, p)
    adv = poiseuille_velocity(pos[:, 1], p)
    dx = (adv + F[:, 0] / p.zeta) * p.dt
    dy = (F[:, 1] / p.zeta) * p.dt
    if noise_std > 0.0:
        dx = dx + noise_std * rng.standard_normal(pos.shape[0])
        dy = dy + noise_std * rng.standard_normal(pos.shape[0])
    pos[:, 0] = (pos[:, 0] + dx) % p.L
    pos[:, 1] = pos[:, 1] + dy
    return dx


class Result:
    """Simulation outcome: per-particle transit and wall-residence data."""

    def __init__(self, p, transit, wall_time, final_pos, total_time):
        self.params = p
        self.transit = transit          # time to travel distance L, NaN if censored
        self.wall_time = wall_time      # cumulative time within one diameter of a wall
        self.final_pos = final_pos
        self.total_time = total_time


def simulate(p, n_steps, equil_steps=0, pos0=None):
    """Run the model.

    An equilibration phase (flow on, clocks off) lets aggregates form
    before the transit clocks start. Transit time is the first time a
    particle's unwrapped x-displacement reaches L; particles that never
    make it within n_steps keep NaN (censored). pos0 optionally overrides
    the random initial positions (used by the validation test).
    """
    rng = np.random.default_rng(p.seed)
    pos = pos0.copy() if pos0 is not None else init_positions(p, rng)
    noise_std = 0.0
    if p.kT > 0.0:
        D = p.kT / p.zeta
        noise_std = np.sqrt(2.0 * D * p.dt)
    for _ in range(equil_steps):
        _step(pos, p, rng, noise_std)
    x0 = pos[:, 0].copy()
    x_unwrapped = pos[:, 0].copy()
    transit = np.full(p.N, np.nan)
    wall_time = np.zeros(p.N)
    for step in range(n_steps):
        dx = _step(pos, p, rng, noise_std)
        x_unwrapped += dx
        near_wall = (pos[:, 1] < 2.0 * p.a) | (pos[:, 1] > p.H - 2.0 * p.a)
        wall_time[near_wall] += p.dt
        done = np.isnan(transit) & (x_unwrapped - x0 >= p.L)
        transit[done] = (step + 1) * p.dt
    return Result(p, transit, wall_time, pos.copy(), n_steps * p.dt)
