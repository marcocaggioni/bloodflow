"""Channel with a constriction (stenosis-like) for the Bloodflow toy model.

Geometry: main channel width H_main, narrowing to a throat of width
H_throat over smooth cosine ramps, then back to H_main. The throat is
centred at xc with a flat core of half-length core_half and ramps of
length ramp on each side. H(x) = H_main near x = 0 and x = L, so the
geometry is compatible with the periodic x boundary.

Flow (lubrication approximation): the volume flux Q is constant along x,
so the local centreline velocity scales as H_main / H(x) -- the flow
speeds up through the throat. The local profile is parabolic between the
local walls:
    U(x, y) = Umax_local(x) * 4*xi*(1-xi),  xi = (y - y_lo(x)) / H(x),
    Umax_local(x) = Umax_main * H_main / H(x).

Walls: linear-spring repulsion against the curved walls (finite at
contact -- see wall_forces for why the WCA wall form is unsafe here).
The spring is stiff enough that penetration stays well under a radius.
"""

from dataclasses import dataclass

import numpy as np

from . import core


@dataclass
class Constriction:
    H_main: float = 50.0   # channel width outside the constriction
    H_throat: float = 10.0  # throat width (= 5 particle diameters for a = 1)
    L: float = 100.0       # channel period in x
    xc: float = 50.0       # throat centre
    core_half: float = 10.0  # half-length of the flat throat core
    ramp: float = 15.0     # length of each cosine ramp

    def width(self, x):
        """Local channel width H(x), array-friendly."""
        d = np.abs(np.asarray(x, dtype=float) - self.xc)
        s = np.zeros_like(d)
        s[d >= self.core_half + self.ramp] = 1.0
        ramp_zone = (d > self.core_half) & (d < self.core_half + self.ramp)
        s[ramp_zone] = 0.5 - 0.5 * np.cos(
            np.pi * (d[ramp_zone] - self.core_half) / self.ramp
        )
        return self.H_throat + (self.H_main - self.H_throat) * s

    def y_lo(self, x):
        return 0.5 * (self.H_main - self.width(x))

    def y_hi(self, x):
        return self.H_main - self.y_lo(x)

    def umax_local(self, x, Umax_main):
        return Umax_main * self.H_main / self.width(x)

    def velocity(self, x, y, Umax_main):
        """Local parabolic velocity at (x, y)."""
        w = self.width(x)
        xi = np.clip((y - self.y_lo(x)) / w, 0.0, 1.0)
        return self.umax_local(x, Umax_main) * 4.0 * xi * (1.0 - xi)


def wall_forces(pos, p, constr):
    """Linear-spring repulsion from the curved walls.

    F = k*(a - gap) pushing inward whenever gap < a (zero otherwise),
    capped at _WALL_FCAP. Deliberately NOT the WCA wall form: the WCA
    force diverges as gap -> 0, and with explicit Euler a particle can
    land at a tiny gap when the steep ramp wall sweeps past it, which
    explodes the step. The spring force is finite everywhere, so a
    swept particle is pushed back over a few steps instead of ejected.
    A hard position clamp was tried first and rejected: it piled
    particles onto a single line at the clamp distance, created
    overlaps, and produced impossible transit times.
    """
    F = np.zeros((pos.shape[0], 2))
    g_lo = pos[:, 1] - constr.y_lo(pos[:, 0])
    g_hi = constr.y_hi(pos[:, 0]) - pos[:, 1]
    f_lo = np.minimum(_WALL_K * np.clip(p.a - g_lo, 0.0, None), _WALL_FCAP)
    f_hi = np.minimum(_WALL_K * np.clip(p.a - g_hi, 0.0, None), _WALL_FCAP)
    F[:, 1] += f_lo
    F[:, 1] -= f_hi
    return F


_WALL_K = 500.0
_WALL_FCAP = 200.0


def init_positions_constriction(p, constr, rng):
    """Random non-overlapping positions inside the curved channel."""
    pos = np.zeros((p.N, 2))
    placed = 0
    tries = 0
    min_sep2 = (2.05 * p.a) ** 2
    while placed < p.N and tries < 400 * p.N:
        tries += 1
        x = rng.uniform(0.0, p.L)
        lo = constr.y_lo(x) + 1.3 * p.a
        hi = constr.y_hi(x) - 1.3 * p.a
        if hi <= lo:
            continue
        cand = np.array([x, rng.uniform(lo, hi)])
        if placed == 0:
            pos[placed] = cand
            placed += 1
            continue
        dx = pos[:placed, 0] - cand[0]
        dx -= p.L * np.round(dx / p.L)
        d2 = dx * dx + (pos[:placed, 1] - cand[1]) ** 2
        if np.all(d2 > min_sep2):
            pos[placed] = cand
            placed += 1
    if placed < p.N:  # fall back: fill the rest randomly (WCA sorts it out)
        xs = rng.uniform(0.0, p.L, size=p.N - placed)
        pos[placed:, 0] = xs
        pos[placed:, 1] = 0.5 * (
            constr.y_lo(xs) + constr.y_hi(xs)
        ) + rng.uniform(-1.0, 1.0, size=p.N - placed) * 0.5 * (
            constr.y_hi(xs) - constr.y_lo(xs) - 2.6 * p.a
        )
    return pos


def init_positions_grid(p, constr, target_n=None):
    """Hexagonal-lattice starting positions (a uniform distribution).

    Points of a hex lattice covering the bounding box are kept when they
    lie inside the local channel bounds with a 1.3a margin. The lattice
    spacing is tuned so the kept count lands as close as possible to
    target_n (default p.N); the caller should use len(positions) as the
    actual particle count, since an exact match is not always possible.
    The x-spacing divides L exactly, so the lattice is seamless across
    the periodic x boundary. Deterministic: no randomness, so both cases
    start identically, the lattice spacing (~3a) exceeds the cluster
    cutoff (so the video opens with all singletons), and aggregation
    must develop from the lattice rather than from random initial
    proximity.
    """
    target = target_n or p.N
    xs_grid = np.linspace(0.0, p.L, 2001)
    area = float(np.mean(constr.width(xs_grid))) * p.L
    s0 = float(np.sqrt(area / target))
    n_cols = max(1, int(round(p.L / s0)))
    best = None
    seen = set()
    for _ in range(40):
        if n_cols in seen or n_cols < 1:
            break
        seen.add(n_cols)
        pts = _hex_lattice_points(p.L / n_cols, p, constr)
        if best is None or abs(len(pts) - target) < abs(len(best) - target):
            best = pts
        if len(pts) == target:
            break
        n_cols += 1 if len(pts) < target else -1
    return best


def _hex_lattice_points(s_x, p, constr):
    row_h = s_x * np.sqrt(3.0) / 2.0
    n_rows = int(np.floor(p.H / row_h)) + 1
    y0 = 0.5 * p.H - 0.5 * (n_rows - 1) * row_h
    n_cols = int(round(p.L / s_x))
    pts = []
    for r in range(n_rows):
        y = y0 + r * row_h
        x_off = 0.5 * s_x if r % 2 else 0.0
        for c in range(n_cols):
            x = (c * s_x + x_off) % p.L
            if constr.y_lo(x) + 1.3 * p.a <= y <= constr.y_hi(x) - 1.3 * p.a:
                pts.append((x, y))
    pts.sort()
    return np.array(pts)


class ConstrictionResult:
    def __init__(self, p, transit, wall_time, passage, final_pos,
                 total_time, n_repairs=0):
        self.params = p
        self.transit = transit        # time to travel distance L, NaN if censored
        self.wall_time = wall_time    # cumulative time within 2a of a local wall
        self.passage = passage        # first throat-core passage time, NaN if none
        self.final_pos = final_pos
        self.total_time = total_time
        self.n_repairs = n_repairs    # particles re-injected after escaping far


def simulate_constriction(p, constr, n_steps, equil_steps=0, pos0=None):
    """Run the model in the constricted channel.

    Throat passage: the throat core is [xc - core_half, xc + core_half].
    Passage time is the first entry-to-exit time after the measurement
    phase starts; particles that never complete a passage keep NaN.
    """
    rng = np.random.default_rng(p.seed)
    pos = pos0.copy() if pos0 is not None else init_positions_constriction(
        p, constr, rng
    )
    x1 = constr.xc - constr.core_half
    x2 = constr.xc + constr.core_half
    n_repairs = 0
    for _ in range(equil_steps):
        n_repairs += _step_constriction(pos, p, constr, rng)[1]
    x0 = pos[:, 0].copy()
    x_unwrapped = pos[:, 0].copy()
    transit = np.full(p.N, np.nan)
    wall_time = np.zeros(p.N)
    passage = np.full(p.N, np.nan)
    in_throat = np.zeros(p.N, dtype=bool)
    entry_step = np.zeros(p.N, dtype=int)
    for step in range(1, n_steps + 1):
        dx, n_rep = _step_constriction(pos, p, constr, rng)
        n_repairs += n_rep
        x_unwrapped += dx
        x_unwrapped += dx
        near_wall = (pos[:, 1] - constr.y_lo(pos[:, 0]) < 2.0 * p.a) | (
            constr.y_hi(pos[:, 0]) - pos[:, 1] < 2.0 * p.a
        )
        wall_time[near_wall] += p.dt
        done = np.isnan(transit) & (x_unwrapped - x0 >= p.L)
        transit[done] = step * p.dt
        # Throat passage tracking (first complete passage only).
        inside = (pos[:, 0] >= x1) & (pos[:, 0] <= x2)
        entering = inside & ~in_throat & np.isnan(passage)
        in_throat[entering] = True
        entry_step[entering] = step
        exiting = ~inside & in_throat
        passage[exiting] = (step - entry_step[exiting]) * p.dt
        in_throat[exiting] = False
    return ConstrictionResult(p, transit, wall_time, passage,
                              pos.copy(), n_steps * p.dt, n_repairs)


def _step_constriction(pos, p, constr, rng):
    F = core.compute_pair_forces(pos, p) + wall_forces(pos, p, constr)
    # Cap the total force per particle. Normal forces here are <= ~100;
    # only pathological overlaps (a particle squeezed between neighbours
    # in a throat-entrance jam) reach the cap. Without it, the diverging
    # WCA pair force at deep overlap ejects the particle across the
    # domain in one step and corrupts every statistic downstream.
    norms = np.sqrt(F[:, 0] ** 2 + F[:, 1] ** 2)
    over = norms > _FORCE_CAP
    if np.any(over):
        F[over] *= (_FORCE_CAP / norms[over])[:, None]
    adv = constr.velocity(pos[:, 0], pos[:, 1], p.Umax)
    dx = (adv + F[:, 0] / p.zeta) * p.dt
    dy = (F[:, 1] / p.zeta) * p.dt
    pos[:, 0] = (pos[:, 0] + dx) % p.L
    pos[:, 1] = pos[:, 1] + dy
    # Safety valve: a particle found far outside the channel (more than
    # 5a past a wall) is re-injected near the centreline at the same x.
    # This should not trigger once forces are capped; the count is
    # reported so a run that needed it is visible, not silent.
    y_lo = constr.y_lo(pos[:, 0])
    y_hi = constr.y_hi(pos[:, 0])
    outside = (pos[:, 1] < y_lo - 5.0 * p.a) | (pos[:, 1] > y_hi + 5.0 * p.a)
    n_rep = int(np.sum(outside))
    if n_rep:
        mid = 0.5 * (y_lo[outside] + y_hi[outside])
        half = np.maximum(0.5 * (y_hi[outside] - y_lo[outside]) - 2.0 * p.a,
                          0.0)
        pos[outside, 1] = mid + rng.uniform(-1.0, 1.0, size=n_rep) * half
    return dx, n_rep


_FORCE_CAP = 300.0
