"""Branching vascular network for the Bloodflow toy model.

Geometry: a symmetric ladder. One trunk (width W0) splits 1 -> 2 -> 4
-> 8 through Y-junctions, the 8 terminal branches run in parallel,
then a mirror-image recombination merges 8 -> 4 -> 2 -> 1 into an
outlet trunk collinear with the inlet trunk. The outlet trunk wraps
periodically onto the inlet trunk, closing the loop the way the
circulation returns to the heart.

Widths halve at each split (W0 -> W0/2 -> W0/4 -> W0/8), so with
equal flux splits the centreline velocity is the same at every level
(mean velocity = flux / width, both halving together) while the
vessel-to-cell diameter ratio spans a decade (W0/2a down to W0/16a).

Channel representation: the fluid region is the union of capsules
around straight centreline segments (a segment's capsule = points
within width/2 of the segment). Flow is an imposed local Poiseuille
profile per segment, scaled by the segment's flux share, blended by
capsule depth where segments overlap at junctions. Walls use the
linear-spring force from constriction.py (finite at contact -- the
WCA wall form diverges and explodes steps on curved walls).

Periodic wrap: only the two trunks touch x = 0 and x = L_total, and
they share centreline and width, so a plain x-wrap at equal y is
exact. Pair forces use core.compute_pair_forces with wrap_x=False:
particles must not interact across the wrap plane (a particle at the
outlet must not feel one at the inlet).
"""

from dataclasses import dataclass, field

import numpy as np

from . import core
from .constriction import _WALL_FCAP, _WALL_K


@dataclass
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    level: int       # generation: 0 = trunk, 1, 2, 3 = terminals
    share: float     # fraction of trunk flux carried
    # Derived, filled in by build_network.
    ux: float = field(init=False)    # unit axis vector
    uy: float = field(init=False)
    length: float = field(init=False)
    umax: float = field(init=False)  # centreline velocity


@dataclass
class Network:
    segments: list
    L_total: float   # domain length in x (wrap plane)
    y_lo: float      # plot bounds
    y_hi: float
    W0: float
    Umax_trunk: float

    def __post_init__(self):
        self._sx0 = np.array([s.x0 for s in self.segments])
        self._sy0 = np.array([s.y0 for s in self.segments])
        self._sux = np.array([s.ux for s in self.segments])
        self._suy = np.array([s.uy for s in self.segments])
        self._slen = np.array([s.length for s in self.segments])
        self._sw = np.array([s.width for s in self.segments])
        self._sumax = np.array([s.umax for s in self.segments])
        self._slevel = np.array([s.level for s in self.segments])


def build_network(W0=100.0, YC=100.0, Umax_trunk=2.0):
    """Build the symmetric 1->2->4->8->4->2->1 ladder.

    Layout (x, y): trunk (0..80), L1 diagonals to x=130 then
    horizontals to x=180, L2 diagonals to x=230 then horizontals to
    x=280, L3 diagonals to x=330 then terminal horizontals to x=400,
    mirror recombination to x=650, outlet trunk to x=730.
    """
    segs = []

    def add(x0, y0, x1, y1, w, level, share):
        segs.append(Segment(x0, y0, x1, y1, w, level, share))

    # Trunk in.
    add(0.0, YC, 80.0, YC, W0, 0, 1.0)
    # Level 1: split to 2 (width W0/2).
    for sgn in (1.0, -1.0):
        yc = YC + sgn * 35.0
        add(80.0, YC, 130.0, yc, W0 / 2, 1, 0.5)
        add(130.0, yc, 180.0, yc, W0 / 2, 1, 0.5)
    # Level 2: split to 4 (width W0/4).
    for yc0 in (YC + 35.0, YC - 35.0):
        for sgn in (1.0, -1.0):
            yc = yc0 + sgn * 20.0
            add(180.0, yc0, 230.0, yc, W0 / 4, 2, 0.25)
            add(230.0, yc, 280.0, yc, W0 / 4, 2, 0.25)
    # Level 3: split to 8 terminals (width W0/8).
    for yc0 in (YC + 55.0, YC + 15.0, YC - 15.0, YC - 55.0):
        for sgn in (1.0, -1.0):
            yc = yc0 + sgn * 10.0
            add(280.0, yc0, 330.0, yc, W0 / 8, 3, 0.125)
            add(330.0, yc, 400.0, yc, W0 / 8, 3, 0.125)
    # Recombination, mirror image. Merge diagonals keep the INCOMING
    # branch's width and share (they are that branch until the merge
    # point); only the merged horizontals take the summed share.
    for yc0, yc1 in ((165.0, 155.0), (145.0, 155.0),
                     (125.0, 115.0), (105.0, 115.0),
                     (95.0, 85.0), (75.0, 85.0),
                     (55.0, 45.0), (35.0, 45.0)):
        add(400.0, yc0, 450.0, yc1, W0 / 8, 3, 0.125)
    for yc in (155.0, 115.0, 85.0, 45.0):
        add(450.0, yc, 500.0, yc, W0 / 4, 2, 0.25)
    for yc0, yc1 in ((155.0, 135.0), (115.0, 135.0),
                     (85.0, 65.0), (45.0, 65.0)):
        add(500.0, yc0, 550.0, yc1, W0 / 4, 2, 0.25)
    for yc in (135.0, 65.0):
        add(550.0, yc, 600.0, yc, W0 / 2, 1, 0.5)
    for yc in (135.0, 65.0):
        add(600.0, yc, 650.0, YC, W0 / 2, 1, 0.5)
    # Trunk out.
    add(650.0, YC, 730.0, YC, W0, 0, 1.0)

    # Derived per-segment quantities. Flux Q = (2/3)*Umax*W for a
    # parabolic profile; equal splits give share-scaled flux, and
    # Umax_seg = 1.5 * mean = 1.5 * Q*share / width.
    Q = (2.0 / 3.0) * Umax_trunk * W0
    for s in segs:
        dx, dy = s.x1 - s.x0, s.y1 - s.y0
        s.length = float(np.hypot(dx, dy))
        s.ux, s.uy = dx / s.length, dy / s.length
        s.umax = 1.5 * Q * s.share / s.width
    y_lo = min(min(s.y0, s.y1) - s.width / 2 for s in segs)
    y_hi = max(max(s.y0, s.y1) + s.width / 2 for s in segs)
    return Network(segs, 730.0, y_lo, y_hi, W0, Umax_trunk)


def _segment_geometry(pos, net):
    """Per-particle, per-segment capsule geometry.

    Returns (inside, xi, dvec_x, dvec_y): inside = width/2 - distance
    to the segment (positive inside the capsule); xi = across-channel
    coordinate in [0, 1] from the segment's profile; dvec = vector
    from the closest centreline point to the particle.
    """
    px = pos[:, 0][:, None]          # (N, 1)
    py = pos[:, 1][:, None]
    relx = px - net._sx0[None, :]
    rely = py - net._sy0[None, :]
    t = (relx * net._sux[None, :] + rely * net._suy[None, :]) / net._slen[None, :]
    t = np.clip(t, 0.0, 1.0)
    cx = net._sx0[None, :] + t * net._slen[None, :] * net._sux[None, :]
    cy = net._sy0[None, :] + t * net._slen[None, :] * net._suy[None, :]
    dvx = px - cx
    dvy = py - cy
    dist = np.hypot(dvx, dvy)
    inside = net._sw[None, :] / 2.0 - dist
    # Signed perpendicular offset from the centreline (left normal).
    perp = dvx * (-net._suy[None, :]) + dvy * net._sux[None, :]
    xi = np.clip((perp + net._sw[None, :] / 2.0) / net._sw[None, :], 0.0, 1.0)
    return inside, xi, dvx, dvy


def _velocity_from_geom(net, inside, xi):
    """Blend per-segment parabolic profiles by capsule depth."""
    wgt = np.clip(inside, 0.0, None)
    prof = net._sumax[None, :] * 4.0 * xi * (1.0 - xi)
    vx = (wgt * prof * net._sux[None, :]).sum(axis=1)
    vy = (wgt * prof * net._suy[None, :]).sum(axis=1)
    wsum = wgt.sum(axis=1)
    ok = wsum > 0.0
    vx = np.where(ok, vx / np.where(ok, wsum, 1.0), 0.0)
    vy = np.where(ok, vy / np.where(ok, wsum, 1.0), 0.0)
    return np.stack([vx, vy], axis=1)


def _wall_from_geom(p, inside, dvx, dvy):
    """Linear-spring wall force from precomputed capsule geometry.

    Gap = capsule depth of the dominant segment. The force pushes
    TOWARD that segment's centreline (minus the centreline-to-particle
    vector) whenever the gap is smaller than the particle radius, and
    is capped like constriction.py. (The sign matters: pushing along
    the outward vector herds particles through the wall and explodes
    the run -- caught by the smoke test, 2026-10-29.)
    """
    best = np.argmax(inside, axis=1)
    rows = np.arange(inside.shape[0])
    gap = inside[rows, best]
    bx = dvx[rows, best]
    by = dvy[rows, best]
    norm = np.hypot(bx, by)
    norm = np.where(norm > 0.0, norm, 1.0)
    mag = np.minimum(_WALL_K * np.clip(p.a - gap, 0.0, None), _WALL_FCAP)
    F = np.zeros((inside.shape[0], 2))
    F[:, 0] = -mag * bx / norm
    F[:, 1] = -mag * by / norm
    return F


def velocity_field(pos, net):
    """Imposed flow velocity at particle positions (N, 2).

    Per-segment parabolic profiles, blended by capsule depth where
    segments overlap at junctions. Returns (N, 2) velocities.
    """
    inside, xi, _, _ = _segment_geometry(pos, net)
    return _velocity_from_geom(net, inside, xi)


def dominant_level(pos, net):
    """Level of the dominant (deepest-inside) segment per particle."""
    inside, _, _, _ = _segment_geometry(pos, net)
    return net._slevel[np.argmax(inside, axis=1)]


def inside_measure(pos, net):
    """Max capsule depth per particle (> 0 inside the channel)."""
    inside, _, _, _ = _segment_geometry(pos, net)
    return inside.max(axis=1)


def wall_forces(pos, p, net):
    """Linear-spring repulsion from the network walls.

    Gap = capsule depth of the dominant segment; force pushes away
    from that segment's centreline, capped like constriction.py.
    """
    inside, _, dvx, dvy = _segment_geometry(pos, net)
    return _wall_from_geom(p, inside, dvx, dvy)


def step_network(pos, p, net, rng, noise_std=0.0):
    """One Euler step with network flow, network walls, no-wrap pair
    forces, and the periodic trunk-to-trunk wrap in x. Geometry is
    computed once and shared between flow and walls. Returns
    (displacement (N, 2), wrapped (N,) bool)."""
    inside, xi, dvx, dvy = _segment_geometry(pos, net)
    F = core.compute_pair_forces(pos, p, wrap_x=False)
    F = F + _wall_from_geom(p, inside, dvx, dvy)
    vel = _velocity_from_geom(net, inside, xi)
    dx = (vel[:, 0] + F[:, 0] / p.zeta) * p.dt
    dy = (vel[:, 1] + F[:, 1] / p.zeta) * p.dt
    if noise_std > 0.0:
        dx = dx + noise_std * rng.standard_normal(pos.shape[0])
        dy = dy + noise_std * rng.standard_normal(pos.shape[0])
    old_x = pos[:, 0].copy()
    pos[:, 0] = (pos[:, 0] + dx) % net.L_total
    pos[:, 1] = pos[:, 1] + dy
    wrapped = (pos[:, 0] - old_x) < -net.L_total / 2.0
    return np.stack([dx, dy], axis=1), wrapped


def channel_area(net, n_samples=200000, seed=12345):
    """Monte Carlo estimate of the fluid area (for particle count)."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, net.L_total, n_samples)
    ys = rng.uniform(net.y_lo, net.y_hi, n_samples)
    pts = np.stack([xs, ys], axis=1)
    frac = float((inside_measure(pts, net) > 0.0).mean())
    return frac * net.L_total * (net.y_hi - net.y_lo)


def init_positions_network(net, p, rng, phi=0.25):
    """Lattice initial positions inside the channel, non-overlapping.

    Hex lattice points inside the channel (clear of walls) are
    shuffled and subsampled to the target count for area fraction
    phi; subsampling a lattice preserves the minimum separation.
    The lattice is wrap-consistent by construction: the x-spacing is
    L_total / floor(L_total / spacing), so columns tile the period
    exactly and the wrapped distance between the last and first
    columns equals the spacing. (An earlier version used plain
    arange spacing, which left the first and last columns 0.9 apart
    across the wrap plane; the first particle to wrap landed on a
    neighbour and exploded the pair forces -- caught 2026-10-29.)
    """
    area = channel_area(net)
    n_target = int(phi * area / (np.pi * p.a * p.a))
    for spacing in (2.3, 2.15, 2.05):
        n_x = max(1, int(net.L_total // spacing))
        dx = net.L_total / n_x
        dy = dx * np.sqrt(3.0) / 2.0
        xs = np.arange(n_x) * dx
        ys = np.arange(net.y_lo, net.y_hi, dy)
        xx, yy = np.meshgrid(xs, ys)
        xx[1::2] += dx / 2.0
        pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        # Belt and braces: no point at or beyond the wrap plane.
        keep = (inside_measure(pts, net) > 1.3 * p.a) & (pts[:, 0] < net.L_total)
        pts = pts[keep]
        if pts.shape[0] >= n_target:
            idx = rng.permutation(pts.shape[0])[:n_target]
            return pts[idx], n_target, area
    raise RuntimeError("lattice too sparse for target count")
