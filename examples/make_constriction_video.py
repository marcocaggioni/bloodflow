"""Render a side-by-side video of flow through the constricted channel:
no attraction (left) vs strong attraction (right), same initial
positions. The throat is 10 wide (5 particle diameters) between x = 40
and x = 60, with cosine ramps back to the full 50-wide channel.

Watch for: aggregates approaching the throat, squeezing through it
(they fit -- a ~15-particle cluster spans about 4 diameters), and what
they look like downstream.

Usage:
    python3 examples/make_constriction_video.py [--out FILE] [--seconds S]
Requires matplotlib and ffmpeg on PATH. Frames go to a temp dir.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bloodflow.constriction import (
    Constriction,
    _step_constriction,
    init_positions_constriction,
    init_positions_grid,
)
from bloodflow.core import Params

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def cluster_colors(pos, p, cutoff):
    """Vectorised clustering; singletons grey, clusters tab colours."""
    n = pos.shape[0]
    dx = pos[:, 0][:, None] - pos[:, 0][None, :]
    dx -= p.L * np.round(dx / p.L)  # minimum image in x
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    adj = (dx * dx + dy * dy) < cutoff * cutoff
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    iu = np.triu_indices(n, k=1)
    for i, j in zip(*iu):
        if adj[i, j]:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[ri] = rj
    labels = np.array([find(i) for i in range(n)])
    _, counts = np.unique(labels, return_counts=True)
    size_of = {lab: int(c) for lab, c in zip(np.unique(labels), counts)}
    ranked = sorted(
        (lab for lab in np.unique(labels) if size_of[lab] >= 2),
        key=lambda lab: -size_of[lab],
    )
    palette = list(mcolors.TABLEAU_COLORS.values()) + list(
        mcolors.CSS4_COLORS.values()
    )
    col_of = {}
    for k, lab in enumerate(ranked):
        col_of[lab] = mcolors.to_rgba(palette[k % len(palette)])
    grey = (0.72, 0.72, 0.72, 1.0)
    return np.array([col_of.get(lab, grey) for lab in labels])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="constriction-video.mp4")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--init", choices=["random", "grid"], default="random")
    ap.add_argument("--equil-steps", type=int, default=3000)
    args = ap.parse_args()

    a, H_main, H_throat, L, phi = 1.0, 50.0, 10.0, 100.0, 0.25
    constr = Constriction(H_main=H_main, H_throat=H_throat, L=L,
                          xc=50.0, core_half=10.0, ramp=15.0)
    n = int(phi * L * H_main / (np.pi * a * a))
    Umax, dt, kappa = 2.0, 0.02, 2.0
    n_steps = 6000
    n_frames = int(args.fps * args.seconds)
    stride = max(1, n_steps // n_frames)

    base = dict(N=n, a=a, H=H_main, L=L, eta=1.0, Umax=Umax,
                kappa=kappa, kT=0.0, dt=dt, seed=7)
    p_off = Params(eps=0.0, **base)
    p_on = Params(eps=1.28, **base)
    if args.init == "grid":
        pos_off = init_positions_grid(p_off, constr)
        n = len(pos_off)
        base["N"] = n
        p_off = Params(eps=0.0, **base)
        p_on = Params(eps=1.28, **base)
    else:
        rng = np.random.default_rng(7)
        pos_off = init_positions_constriction(p_off, constr, rng)
    pos_on = pos_off.copy()
    print(f"init={args.init}  N={n}  throat={H_throat} (5 diameters)  "
          f"Mason main={p_on.mason_number():.1f}")
    rng_off = np.random.default_rng(7)
    rng_on = np.random.default_rng(7)
    # Equilibration, as in the analysis runs (skipped for a grid start
    # so the video opens on the uniform lattice itself).
    for _ in range(args.equil_steps):
        _step_constriction(pos_off, p_off, constr, rng_off)
        _step_constriction(pos_on, p_on, constr, rng_on)

    xs = np.linspace(0, L, 400)
    y_lo, y_hi = constr.y_lo(xs), constr.y_hi(xs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), dpi=100)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.86, bottom=0.10,
                        wspace=0.08)
    titles = ["No attraction (eps = 0)", "Strong attraction (eps = 1.28)"]
    scats = []
    for ax, title in zip(axes, titles):
        ax.fill_between(xs, y_lo, y_hi, color="#dbe7f3", zorder=0)
        ax.plot(xs, y_lo, color="#4a7bb5")
        ax.plot(xs, y_hi, color="#4a7bb5")
        ax.set_xlim(0, L)
        ax.set_ylim(0, H_main)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("x (flow direction)")
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("y (channel width)")
    axes[1].set_yticklabels([])
    time_text = fig.suptitle("", fontsize=13)
    fig.canvas.draw()
    p0 = axes[0].transData.transform((0.0, 0.0))
    p1 = axes[0].transData.transform((2.0 * a, 0.0))
    r_px = abs(p1[0] - p0[0]) / 2.0
    size = (2.0 * r_px * 72.0 / fig.dpi) ** 2
    for ax in axes:
        scats.append(ax.scatter([], [], s=size, edgecolors="none", zorder=3))

    cutoff = 2.0 * a + 0.5 / kappa
    tmp = tempfile.mkdtemp(prefix="constriction_frames_")
    frame = 0
    for step in range(1, n_steps + 1):
        _step_constriction(pos_off, p_off, constr, rng_off)
        _step_constriction(pos_on, p_on, constr, rng_on)
        if step % stride != 0 and step != n_steps:
            continue
        scats[0].set_offsets(pos_off)
        scats[0].set_color(cluster_colors(pos_off, p_off, cutoff))
        scats[1].set_offsets(pos_on)
        scats[1].set_color(cluster_colors(pos_on, p_on, cutoff))
        time_text.set_text(
            f"Constricted channel, t = {step * dt:.1f}  |  grey = single,"
            " colour = in an aggregate  |  throat between x = 40 and 60"
        )
        fig.savefig(os.path.join(tmp, f"frame_{frame:04d}.png"))
        frame += 1
    print(f"rendered {frame} frames")
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(args.fps), "-i",
         os.path.join(tmp, "frame_%04d.png"), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "20", args.out],
        check=True, capture_output=True, text=True,
    )
    print("wrote", args.out, os.path.getsize(args.out), "bytes")


if __name__ == "__main__":
    main()
