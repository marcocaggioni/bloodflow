"""Render a side-by-side video: Poiseuille flow without attraction (left)
vs with strong attraction (right), same initial positions.

Physics story: without attraction, near-wall particles crawl (Poiseuille
parabola, velocity -> 0 at the walls) while centre particles race ahead.
With attraction, particles clump into aggregates that move together as a
plug, cutting off the slow near-wall tail.

Scaled parameters: particle radius a = 1, channel width H = 100, so
a/H = 0.01 -- a large-vessel regime (channel 50 cell diameters wide). A
mouse-like vessel would be a/H ~ 0.5, i.e. barely wider than a cell.
In real units, a = 4 um (red blood cell radius) makes H = 400 um here;
the ratio a/H, not the absolute size, is what the toy resolves.

Usage:
    python3 examples/make_video.py [--out FILE] [--seconds S]
Requires matplotlib and ffmpeg on PATH. Frames go to a temp dir.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bloodflow import core
from bloodflow.core import Params

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def cluster_colors(pos, p, cutoff):
    """Vectorised clustering; singletons grey, clusters tab20 colours."""
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
    # Rank clusters by size so the biggest get the most distinct colours.
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
    ap.add_argument("--out", default="bloodflow-poiseuille-vs-plug.mp4")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()

    a, H, L, phi = 1.0, 100.0, 100.0, 0.25
    n = int(phi * L * H / (np.pi * a * a))
    Umax, dt, kappa = 2.0, 0.02, 2.0
    n_steps = 6000
    n_frames = int(args.fps * args.seconds)
    stride = max(1, n_steps // n_frames)

    base = dict(N=n, a=a, H=H, L=L, eta=1.0, Umax=Umax,
                kappa=kappa, kT=0.0, dt=dt, seed=7)
    p_off = Params(eps=0.0, **base)
    p_on = Params(eps=0.64, **base)  # Mason ~ 16 at the wall shear rate
    print(f"N={n}  H={H} (a/H={a / H})  Mason(on)={p_on.mason_number():.1f}")

    rng = np.random.default_rng(7)
    pos_off = core.init_positions(p_off, rng)
    pos_on = pos_off.copy()
    rng_off = np.random.default_rng(7)
    rng_on = np.random.default_rng(7)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), dpi=100)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.86, bottom=0.10,
                        wspace=0.08)
    titles = ["No attraction (eps = 0)", "Strong attraction (eps = 0.64)"]
    scats = []
    for ax, title in zip(axes, titles):
        ax.set_xlim(0, L)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("x (flow direction)")
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("y (channel width)")
    axes[1].set_yticklabels([])
    time_text = fig.suptitle("", fontsize=13)
    fig.canvas.draw()
    # Marker area (points^2) matching a disk of radius a in data units.
    p0 = axes[0].transData.transform((0.0, 0.0))
    p1 = axes[0].transData.transform((2.0 * a, 0.0))
    r_px = abs(p1[0] - p0[0]) / 2.0
    size = (2.0 * r_px * 72.0 / fig.dpi) ** 2
    for ax in axes:
        scats.append(ax.scatter([], [], s=size, edgecolors="none"))

    cutoff = 2.0 * a + 0.5 / kappa
    tmp = tempfile.mkdtemp(prefix="bloodflow_frames_")
    frame = 0
    for step in range(1, n_steps + 1):
        core._step(pos_off, p_off, rng_off, 0.0)
        core._step(pos_on, p_on, rng_on, 0.0)
        if step % stride != 0 and step != n_steps:
            continue
        scats[0].set_offsets(pos_off)
        scats[0].set_color(cluster_colors(pos_off, p_off, cutoff))
        scats[1].set_offsets(pos_on)
        scats[1].set_color(cluster_colors(pos_on, p_on, cutoff))
        time_text.set_text(
            f"Poiseuille flow, t = {step * dt:.1f}  |  grey = single particle,"
            " colour = in an aggregate"
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
