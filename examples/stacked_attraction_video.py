"""Stacked attraction ladder, straight channel, long channel.

Four panels stacked vertically in one video, same starting positions
(random uniform), attraction increasing top to bottom:
    eps = 0.00 (pure Poiseuille), 0.32, 0.64, 1.28
At H = 100, Umax = 2 these are Mason numbers 0, 8, 16, 32; eps = 0.64 is
the attraction used in the first Poiseuille-vs-plug video.

Channel: H = 100, L = 400 (4x the first video's length), phi = 0.25,
so N = 3183. Run: 60,000 steps = 1,200 time units (10x the first
video's 6,000 steps). Video: 24 s at 25 fps = 600 frames, one every
100 steps. No equilibration: frame one is the random uniform start,
and the age clock runs over the whole simulation.

Colour rule (per the request): only groups of MORE than 6 particles
are coloured; groups of 6 or fewer are grey like singletons. Clustering
uses a cell-list union-find (the old O(N^2) loop is far too slow at
N = 3183).

Outputs (besides the video):
  - CSV: case, particle, age, n_wraps (age = time since last wrap at
    the end of the run; never-wrapped particles get the full 1,200)
  - PNG: age histograms per case, counts on a log scale, 25 bins
  - Printed age summaries per case

Usage:
    python3 examples/stacked_attraction_video.py [--out FILE]
Requires matplotlib and ffmpeg on PATH. Frames stream into ffmpeg;
no temp files (an earlier version wrote frames to /tmp, a 512 MB
tmpfs, and died on "No space left on device").
"""

import argparse
import csv
import io
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bloodflow import core
from bloodflow.core import Params

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def cluster_colors_fast(pos, p, cutoff, min_size=7):
    """Cell-list clustering; groups of >= min_size coloured, rest grey."""
    n = pos.shape[0]
    cell = cutoff
    nx = max(1, int(p.L // cell))
    ny = max(1, int(p.H // cell))
    ix = (np.floor(pos[:, 0] / cell).astype(int)) % nx
    iy = np.clip(np.floor(pos[:, 1] / cell).astype(int), 0, ny - 1)
    buckets = {}
    for i in range(n):
        buckets.setdefault((int(ix[i]), int(iy[i])), []).append(i)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cut2 = cutoff * cutoff
    for (cx, cy), members in buckets.items():
        for dcx in (-1, 0, 1):
            for dcy in (-1, 0, 1):
                key = ((cx + dcx) % nx, cy + dcy)
                others = buckets.get(key)
                if not others:
                    continue
                for i in members:
                    xi, yi = pos[i, 0], pos[i, 1]
                    for j in others:
                        if j <= i:
                            continue
                        dx = xi - pos[j, 0]
                        dx -= p.L * np.round(dx / p.L)
                        dy = yi - pos[j, 1]
                        if dx * dx + dy * dy < cut2:
                            ri, rj = find(i), find(j)
                            if ri != rj:
                                parent[ri] = rj
    labels = np.array([find(i) for i in range(n)])
    _, counts = np.unique(labels, return_counts=True)
    size_of = {lab: int(c) for lab, c in zip(np.unique(labels), counts)}
    ranked = sorted(
        (lab for lab in np.unique(labels) if size_of[lab] >= min_size),
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


def summarize(age):
    mean = float(np.mean(age))
    q25, q75 = np.percentile(age, [25, 75])
    return {
        "mean": mean,
        "std": float(np.std(age)),
        "cv": float(np.std(age) / mean),
        "median": float(np.median(age)),
        "iqr": float(q75 - q25),
        "p95": float(np.percentile(age, 95)),
        "frac_at_max_age": float(np.mean(age >= age.max() - 1e-9)),
        "max": float(age.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stacked-attraction-video.mp4")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--age-prefix", default="stacked-age")
    args = ap.parse_args()

    a, H, L, phi = 1.0, 100.0, 400.0, 0.25
    n = int(phi * L * H / (np.pi * a * a))
    Umax, dt, kappa = 2.0, 0.02, 2.0
    eps_list = [0.0, 0.32, 0.64, 1.28]
    n_frames = int(args.fps * args.seconds)
    stride = max(1, args.steps // n_frames)

    base = dict(N=n, a=a, H=H, L=L, eta=1.0, Umax=Umax,
                kappa=kappa, kT=0.0, dt=dt, seed=7)
    params = [Params(eps=e, **base) for e in eps_list]
    names = [f"eps-{e:g}" for e in eps_list]
    print(f"N={n}  H={H} L={L}  Mason numbers: "
          f"{[round(p.mason_number(), 1) for p in params]}  "
          f"steps={args.steps} ({args.steps * dt:.0f} time units)  "
          f"frames={n_frames} (stride {stride})")

    rng = np.random.default_rng(7)
    pos0 = core.init_positions(params[0], rng)
    positions = [pos0.copy() for _ in params]
    rngs = [np.random.default_rng(7) for _ in params]
    last_wrap = [np.zeros(n, dtype=int) for _ in params]
    n_wraps = [np.zeros(n, dtype=int) for _ in params]

    fig, axes = plt.subplots(4, 1, figsize=(16, 16), dpi=100)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.03,
                        hspace=0.45)
    scats = []
    for ax, p, name in zip(axes, params, names):
        ax.set_xlim(0, L)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_title(f"eps = {p.eps:g}  (Mason {p.mason_number():.0f})",
                     fontsize=13)
        ax.tick_params(labelsize=9)
        if ax is not axes[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("x (flow direction)")
    axes[0].set_ylabel("y (channel width)")
    time_text = fig.suptitle("", fontsize=13)
    fig.canvas.draw()
    p0 = axes[0].transData.transform((0.0, 0.0))
    p1 = axes[0].transData.transform((2.0 * a, 0.0))
    r_px = abs(p1[0] - p0[0]) / 2.0
    size = (2.0 * r_px * 72.0 / fig.dpi) ** 2
    for ax in axes:
        scats.append(ax.scatter([], [], s=size, edgecolors="none"))

    cutoff = 2.0 * a + 0.5 / kappa
    # Stream frames straight into ffmpeg (image2pipe): writing them to
    # /tmp killed the first full run, since /tmp is a 512 MB tmpfs.
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-framerate", str(args.fps),
         "-f", "image2pipe", "-vcodec", "png", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.out],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    frame = 0
    for step in range(1, args.steps + 1):
        for k, p in enumerate(params):
            x_old = positions[k][:, 0].copy()
            dx = core._step(positions[k], p, rngs[k], 0.0)
            wrapped = (x_old + dx >= p.L) | (x_old + dx < 0.0)
            last_wrap[k][wrapped] = step
            n_wraps[k][wrapped] += 1
        if step % stride != 0 and step != args.steps:
            continue
        for k, p in enumerate(params):
            scats[k].set_offsets(positions[k])
            scats[k].set_color(cluster_colors_fast(positions[k], p, cutoff))
        time_text.set_text(
            f"Straight channel, t = {step * dt:.0f}  |  grey = single or in"
            " a group of 6 or fewer, colour = in a group of 7+"
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        ffmpeg.stdin.write(buf.getvalue())
        frame += 1
    ffmpeg.stdin.close()
    ffmpeg.wait()
    if ffmpeg.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {ffmpeg.returncode}")
    print(f"rendered {frame} frames")
    print("wrote", args.out, os.path.getsize(args.out), "bytes")

    # Age analysis at the end of the run.
    results = {}
    for name, p, lw, nw in zip(names, params, last_wrap, n_wraps):
        age = (args.steps - lw) * dt
        results[name] = (age, nw)
        print(f"\n[{name}]")
        for k, v in summarize(age).items():
            print(f"  {k}: {v:.4g}")

    csv_path = f"{args.age_prefix}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "particle", "age", "n_wraps"])
        for name, (age, nw) in results.items():
            for i in range(n):
                w.writerow([name, i, f"{age[i]:.4f}", int(nw[i])])
    print("\nwrote", csv_path)

    fig2, axes2 = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    for ax, name, color in zip(axes2, names,
                               ["#4a7bb5", "#6a9e6a", "#d95f2b", "#8e44ad"]):
        age = results[name][0]
        ax.hist(age, bins=25, alpha=0.6, color=color)
        ax.axvline(np.median(age), color=color, linestyle="--")
        ax.set_yscale("log")
        ax.set_ylim(0.5, None)
        ax.set_ylabel("Count (log)")
        ax.set_title(f"{name}  (median {np.median(age):.1f})")
    axes2[-1].set_xlabel("Particle age at end of run (time units)")
    fig2.suptitle("Particle-age distribution at end of run (1,200 time units)")
    png_path = f"{args.age_prefix}.png"
    fig2.savefig(png_path, dpi=130, bbox_inches="tight")
    print("wrote", png_path)


if __name__ == "__main__":
    main()
