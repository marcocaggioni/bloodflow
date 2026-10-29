"""Pulsatile-flow video: same side-by-side setup as make_video.py
(H = 100, eps 0 vs 0.64, same initial positions), but the driving flow
beats: Umax(t) = Umean * (1 + amp*sin(2*pi*t/T)).

The bottom strip shows the driving waveform over one beat with a
marker at the current phase and the instantaneous Mason number of the
attraction panel, so aggregate breakup/reformation can be read
against the heartbeat.

Physics question: with mean Ma = 16, systole peak shear (1.8x mean)
drops Ma to ~9 and diastole trough (0.2x mean) raises it to ~80, so
the prediction is that the plug breathes but survives. The recorded
cluster fraction vs beat phase tests that.

Modes:
  video (default): stream frames to ffmpeg via image2pipe (no temp
      files), 25 fps.
  pilot (--snapshots DIR): run a few beats, save one snapshot per 1/8
      beat, print cluster fraction vs phase and timing.

Usage:
    python3 examples/make_pulsatile_video.py [--out FILE] [--beats N]
        [--fps 25] [--seconds 24] [--amp 0.8] [--period 100]
        [--snapshots DIR]
Requires matplotlib and ffmpeg on PATH.
"""

import argparse
import io
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bloodflow import core
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


def cluster_fraction(pos, p, cutoff):
    """Fraction of particles in a cluster of size >= 2."""
    n = pos.shape[0]
    dx = pos[:, 0][:, None] - pos[:, 0][None, :]
    dx -= p.L * np.round(dx / p.L)
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    adj = (dx * dx + dy * dy) < cutoff * cutoff
    np.fill_diagonal(adj, False)
    return float(adj.any(axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bloodflow-pulsatile.mp4")
    ap.add_argument("--beats", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--amp", type=float, default=0.8)
    ap.add_argument("--period", type=float, default=100.0)
    ap.add_argument("--snapshots", default=None,
                    help="pilot mode: save phase snapshots to DIR")
    args = ap.parse_args()

    a, H, L, phi = 1.0, 100.0, 100.0, 0.25
    n = int(phi * L * H / (np.pi * a * a))
    Umean, dt, kappa = 2.0, 0.02, 2.0
    n_steps = int(args.beats * args.period / dt)
    n_frames = int(args.fps * args.seconds)
    stride = max(1, n_steps // n_frames)

    base = dict(N=n, a=a, H=H, L=L, eta=1.0, Umax=Umean,
                kappa=kappa, kT=0.0, dt=dt, seed=7,
                pulse_amp=args.amp, pulse_period=args.period)
    p_off = Params(eps=0.0, **base)
    p_on = Params(eps=0.64, **base)
    print(f"N={n}  H={H}  mean Mason(on)={p_on.mason_number():.1f}  "
          f"amp={args.amp}  period={args.period}  steps={n_steps}  "
          f"frames={n_frames} (stride {stride})", flush=True)

    rng = np.random.default_rng(7)
    pos_off = core.init_positions(p_off, rng)
    pos_on = pos_off.copy()
    rng_off = np.random.default_rng(7)
    rng_on = np.random.default_rng(7)

    fig = plt.figure(figsize=(12, 6.8), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[5.2, 1.2], hspace=0.45,
                          wspace=0.08, left=0.04, right=0.98,
                          top=0.88, bottom=0.08)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    axw = fig.add_subplot(gs[1, :])
    titles = ["No attraction (eps = 0)", "Strong attraction (eps = 0.64)"]
    scats = []
    for ax, title in zip((ax0, ax1), titles):
        ax.set_xlim(0, L)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("x (flow direction)")
        ax.tick_params(labelsize=9)
    ax0.set_ylabel("y (channel width)")
    ax1.set_yticklabels([])
    time_text = fig.suptitle("", fontsize=13)
    p0 = ax0.transData.transform((0.0, 0.0))
    p1 = ax0.transData.transform((2.0 * a, 0.0))
    r_px = abs(p1[0] - p0[0]) / 2.0
    size = (2.0 * r_px * 72.0 / fig.dpi) ** 2
    for ax in (ax0, ax1):
        scats.append(ax.scatter([], [], s=size, edgecolors="none"))

    # Waveform strip: one beat, static curve + moving phase marker.
    phase = np.linspace(0, args.period, 400)
    wave = Umean * (1 + args.amp * np.sin(2 * np.pi * phase / args.period))
    axw.plot(phase, wave, color="black", lw=1.5)
    axw.axhline(Umean, color="grey", lw=0.8, ls="--")
    axw.set_xlim(0, args.period)
    axw.set_ylim(Umean * (1 - args.amp) * 0.8, Umean * (1 + args.amp) * 1.15)
    axw.set_xlabel("phase within beat (time units)")
    axw.set_ylabel("Umax")
    axw.tick_params(labelsize=9)
    marker, = axw.plot([], [], "ro", ms=6)
    wave_text = axw.set_title("", fontsize=11, loc="left")

    cutoff = 2.0 * a + 0.5 / kappa
    fig.canvas.draw()

    def render(step):
        """Update all artists for simulation step `step`. Returns
        (time, phase_fraction, cluster_fraction_on)."""
        t = step * dt
        ph = (t % args.period) / args.period
        scats[0].set_offsets(pos_off)
        scats[0].set_color(cluster_colors(pos_off, p_off, cutoff))
        scats[1].set_offsets(pos_on)
        scats[1].set_color(cluster_colors(pos_on, p_on, cutoff))
        beat = int(t // args.period) + 1
        time_text.set_text(
            f"Pulsatile flow, t = {t:.1f}  (beat {beat})  |  "
            "grey = single particle, colour = in an aggregate"
        )
        u_now = core.pulsatile_umax(p_on, t)
        gamma_now = 4.0 * u_now / H
        ma_now = (p_on.eps * p_on.kappa) / (gamma_now * a * a)
        cos_ph = np.cos(2 * np.pi * ph)
        sin_ph = np.sin(2 * np.pi * ph)
        if cos_ph > 0.05:
            phase_label = "systole, flow rising"
        elif cos_ph < -0.05:
            phase_label = "diastole, flow falling"
        elif sin_ph > 0:
            phase_label = "peak systole"
        else:
            phase_label = "end diastole"
        marker.set_data([(t % args.period)], [u_now])
        wave_text.set_text(
            f"Driving flow: Umax = {u_now:.2f}, instantaneous Mason = "
            f"{ma_now:.0f} ({phase_label})"
        )
        frac = cluster_fraction(pos_on, p_on, cutoff)
        return t, ph, frac

    t_start = time.time()
    if args.snapshots is not None:
        os.makedirs(args.snapshots, exist_ok=True)
        snap_every = int(args.period / dt / 8)  # 8 snapshots per beat
        stats = []
        for step in range(1, n_steps + 1):
            core._step(pos_off, p_off, rng_off, 0.0, t=(step - 1) * dt)
            core._step(pos_on, p_on, rng_on, 0.0, t=(step - 1) * dt)
            if step % snap_every != 0 and step != n_steps:
                continue
            t, ph, frac = render(step)
            path = os.path.join(args.snapshots, f"snap_{step:06d}.png")
            fig.savefig(path)
            stats.append((t, ph, frac))
            print(f"  t={t:7.1f} phase={ph:.3f} cluster_frac={frac:.3f}"
                  f" -> {path}", flush=True)
        el = time.time() - t_start
        print(f"pilot done: {n_steps} steps in {el:.0f}s "
              f"({n_steps / el:.0f} steps/s); full {int(12 * args.period / dt)}"
              f"-step run projects to ~{12 * args.period / dt / (n_steps / el) / 60:.0f} min of stepping")
        with open(os.path.join(args.snapshots, "phase_stats.csv"), "w") as f:
            f.write("t,phase,cluster_fraction\n")
            for t, ph, frac in stats:
                f.write(f"{t:.2f},{ph:.4f},{frac:.4f}\n")
        return

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "image2pipe", "-vcodec", "png",
         "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "20", args.out],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    frame = 0
    stats = []
    for step in range(1, n_steps + 1):
        core._step(pos_off, p_off, rng_off, 0.0, t=(step - 1) * dt)
        core._step(pos_on, p_on, rng_on, 0.0, t=(step - 1) * dt)
        if step % stride != 0 and step != n_steps:
            continue
        t, ph, frac = render(step)
        stats.append((t, ph, frac))
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        proc.stdin.write(buf.getvalue())
        frame += 1
        if frame % 100 == 0:
            print(f"  frame {frame}/{n_frames} (t={t:.0f})", flush=True)
    proc.stdin.close()
    proc.wait()
    el = time.time() - t_start
    print(f"wrote {args.out} ({frame} frames, {el:.0f}s)",
          os.path.getsize(args.out), "bytes")
    with open("pulsatile_phase_stats.csv", "w") as f:
        f.write("t,phase,cluster_fraction\n")
        for t, ph, frac in stats:
            f.write(f"{t:.2f},{ph:.4f},{frac:.4f}\n")


if __name__ == "__main__":
    main()
