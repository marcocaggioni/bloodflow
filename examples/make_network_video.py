"""Branching-network video: symmetric 1->2->4->8->4->2->1 ladder,
periodic trunk-to-trunk wrap, two panels (no attraction vs strong
attraction), same initial positions.

Panels are stacked vertically; the grey background is the channel
itself (capsules drawn as thick round-capped lines). Particles are
true size (radius 1 data unit); grey = single, colour = in an
aggregate (grid-based clustering, no wrap in the distance -- consistent
with the no-wrap pair forces).

Statistics tracked alongside: loop completions (wraps), time spent in
each generation (level 0 = trunk ... 3 = terminals), and per-frame
level populations.

Modes:
  video (default): stream frames to ffmpeg via image2pipe.
  pilot (--snapshots DIR): short run, phase snapshots plus checks
      (wall penetration, stuck particles, wrap counts, speed).

Usage:
    python3 examples/make_network_video.py [--out FILE]
        [--units 800] [--fps 25] [--seconds 24] [--snapshots DIR]
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
from bloodflow import core, network
from bloodflow.core import Params

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def cluster_colors_fast(pos, cutoff):
    """Grid-based clustering; singletons grey, clusters tab colours."""
    n = pos.shape[0]
    cell = cutoff
    grid = {}
    for i in range(n):
        grid.setdefault((int(pos[i, 0] // cell), int(pos[i, 1] // cell)),
                        []).append(i)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cut2 = cutoff * cutoff
    for i in range(n):
        cx, cy = int(pos[i, 0] // cell), int(pos[i, 1] // cell)
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for j in grid.get((gx, gy), ()):
                    if j <= i:
                        continue
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    if dx * dx + dy * dy < cut2:
                        ri, rj = find(i), find(j)
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
    ap.add_argument("--out", default="bloodflow-network.mp4")
    ap.add_argument("--units", type=float, default=800.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--snapshots", default=None,
                    help="pilot mode: save snapshots + checks to DIR")
    args = ap.parse_args()

    net = network.build_network()
    a, kappa, dt = 1.0, 2.0, 0.02
    n_steps = int(args.units / dt)
    n_frames = int(args.fps * args.seconds)
    stride = max(1, n_steps // n_frames)

    p_tmp = Params(N=1, a=a, H=200.0, L=net.L_total, eps=0.0,
                   kappa=kappa, dt=dt, seed=7)
    rng = np.random.default_rng(7)
    pos0, n, area = network.init_positions_network(net, p_tmp, rng)
    print(f"N={n}  area~{area:.0f}  steps={n_steps}  frames={n_frames} "
          f"(stride {stride})", flush=True)
    base = dict(N=n, a=a, H=200.0, L=net.L_total, eta=1.0, Umax=2.0,
                kappa=kappa, kT=0.0, dt=dt, seed=7)
    p_off = Params(eps=0.0, **base)
    p_on = Params(eps=0.64, **base)
    pos_off = pos0.copy()
    pos_on = pos0.copy()
    rng_off = np.random.default_rng(7)
    rng_on = np.random.default_rng(7)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), dpi=100)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.06,
                        hspace=0.35)
    # Channel background: capsules as thick round-capped lines. Points
    # per data unit from the axes geometry (no draw needed first).
    px_per_unit = 0.94 * 14.0 * 100.0 / net.L_total
    lw_per_width = px_per_unit * 72.0 / 100.0
    for ax in axes:
        ax.set_xlim(0, net.L_total)
        ax.set_ylim(0, 200.0)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=9)
        for s in net.segments:
            ax.plot([s.x0, s.x1], [s.y0, s.y1], color="#dddddd",
                    lw=s.width * lw_per_width, solid_capstyle="round",
                    zorder=0)
    axes[0].set_title("No attraction (eps = 0)", fontsize=13)
    axes[1].set_title("Strong attraction (eps = 0.64)", fontsize=13)
    axes[1].set_xlabel("x (flow direction, periodic trunk-to-trunk wrap)")
    axes[0].set_ylabel("y")
    axes[1].set_ylabel("y")
    time_text = fig.suptitle("", fontsize=13)
    size = (2.0 * px_per_unit * 72.0 / 100.0) ** 2
    scats = [ax.scatter([], [], s=size, edgecolors="none", zorder=2)
             for ax in axes]
    fig.canvas.draw()

    cutoff = 2.0 * a + 0.5 / kappa
    # Per-particle tracking.
    level_now = [network.dominant_level(pos_off, net),
                 network.dominant_level(pos_on, net)]
    level_since = [np.zeros(n), np.zeros(n)]
    level_dur = [{0: [], 1: [], 2: [], 3: []},
                 {0: [], 1: [], 2: [], 3: []}]
    loop_start = [np.zeros(n), np.zeros(n)]
    loop_times = [[], []]
    wraps = [0, 0]

    def track(step):
        """Update level durations (every 25 steps) and loop times."""
        t = step * dt
        if step % 25 != 0:
            return
        for k, pos in enumerate((pos_off, pos_on)):
            lv = network.dominant_level(pos, net)
            changed = lv != level_now[k]
            if changed.any():
                for old in range(4):
                    m = changed & (level_now[k] == old)
                    if m.any():
                        level_dur[k][old].extend(
                            (t - level_since[k][m]).tolist())
                level_now[k][changed] = lv[changed]
                level_since[k][changed] = t

    def render(step):
        t = step * dt
        for k, pos in enumerate((pos_off, pos_on)):
            scats[k].set_offsets(pos)
            scats[k].set_color(cluster_colors_fast(pos, cutoff))
        time_text.set_text(
            f"Branching network, t = {t:.1f}  |  loops completed: "
            f"{wraps[0]} (no attraction) / {wraps[1]} (attraction)  |  "
            "grey = single particle, colour = in an aggregate"
        )
        return t

    t_start = time.time()
    if args.snapshots is not None:
        os.makedirs(args.snapshots, exist_ok=True)
        snap_every = max(1, int(12.5 / dt))
        hist = []
        min_gap = 1e9
        for step in range(1, n_steps + 1):
            _, w0 = network.step_network(pos_off, p_off, net, rng_off, 0.0)
            _, w1 = network.step_network(pos_on, p_on, net, rng_on, 0.0)
            wraps[0] += int(w0.sum())
            wraps[1] += int(w1.sum())
            track(step)
            if step % 1250 == 0:
                hist.append((pos_off.copy(), pos_on.copy()))
                if len(hist) > 2:
                    hist.pop(0)
            if step % snap_every != 0 and step != n_steps:
                continue
            min_gap = min(min_gap,
                          float(network.inside_measure(pos_off, net).min()),
                          float(network.inside_measure(pos_on, net).min()))
            render(step)
            path = os.path.join(args.snapshots, f"snap_{step:06d}.png")
            fig.savefig(path)
            stuck = 0
            if len(hist) == 2:
                for pos_a, pos_b in ((hist[0][0], hist[1][0]),
                                     (hist[0][1], hist[1][1])):
                    d = np.hypot(pos_b[:, 0] - pos_a[:, 0],
                                 pos_b[:, 1] - pos_a[:, 1])
                    stuck += int((d < 1.0).sum())
            print(f"  t={step * dt:7.1f} wraps={wraps} min_gap={min_gap:.2f} "
                  f"stuck(<1 unit in 50)={stuck} -> {path}", flush=True)
        el = time.time() - t_start
        print(f"pilot done: {n_steps} steps in {el:.0f}s "
              f"({n_steps / el:.1f} steps/s/panel); full 40000-step run "
              f"projects to ~{2 * 40000 / (n_steps / el) / 60:.0f} min")
        return

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "image2pipe", "-vcodec", "png",
         "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "20", args.out],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    frame = 0
    rows = []
    for step in range(1, n_steps + 1):
        _, w0 = network.step_network(pos_off, p_off, net, rng_off, 0.0)
        _, w1 = network.step_network(pos_on, p_on, net, rng_on, 0.0)
        wraps[0] += int(w0.sum())
        wraps[1] += int(w1.sum())
        for k, w in enumerate((w0, w1)):
            if w.any():
                t = step * dt
                loop_times[k].extend((t - loop_start[k][w]).tolist())
                loop_start[k][w] = t
        track(step)
        if step % stride != 0 and step != n_steps:
            continue
        t = render(step)
        lv0 = np.bincount(level_now[0], minlength=4)
        lv1 = np.bincount(level_now[1], minlength=4)
        rows.append((t, wraps[0], wraps[1], *lv0.tolist(), *lv1.tolist()))
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
    with open("network_frame_stats.csv", "w") as f:
        f.write("t,wraps_off,wraps_on,lv0_off,lv1_off,lv2_off,lv3_off,"
                "lv0_on,lv1_on,lv2_on,lv3_on\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    for k, name in enumerate(("no attraction", "attraction")):
        lt = np.array(loop_times[k])
        print(f"  loop times ({name}): n={len(lt)}", end="")
        if len(lt):
            print(f" mean={lt.mean():.1f} median={np.median(lt):.1f} "
                  f"p95={np.percentile(lt, 95):.1f}")
        else:
            print(" (none completed)")
        for lv in range(4):
            d = np.array(level_dur[k][lv])
            if len(d):
                print(f"    level {lv} durations ({name}): n={len(d)} "
                      f"mean={d.mean():.1f}")


if __name__ == "__main__":
    main()
