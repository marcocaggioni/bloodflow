# Bloodflow

A toy model for a simple question: **can red blood cell aggregation turn
Poiseuille flow into plug flow, and does that cut the long residence time
of cells near the vessel wall?**

Rigid disks flow in a 2D channel under an imposed Poiseuille profile.
Turn on a short-range attraction between the disks and they aggregate;
the aggregates move as a plug, so slow near-wall particles get carried
along instead of lingering. The strength of the effect is measured through
the distribution of per-particle transit times.

## Model

Overdamped Brownian dynamics of N disks of radius `a` in a channel of
width `H` (walls at y = 0 and y = H), periodic in x with period `L`:

```
dr_i/dt = U(y_i) x-hat + (1/zeta) * sum_j F_ij + thermal noise
zeta = 6*pi*eta*a
```

The 3D Stokes drag for spheres constrained to a plane avoids the 2D
Stokes paradox. The imposed Poiseuille profile is

```
U(y) = 4*Umax*(y/H)*(1 - y/H),   Umax = dP*H^2/(8*eta*L)
```

so the pressure drop enters through `Umax` (or equivalently the wall
shear rate `gamma_w = 4*Umax/H`).

Forces:

- **WCA repulsion** (repulsive Lennard-Jones, cutoff at 2^(1/6)*sigma,
  sigma = 2a) between particles, and the same form against both walls.
- **Morse attraction**,
  `U(r) = eps*(exp(-2*kappa*(r-2a)) - 2*exp(-kappa*(r-2a)))`,
  with tunable depth `eps` and inverse range `kappa`. Minimum `-eps` at
  contact. (`eps = 0` = no aggregation.)
- **Thermal noise** (Euler-Maruyama, diffusion coefficient `D = kT/zeta`);
  default `kT = 0` for deterministic runs.

Integration is explicit Euler with a linked-cell neighbour list (padded
grid, vectorised over cell pairs), so a few thousand particles run fast.

## Parameters and dimensionless groups

| Parameter | Meaning |
|---|---|
| `a`, `H`, `L` | particle radius, channel width, channel period |
| `eta`, `Umax` | plasma viscosity, centreline velocity (from pressure drop) |
| `eps`, `kappa` | attraction depth, inverse attraction range |
| `N` | particle number (sets the area fraction) |
| `kT`, `dt`, `seed` | thermal energy, time step, RNG seed |

| Group | Meaning |
|---|---|
| `a/H` | confinement: large = vessel barely bigger than a cell (mouse-like), small = artery much bigger than a cell |
| `phi = N*pi*a^2/(L*H)` | area fraction (2D hematocrit) |
| `Mason = (eps*kappa)/(eta*gamma_w*a^2)` | attraction force vs hydrodynamic force; >> 1 means aggregates survive the shear |
| `Pe` | Peclet number (advection vs diffusion); huge for real RBCs, so noise is kept tiny |

## Install

```bash
pip install numpy
pip install -e .        # or just run from this directory
```

Only `numpy` is required at runtime.

## Run the example sweep

```bash
python examples/sweep_attraction.py
```

Runs a short sweep over attraction strength at fixed flow and prints the
transit-time summary per `eps`. Expected output (takes ~30 s):

```
channel H=16.0, L=40.0, a=1.0 (a/H=0.062), area fraction phi=0.29, Umax=1.0, wall shear=0.250, dt=0.02
  eps   Mason    mean  median     p95  frac>3x  compl  wall_t  clust  in_cl
  0.0     0.0    66.6    48.4   137.9    0.000   1.00    34.8   1.50   0.45
  0.5     4.0    60.1    47.4   123.2    0.000   1.00    19.6   6.00   0.93
  2.0    16.0    52.2    49.5    75.0    0.000   1.00     4.4  15.00   0.98
```

Reading: as attraction grows, the transit-time tail shrinks (p95 138 → 75),
the mean near-wall residence time collapses (35 → 4), and clusters grow
(mean size 1.5 → 15). `frac>3x` is the fraction of completed transits
slower than 3x the mean; `compl` is the fraction of particles that
completed a transit (the rest are censored at the end of the run).

## Videos

- [Pulsatile flow](videos/bloodflow-pulsatile.mp4) — sinusoidal pressure drop, no reversal: 12 beats at the same mean flow as the steady run, with a waveform strip marking beat phase and Mason number. The plug breathes but survives (phase-averaged cluster fraction ~0.92 at systole vs ~0.97 at diastole).

Notebook versions of all five video scripts live in [`examples/videos.ipynb`](examples/videos.ipynb): Poiseuille vs plug, constriction, pulsatile, the stacked attraction ladder, and the branching network. Each section renders its frames and encodes the MP4 in the browser (see below); the runs are trimmed relative to the scripts — fewer seconds, steps, or beats — so they finish in a few minutes in the browser, and each section says exactly how.

## Export an MP4 in the browser

The notebooks can encode their own MP4s without leaving the browser.
JupyterLite ships no ffmpeg binary, so the export cells in `demo.ipynb`
and the `export_mp4` helper in `examples/videos.ipynb` use
[ffmpeg.wasm](https://ffmpegwasm.netlify.app/) — FFmpeg compiled to
WebAssembly — vendored in this repo under `vendor/ffmpeg/` and served
from the same site as the notebook (no CDN; the CDN version failed live
because the browser blocks the cross-origin worker file), driven from
Python through Pyodide's JavaScript bridge. The first run downloads
~31 MB of WASM; the browser caches it afterwards, and in
`examples/videos.ipynb` the loaded encoder is reused for all five
videos. The single-threaded core is used on purpose: GitHub Pages sends
no cross-origin-isolation headers, so the multi-threaded core cannot
load here.

In `demo.ipynb`, run the two cells (render frames, then encode): they
render the eps = 2 run as 90 PNG frames (3 s at 30 fps), encode them
with libx264 (`-pix_fmt yuv420p`), and save `bloodflow-export.mp4`
next to the notebook — it appears in the file browser (right-click →
Download) and plays inline. The temporary frames are deleted from
ffmpeg's in-memory filesystem after encoding; only the intermediates
are cleared, the simulation data stays in the notebook.

Two things the export depends on, found the hard way:
- The core must be the **ESM build** (`@ffmpeg/core` `dist/esm`).
  The wrapper spawns a *module* worker, which has no `importScripts`,
  so it falls back to importing the core as an ES module — and the UMD
  build exports nothing, so `load()` hangs forever with no error.
- Every Python value handed to ffmpeg's JS methods must cross with
  `pyodide.ffi.to_js` first. Otherwise it arrives as a PyProxy, and
  `postMessage` cannot clone a PyProxy into ffmpeg's worker
  (`DataCloneError`). That covers the PNG bytes in `writeFile` and the
  argument list in `exec`.

## Run the validation test

```bash
python tests/test_poiseuille.py   # plain python, no pytest needed
python -m pytest tests/           # also pytest-compatible
```

With attraction off and no noise, every particle is advected at exactly
the Poiseuille velocity of its height, so its transit time must equal
`L / U(y)`; the test checks this, plus that the cell-list forces match a
direct O(N^2) computation.

## Layout

```
bloodflow/
  potentials.py   WCA, Morse, wall forces
  core.py         parameters, cell-list forces, integrator, Result
  constriction.py constricted channel (throat + ramps)
  network.py      branching-network geometry and stepping
  analysis.py     transit times, wall residence, clustering
examples/
  sweep_attraction.py
  make_video.py, make_constriction_video.py, make_pulsatile_video.py,
  stacked_attraction_video.py, make_network_video.py
                  video scripts (need matplotlib + system ffmpeg)
  videos.ipynb    notebook versions of the five video scripts;
                  each section renders its frames and encodes the MP4
                  in the browser with ffmpeg.wasm
  snapshot_figure.py, residence_time_analysis.py,
  particle_age_analysis.py, constriction.py
                  analysis scripts
demo.ipynb        the sweep, run in the browser, with in-browser MP4 export
tests/
  test_poiseuille.py
vendor/ffmpeg/    ffmpeg.wasm 0.12.15 + single-threaded core 0.12.10 (ESM),
                  served same-origin for the in-browser MP4 export
videos/           pre-rendered videos (pulsatile)
```

## Deliberate limitations (roadmap, not bugs)

- 2D disks, rigid, free-draining (no hydrodynamic interactions).
- Steady flow only; the heartbeat version (pulsatile pressure drop, gel
  forming in diastole and plug moving in systole) is the natural next step.
- Morse attraction stands in for depletion; an Asakura-Oosawa potential
  (range = depletant size, depth ∝ depletant concentration) would make
  the depletion hypothesis literal.
- Later: deformable particles, 3D, real hydrodynamics.
