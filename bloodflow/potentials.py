# Pair potentials for the Bloodflow toy model.
#
# All forces are returned as the scalar s = -dU/dr, so that the force on
# particle i due to particle j is  F = s * (r_i - r_j) / r.
# Positive s = repulsion, negative s = attraction.

import numpy as np

_WCA_C = 2.0 ** (1.0 / 6.0)


def wca_cutoff(sigma):
    """Cutoff of the WCA (purely repulsive Lennard-Jones) potential."""
    return _WCA_C * sigma


def wca_force_scalar(r, sigma, eps_rep):
    """s = -dU/dr for the WCA potential. Valid for r < wca_cutoff(sigma)."""
    sr = sigma / r
    sr6 = sr ** 6
    sr12 = sr6 * sr6
    return (24.0 * eps_rep / r) * (2.0 * sr12 - sr6)


def morse_cutoff(a, kappa):
    """Cutoff beyond which the Morse attraction is negligible (6 decay lengths)."""
    return 2.0 * a + 6.0 / kappa


def morse_force_scalar(r, a, eps, kappa):
    """s = -dU/dr for the Morse attraction.

    U(r) = eps * (exp(-2*kappa*(r-2a)) - 2*exp(-kappa*(r-2a))).
    Minimum -eps at contact (r = 2a); attractive for r > 2a, repulsive for
    r < 2a (the WCA term dominates there anyway).
    """
    x = r - 2.0 * a
    return 2.0 * kappa * eps * (np.exp(-2.0 * kappa * x) - np.exp(-kappa * x))


def wall_force_scalar(d, a, eps_rep):
    """s = -dU/dd for the WCA wall repulsion (scalar- or array-friendly).

    d is the distance from the particle centre to the wall; contact is at
    d = a. Returns 0 beyond the wall cutoff. The returned scalar is the
    force magnitude pushing *away* from the wall.
    """
    d = np.asarray(d, dtype=float)
    cut = _WCA_C * a
    safe_d = np.where(d > 0.0, d, 1.0)
    ar = a / safe_d
    ar6 = ar ** 6
    return np.where(
        (d < cut) & (d > 0.0),
        (24.0 * eps_rep / safe_d) * (2.0 * ar6 * ar6 - ar6),
        0.0,
    )
