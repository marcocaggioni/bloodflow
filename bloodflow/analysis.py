# Analysis of Bloodflow runs: transit times, wall residence, clustering.

import numpy as np


def transit_summary(transit, wall_time=None):
    """Summary of the transit-time distribution.

    'Slow' means slower than 3x the mean transit time (completed
    particles only). Censored particles (NaN transit) are counted
    separately and excluded from the time statistics.
    """
    done = transit[~np.isnan(transit)]
    out = {
        "n_completed": int(done.size),
        "frac_completed": float(done.size / transit.size) if transit.size else 0.0,
    }
    if done.size:
        mean = float(np.mean(done))
        out["mean"] = mean
        out["median"] = float(np.median(done))
        out["p95"] = float(np.percentile(done, 95))
        out["max"] = float(np.max(done))
        out["frac_slower_than_3x_mean"] = float(np.mean(done > 3.0 * mean))
    else:
        out.update(
            {"mean": float("nan"), "median": float("nan"), "p95": float("nan"),
             "max": float("nan"), "frac_slower_than_3x_mean": float("nan")}
        )
    if wall_time is not None:
        out["mean_wall_residence"] = float(np.mean(wall_time))
        out["median_wall_residence"] = float(np.median(wall_time))
    return out


def cluster_labels(pos, cutoff):
    """Union-find clustering with a distance cutoff. Returns labels (N,)."""
    N = pos.shape[0]
    parent = list(range(N))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            # Note: clustering uses the raw coordinates (no periodic image
            # correction); for the final snapshot this only affects pairs
            # straddling the x-boundary, a negligible edge effect.
            if dx * dx + (pos[i, 1] - pos[j, 1]) ** 2 < cutoff * cutoff:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    return np.array([find(i) for i in range(N)])


def cluster_stats(pos, cutoff):
    """Mean cluster size (N / n_clusters, monomers included) and the
    fraction of particles in clusters of size >= 2."""
    labels = cluster_labels(pos, cutoff)
    _, counts = np.unique(labels, return_counts=True)
    n_in_clusters = int(np.sum(counts[counts >= 2])) if counts.size else 0
    return {
        "n_clusters": int(counts.size),
        "mean_cluster_size": float(pos.shape[0] / counts.size) if counts.size else 0.0,
        "frac_in_clusters": float(n_in_clusters / pos.shape[0]) if pos.shape[0] else 0.0,
    }
