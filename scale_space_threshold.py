"""
Scale-space meaningful-modes detector for finding binarization thresholds.

Intended use: as a drop-in replacement for KDE/histogram threshold-finding
in DTT log-UMI binarization. The pipeline consumes the LEFTMOST meaningful
valley via `scale_space_threshold`, which correctly identifies the boundary
between the zero spike and the signal mode in all tested data regimes.

`scale_space_meaningful_valleys` returns the full list of valleys, suitable
for multi-mode segmentation. On the synthetic test cases (unimodal, bimodal,
trimodal, bimodal-with-spurious-bump, two-modes-close, tight-zero-spike) it
returns exactly the expected population-separator valleys — no extra intra-
mode artifacts.

Based on Gilles & Heal 2014 (arxiv:1401.2686), "A parameterless scale-space
approach to find meaningful modes in histograms," adapted for DTT log-UMI
data with the following modifications:

  1. log-density tracking, not raw counts (handles the order-of-magnitude
     dynamic range between zero spike and signal mode).
  2. Plateau-aware minimum detection via `scipy.signal.find_peaks(-D)` —
     necessary because the smoothed density between zero spike and signal
     mode is often exactly zero across many bins.
  3. Merger-aware tracking: when N lineages collapse to a single new minimum
     across scales, the survivor is the one with the earliest birth; the
     others die into it. This prevents real long-lived valleys from being
     fragmented into many short curves when the underlying valley is wide
     and noisy at small scales.
  4. Per-step jump tolerance: a lineage's nearest new minimum is accepted
     only if the bin-displacement is within `jump_const + jump_scale_step ×
     Δσ`. Rejects bogus cross-data jumps that would otherwise produce
     spurious long curves.
  5. Absolute-lifetime selector: a lineage is meaningful iff its lifetime
     exceeds `min_lifetime_frac × max_possible_lifetime`, where max_possible =
     log(σ_max / σ_min). Default min_lifetime_frac=0.5. Rationale: a true
     population-separator valley persists across roughly half the log-σ range
     or more (the bandwidth required to smooth out the valley between two
     well-separated modes is on the order of half the data range), while
     histogram-noise minima and intra-mode artifacts (including reflect-mode
     edge artifacts) die well before that. This replaces the paper's
     parametric ε-meaningful test with a single scale-invariant cutoff.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from collections import defaultdict


def _scale_space_curves(data, n_bins='fd', log_density=True,
                         max_scale=None, n_scale_steps=None,
                         jump_const=2.0, jump_scale_step=4.0):
    """Build the scale-space and track minima with mergers + tolerance.

    All minima at every scale are admitted as potential lineage seeds. The
    public function applies the lifetime-based meaningfulness test on the
    initial-scale (σ=σ_min) lineages.
    """
    data = np.asarray(data).flatten()
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return [], None, None, None, None, 0.0

    counts, edges = np.histogram(data, bins=n_bins)
    n_bins_actual = len(counts)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if max_scale is None:
        max_scale = float(n_bins_actual) / 2.0
    if n_scale_steps is None:
        n_scale_steps = max(50, n_bins_actual)
    sigmas = np.geomspace(0.5, max_scale, n_scale_steps)

    minima_per_scale = []
    for s in sigmas:
        L = gaussian_filter1d(counts.astype(float), sigma=s, mode='reflect')
        D = np.log(L + 1.0) if log_density else L
        idx, _ = find_peaks(-D, plateau_size=1)
        minima_per_scale.append(idx)

    curves = []
    active = []
    next_cid = 0
    for x in minima_per_scale[0]:
        cid = next_cid; next_cid += 1
        c = {'id': cid, 'born_at_sigma': float(sigmas[0]), 'born_at_x': int(x),
             'current_x': int(x), 'died_at_sigma': None, 'died_at_x': None,
             'x_trace': [int(x)], 'sigma_trace': [float(sigmas[0])],
             'merged_in_births': []}
        curves.append(c); active.append(c)

    for s_idx in range(1, n_scale_steps):
        sigma = float(sigmas[s_idx])
        prev_sigma = float(sigmas[s_idx - 1])
        new_xs = minima_per_scale[s_idx]
        tol = jump_const + jump_scale_step * (sigma - prev_sigma)

        if len(active) == 0:
            new_active = []
            for new_x in new_xs:
                cid = next_cid; next_cid += 1
                c = {'id': cid, 'born_at_sigma': sigma, 'born_at_x': int(new_x),
                     'current_x': int(new_x),
                     'died_at_sigma': None, 'died_at_x': None,
                     'x_trace': [int(new_x)], 'sigma_trace': [sigma],
                     'merged_in_births': []}
                curves.append(c); new_active.append(c)
            active = new_active
            continue
        if len(new_xs) == 0:
            for c in active:
                c['died_at_sigma'] = prev_sigma; c['died_at_x'] = c['current_x']
            active = []
            continue

        active_xs = np.array([c['current_x'] for c in active])
        dist = np.abs(active_xs[:, None] - new_xs[None, :])
        nearest_new_for_old = np.argmin(dist, axis=1)
        nearest_d = dist[np.arange(len(active)), nearest_new_for_old]

        groups = defaultdict(list)
        died_olds = []
        for i, j in enumerate(nearest_new_for_old):
            if nearest_d[i] <= tol:
                groups[int(j)].append(i)
            else:
                died_olds.append(i)

        new_active = []
        used_news = set()
        for j, old_idxs in groups.items():
            used_news.add(j)
            survivor_i = min(old_idxs, key=lambda i: active[i]['born_at_sigma'])
            survivor = active[survivor_i]
            survivor['current_x'] = int(new_xs[j])
            survivor['x_trace'].append(int(new_xs[j]))
            survivor['sigma_trace'].append(sigma)
            for i in old_idxs:
                if i == survivor_i: continue
                merged = active[i]
                merged['died_at_sigma'] = sigma
                merged['died_at_x'] = int(new_xs[j])
                survivor['merged_in_births'].append(merged['born_at_sigma'])
            new_active.append(survivor)
        for i in died_olds:
            c = active[i]
            c['died_at_sigma'] = prev_sigma; c['died_at_x'] = c['current_x']
        for j in range(len(new_xs)):
            if j in used_news: continue
            cid = next_cid; next_cid += 1
            c = {'id': cid, 'born_at_sigma': sigma, 'born_at_x': int(new_xs[j]),
                 'current_x': int(new_xs[j]),
                 'died_at_sigma': None, 'died_at_x': None,
                 'x_trace': [int(new_xs[j])], 'sigma_trace': [sigma],
                 'merged_in_births': []}
            curves.append(c); new_active.append(c)
        active = new_active

    for c in active:
        c['died_at_sigma'] = float(sigmas[-1]); c['died_at_x'] = c['current_x']

    initial = [c for c in curves if c['born_at_sigma'] == float(sigmas[0])]
    max_possible_lifetime = float(np.log(sigmas[-1]) - np.log(sigmas[0]))
    return initial, counts, centers, edges, sigmas, max_possible_lifetime


def scale_space_meaningful_valleys(
    data,
    n_bins='fd',
    log_density=True,
    min_lifetime_frac=0.5,
    return_diagnostics=False,
):
    """
    Find meaningful local minima ("valleys") of the smoothed log-histogram
    of `data`. Returns a sorted list of x-positions; empty if no meaningful
    valley.

    Parameters
    ----------
    data : array
        1D array of values (e.g. flattened log_umi.values).
    n_bins : int or str
        Bin count or histogram-bin-rule string. Default 'fd'
        (Freedman-Diaconis).
    log_density : bool
        If True, track minima in log-density. Default True.
    min_lifetime_frac : float
        Lifetime threshold for meaningfulness, as a fraction of the maximum
        possible lifetime `log(max_scale / min_scale)`. A curve is meaningful
        iff its lifetime exceeds this fraction. Default 0.5 — population-
        separator valleys persist across roughly half the log-σ range or more,
        while histogram noise and intra-mode artifacts die out well before that.
    return_diagnostics : bool
        If True, also return a dict with full intermediate state.

    Returns
    -------
    valleys_x : sorted list of float
        x-positions of meaningful valleys.
    (if return_diagnostics) diagnostics : dict
    """
    initial, counts, centers, edges, sigmas, max_possible = _scale_space_curves(
        data, n_bins=n_bins, log_density=log_density)
    if len(initial) == 0:
        return ([], {
            'n_initial_minima': 0, 'n_valid': 0, 'n_meaningful': 0,
            'lifetimes': np.array([]),
            'max_possible_lifetime': max_possible,
            'min_lifetime': min_lifetime_frac * max_possible,
            'counts': counts, 'centers': centers, 'edges': edges, 'sigmas': sigmas,
            'curves': [], 'meaningful_curves': [],
            'reason': 'no_initial_minima',
        }) if return_diagnostics else []

    # All initial-scale lineages are candidates. The lifetime test below is the
    # only meaningfulness criterion — it handles histogram noise, intra-mode
    # artifacts, and reflect-mode edge artifacts uniformly (all have short
    # lifetimes and fall below the floor).
    initial_valid = initial
    lifetimes_valid = np.array([np.log(c['died_at_sigma']) - np.log(c['born_at_sigma'])
                                 for c in initial_valid])
    min_lt = min_lifetime_frac * max_possible

    # Meaningfulness test: lifetime must exceed `min_lifetime_frac × max_possible`.
    # Rationale: a true population-separator valley persists across roughly half
    # of the log-σ range or more (the bandwidth required to smooth out the valley
    # between two well-separated modes is on the order of half the data range).
    # Histogram-noise minima and intra-mode artifacts die well before this scale.
    sorted_idx = np.argsort(lifetimes_valid)[::-1]
    sorted_lf = lifetimes_valid[sorted_idx]
    is_meaningful = sorted_lf >= min_lt
    n_meaningful = int(is_meaningful.sum())

    if n_meaningful == 0:
        return ([], {
            'n_initial_minima': len(initial), 'n_valid': len(initial_valid),
            'n_meaningful': 0,
            'lifetimes': lifetimes_valid, 'reason': 'no_strong_outlier',
            'max_possible_lifetime': max_possible, 'min_lifetime': min_lt,
            'counts': counts, 'centers': centers, 'edges': edges, 'sigmas': sigmas,
            'curves': initial_valid, 'meaningful_curves': [],
        }) if return_diagnostics else []

    meaningful_idx = sorted_idx[:n_meaningful]
    meaningful_curves = [initial_valid[i] for i in meaningful_idx]
    valleys = sorted(float(centers[c['born_at_x']]) for c in meaningful_curves)
    if return_diagnostics:
        return valleys, {
            'curves': initial_valid, 'meaningful_curves': meaningful_curves,
            'counts': counts, 'centers': centers, 'edges': edges, 'sigmas': sigmas,
            'lifetimes': lifetimes_valid, 'n_initial_minima': len(initial),
            'n_valid': len(initial_valid), 'n_meaningful': n_meaningful,
            'max_possible_lifetime': max_possible, 'min_lifetime': min_lt,
        }
    return valleys


def scale_space_threshold(data, default=None, **kwargs):
    """
    Return the leftmost meaningful valley of `data`, or `default` if none.
    Drop-in replacement for KDE-based threshold helpers.
    """
    valleys = scale_space_meaningful_valleys(data, **kwargs)
    return valleys[0] if valleys else default
