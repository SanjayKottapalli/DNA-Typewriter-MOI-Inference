import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import combinations, groupby
import seaborn as sns
from sklearn.neighbors import BallTree
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scale_space_threshold import scale_space_meaningful_valleys
import igraph as ig
import os
import contextlib
import traceback
import sys
import argparse
import timeit
from datetime import timedelta
from joblib import Parallel, delayed

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 500
plt.rcParams['svg.fonttype'] = 'none'

import random
import functools
print = functools.partial(print, flush=True)


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


# ---------------------------------------------------------------------------
# Reproducibility: a single seed governs all randomness in the pipeline.
# set_seeds() is called at module import and again at the start of main(),
# so reproducibility is robust to import timing and to joblib workers (each
# worker re-imports the module). The graph-coloring random search derives its
# per-batch RNG from this seed (see run_coloring / _coloring_batch).
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


def set_seeds(seed=RANDOM_SEED):
    """Seed every RNG the pipeline uses (numpy + Python random)."""
    np.random.seed(seed)
    random.seed(seed)


set_seeds()

sns.set_style("white")


WORD_DELIMITER = "|"

RPU_FLOOR = 1.0
RPU_CEILING = 1.5

EDIT_KEY = "GGAT"
EDIT_KEY_LEN = len(EDIT_KEY)


def read_csv(csv_path="", umi_cutoff=2):
    df = pd.read_csv(csv_path, header=0)
    df_filter = df[df['nUMI'] >= umi_cutoff]
    df_filter.fillna("", inplace=True)
    return df_filter


def parse_sites(row, site_cols):
    words = []
    for col in site_cols:
        val = str(row[col])
        if val == "" or val == "None" or val == "nan":
            break
        if not val.endswith(EDIT_KEY):
            break
        edit = val[:-EDIT_KEY_LEN]
        if len(edit) == 0:
            break
        words.append(edit)
    if len(words) == 0:
        return "ETY"
    return WORD_DELIMITER.join(words)


def is_word_prefix(short, long):
    short_words = short.split(WORD_DELIMITER)
    long_words = long.split(WORD_DELIMITER)
    if len(short_words) >= len(long_words):
        return False
    return long_words[:len(short_words)] == short_words


def shared_prefix_words(a, b):
    words_a = a.split(WORD_DELIMITER)
    words_b = b.split(WORD_DELIMITER)
    count = 0
    for wa, wb in zip(words_a, words_b):
        if wa == wb:
            count += 1
        else:
            break
    return count


def compute_kde_threshold(log_umi, bw=0.15, rel_height=0.25, log=True,
                           n_min_for_hist=100_000, min_lifetime_frac=0.5):
    """Find the binarization threshold between zero and signal modes.

    Runs the scale-space meaningful-valleys detector (Gilles & Heal 2014,
    adapted) on the FULL flattened distribution of `log_umi` (including the
    zero spike). Returns the LEFTMOST meaningful valley. In zero-inflated
    DTT data the leftmost valley is between the zero spike and the start of
    nonzero data, which acts as an "anything above zero counts as signal"
    threshold. If a real noise mode exists, it presents as a noise smear in
    the leftmost bins and the valley shifts accordingly.

    When no meaningful valley exists (unimodal data), returns 0.0.

    Parameters `bw`, `rel_height`, `log`, and `n_min_for_hist` are kept for
    API compatibility but are not used. `min_lifetime_frac` controls the
    meaningfulness floor.

    Returns
    -------
    threshold : float
    x_grid : np.ndarray
        Bin centers of the histogram, for plot compatibility.
    density : np.ndarray
        Lightly-smoothed log-density at x_grid.
    """
    data = np.asarray(log_umi.values).flatten() if hasattr(log_umi, "values") \
           else np.asarray(log_umi).flatten()
    data = data[np.isfinite(data)]
    n = len(data)
    if n < 2:
        return 0.0, np.array([0.0, 1.0]), np.array([0.0, 0.0])

    # Data-size-adaptive fixed bin count. Real DTT data is heavily
    # zero-inflated (often >99% zeros after row normalization), so we cannot
    # rely on Freedman-Diaconis (IQR collapses to 0). The scale-space
    # smoothing ladder handles bandwidth adaptation internally.
    n_bins_use = int(min(200, max(10, np.sqrt(n))))
    valleys, diag = scale_space_meaningful_valleys(
        data, n_bins=n_bins_use, log_density=log,
        min_lifetime_frac=min_lifetime_frac,
        return_diagnostics=True,
    )
    threshold = float(valleys[0]) if valleys else 0.0

    counts = diag.get('counts')
    centers = diag.get('centers')
    if counts is None or centers is None or len(counts) == 0:
        try:
            counts, edges = np.histogram(data, bins='fd')
        except Exception:
            counts, edges = np.histogram(data, bins=50)
        centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = len(counts)
    sigma_plot = max(1.0, n_bins / 75.0)
    smoothed = gaussian_filter1d(counts.astype(float), sigma=sigma_plot, mode='reflect')
    density = np.log(smoothed + 1.0) if log else smoothed
    return threshold, centers, density



def build_pattern_trie(patterns):
    patterns = sorted(patterns, key=lambda p: len(p.split(WORD_DELIMITER)))
    parent = {}
    children = {p: [] for p in patterns}
    for i, p in enumerate(patterns):
        best_parent = None
        best_parent_len = 0
        for j in range(i - 1, -1, -1):
            candidate = patterns[j]
            candidate_len = len(candidate.split(WORD_DELIMITER))
            if candidate_len < len(p.split(WORD_DELIMITER)) and is_word_prefix(candidate, p):
                if best_parent is None or candidate_len > best_parent_len:
                    best_parent = candidate
                    best_parent_len = candidate_len
        parent[p] = best_parent
        if best_parent is not None:
            children[best_parent].append(p)
    roots = [p for p in patterns if parent[p] is None]
    is_leaf = {p: len(children[p]) == 0 for p in patterns}
    levels = []
    current_level = roots
    while current_level:
        levels.append(current_level)
        next_level = []
        for p in current_level:
            next_level.extend(children[p])
        current_level = next_level
    return children, roots, is_leaf, levels


def redistribute_shadow_umis(nUMI, log_umi, threshold):
    patterns = [p for p in nUMI.columns if p != "ETY"]
    if len(patterns) <= 1:
        return nUMI
    children, roots, is_leaf, levels = build_pattern_trie(patterns)
    original_log_umi = log_umi.copy()
    n_redistributed = 0
    for level in levels:
        for node in level:
            if is_leaf[node]:
                continue
            node_children = children[node]
            if len(node_children) == 0:
                continue
            child_above = original_log_umi[node_children].values > threshold
            any_child_real = child_above.any(axis=1)
            child_raw = nUMI[node_children].values.copy()
            child_raw_masked = child_raw * child_above
            child_sums = child_raw_masked.sum(axis=1)
            safe_sums = np.where(child_sums > 0, child_sums, 1.0)
            proportions = child_raw_masked / safe_sums[:, np.newaxis]
            mask = any_child_real & (child_sums > 0)
            parent_umis = nUMI[node].values
            redistribution = parent_umis[:, np.newaxis] * proportions
            redistribution[~mask] = 0
            nUMI[node_children] += redistribution
            nUMI.loc[mask, node] = 0
            n_redistributed += mask.sum()
    print(f"  Redistributed shadow UMIs in {n_redistributed} cell-pattern entries")
    return nUMI


def _aggregate_cell_pattern(df):
    """Collapse to one row per (Cell, Pattern), summing nUMI.

    parse_sites is many-to-one (many raw site-tuples map to the same parsed
    pattern), so the input routinely has multiple rows per (Cell, Pattern).
    Aggregating here ensures the noise-redistribution proportions are computed
    from the correct per-(cell,pattern) total UMI rather than an arbitrary
    single row's value. Non-aggregated columns (TargetBC, etc.) take the first
    value, which is safe because this function is always called on a single
    TapeBC's slice.
    """
    if len(df) == 0:
        return df
    agg = {c: 'first' for c in df.columns if c not in ('Cell', 'Pattern', 'nUMI')}
    agg['nUMI'] = 'sum'
    out = df.groupby(['Cell', 'Pattern'], as_index=False).agg(agg)
    return out[df.columns]


def redistribute_noise_to_real(test_real, test_noise):
    if len(test_noise) == 0 or len(test_real) == 0:
        print("  No noise redistribution needed")
        return test_real
    # Collapse duplicate (Cell, Pattern) rows so UMI sums and proportional
    # redistribution are computed correctly (see _aggregate_cell_pattern).
    test_real = _aggregate_cell_pattern(test_real)
    test_noise = _aggregate_cell_pattern(test_noise)
    real_patterns = [p for p in test_real['Pattern'].unique() if p != "ETY"]
    noise_patterns = [p for p in test_noise['Pattern'].unique() if p != "ETY"]
    if len(real_patterns) == 0 or len(noise_patterns) == 0:
        print("  No noise redistribution needed")
        return test_real
    all_pats = list(real_patterns) + list(noise_patterns)
    max_len = max(len(p) for p in all_pats)
    sentinel = "~"
    real_padded = [p.ljust(max_len, sentinel) for p in real_patterns]
    noise_padded = [p.ljust(max_len, sentinel) for p in noise_patterns]
    real_encoded = np.array([[ord(c) for c in p] for p in real_padded], dtype=np.float64)
    noise_encoded = np.array([[ord(c) for c in p] for p in noise_padded], dtype=np.float64)
    tree = BallTree(real_encoded, metric='hamming')
    hamming_threshold = 1.0 / max_len + 1e-9
    neighbors = tree.query_radius(noise_encoded, r=hamming_threshold)
    noise_to_real = {}
    for noise_idx, real_indices in enumerate(neighbors):
        if len(real_indices) > 0:
            noise_to_real[noise_patterns[noise_idx]] = [
                real_patterns[ri] for ri in real_indices
            ]
    n_absorbed = 0
    if len(noise_to_real) == 0:
        print("  No noise patterns within 1bp of real patterns")
        return test_real
    test_real = test_real.copy()
    real_lookup = {}
    for idx, row in test_real.iterrows():
        key = (row['Cell'], row['Pattern'])
        if key not in real_lookup:
            real_lookup[key] = idx
    for idx, noise_row in test_noise.iterrows():
        noise_pat = noise_row['Pattern']
        cell = noise_row['Cell']
        noise_umis = noise_row['nUMI']
        if noise_pat not in noise_to_real:
            continue
        candidate_reals = noise_to_real[noise_pat]
        existing_reals = []
        existing_umis = []
        for rp in candidate_reals:
            key = (cell, rp)
            if key in real_lookup:
                existing_reals.append(rp)
                existing_umis.append(test_real.loc[real_lookup[key], 'nUMI'])
        if len(existing_reals) == 0:
            continue
        existing_umis = np.array(existing_umis, dtype=float)
        total = existing_umis.sum()
        if total == 0:
            continue
        proportions = existing_umis / total
        for rp, prop in zip(existing_reals, proportions):
            key = (cell, rp)
            test_real.loc[real_lookup[key], 'nUMI'] += noise_umis * prop
        n_absorbed += 1
    print(f"  Absorbed {n_absorbed} noise entries into real patterns")
    return test_real


# =========================================================================
# Precompute pairwise similarity weights (sparse, grouped by first word)
# =========================================================================

def precompute_pair_weights(patterns):
    pattern_to_idx = {p: i for i, p in enumerate(patterns)}
    n = len(patterns)
    first_word_groups = defaultdict(list)
    for p in patterns:
        first_word_groups[p.split(WORD_DELIMITER)[0]].append(p)
    p1_list = []
    p2_list = []
    w_list = []
    for group in first_word_groups.values():
        if len(group) < 2:
            continue
        for a, b in combinations(group, 2):
            w = shared_prefix_words(a, b)
            key = (min(a, b), max(a, b))
            p1_list.append(pattern_to_idx[key[0]])
            p2_list.append(pattern_to_idx[key[1]])
            w_list.append(w)
    p1_idx = np.array(p1_list, dtype=np.int32)
    p2_idx = np.array(p2_list, dtype=np.int32)
    w_arr = np.array(w_list, dtype=np.float64)
    return pattern_to_idx, p1_idx, p2_idx, w_arr


def evaluate_coloring_weight(coloring_dict, pattern_to_idx, p1_idx, p2_idx, w_arr, n_patterns):
    color_arr = np.full(n_patterns, -1, dtype=np.int32)
    for p, c in coloring_dict.items():
        if p in pattern_to_idx:
            color_arr[pattern_to_idx[p]] = c
    c1 = color_arr[p1_idx]
    c2 = color_arr[p2_idx]
    mask = (c1 == c2) & (c1 >= 0)
    return w_arr[mask].sum()


# =========================================================================
# Co-occurrence graph construction
# =========================================================================

def build_cooccurrence_graph_from_clusters(patterns, cluster_sig, binary_columns):
    G = ig.Graph(directed=False)
    G.add_vertices(patterns)
    name_set = set(patterns)
    edges = set()
    for i in cluster_sig:
        present = [col for col, val in zip(binary_columns, cluster_sig[i]) if val > 0.5]
        present_in_graph = [p for p in present if p in name_set]
        for a, b in combinations(present_in_graph, 2):
            edges.add((a, b))
    if edges:
        G.add_edges(list(edges))
    return G


# =========================================================================
# Graph coloring
# =========================================================================

def greedy_color_random(adj, n, rng, max_colors=None):
    """Greedy graph coloring with random vertex ordering.

    Equivalent to nx.greedy_color(G, strategy='random_sequential') but operates
    on a pre-extracted adjacency list (list of lists of neighbor indices) for
    maximum performance in tight loops.

    Args:
        adj: list of lists — adj[v] = list of neighbor vertex indices
        n: number of vertices
        rng: random.Random instance for shuffling
        max_colors: if provided, abort and return None as soon as this many
            colors are needed (i.e., when we would assign color index
            >= max_colors). This is the early-abort optimization: most random
            orderings on large graphs cannot beat the current best K, so
            we skip them after a few vertices instead of running to completion.

    Returns a list of color assignments indexed by vertex ID, or None if
    aborted because the ordering needed >= max_colors colors.
    """
    order = list(range(n))
    rng.shuffle(order)
    colors = [-1] * n
    used = [False] * n  # boolean array; n is upper bound on colors
    for v in order:
        nbs = adj[v]
        for nb in nbs:
            c = colors[nb]
            if c >= 0:
                used[c] = True
        c = 0
        while used[c]:
            c += 1
        # Early abort: assigning color index c means using (c+1) distinct
        # colors. If c >= max_colors, this ordering cannot match or improve
        # the current best, so bail out. (used is local; no cleanup needed.)
        if max_colors is not None and c >= max_colors:
            return None
        colors[v] = c
        for nb in nbs:
            c = colors[nb]
            if c >= 0:
                used[c] = False
    return colors


def _coloring_batch(adj, n_vertices, names, pattern_to_idx, p1_idx, p2_idx, w_arr,
                    n_patterns, batch_size, current_best_clique, current_best_sum,
                    rng_seed=None):
    """Run a batch of random-ordering greedy colorings and keep the best.

    Uses early-abort via `max_colors=current_best_clique`: any ordering that
    would require >= current_best_clique colors is dropped after the first
    vertex that needs that color, since it cannot improve on the best so far.

    `rng_seed` makes the random vertex orderings reproducible: with a fixed
    seed the same input graph produces the same coloring search every run.
    When None, a system-entropy-seeded RNG is used (non-reproducible).
    """
    import random as _rng_mod
    rng = _rng_mod.Random(rng_seed)
    batch_best_clique = current_best_clique
    batch_best_sum = current_best_sum
    batch_best_coloring = None
    for _ in range(batch_size):
        colors = greedy_color_random(adj, n_vertices, rng,
                                     max_colors=batch_best_clique)
        if colors is None:
            continue  # aborted early; this ordering cannot improve K
        n_colors = max(colors) + 1 if colors else 0
        # n_colors <= batch_best_clique is guaranteed by max_colors gating,
        # but keep the check for safety against off-by-one.
        if n_colors <= batch_best_clique:
            color_arr = np.full(n_patterns, -1, dtype=np.int32)
            for i, name in enumerate(names):
                if name in pattern_to_idx:
                    color_arr[pattern_to_idx[name]] = colors[i]
            c1 = color_arr[p1_idx]
            c2 = color_arr[p2_idx]
            mask = (c1 == c2) & (c1 >= 0)
            weight = float(w_arr[mask].sum())
            if n_colors < batch_best_clique or weight > batch_best_sum:
                batch_best_clique = n_colors
                batch_best_sum = weight
                batch_best_coloring = {name: colors[i] for i, name in enumerate(names)}
    return batch_best_clique, batch_best_sum, batch_best_coloring



def run_coloring(G_cooccur, pattern_to_idx, p1_idx, p2_idx, w_arr, n_patterns, label=""):
    print(f"Finding best graph partition ({label})...")
    names = G_cooccur.vs['name']
    n = len(names)
    if n == 0:
        return None, 0, 0
    if n == 1:
        return {names[0]: 0}, 1, 0

    # --- DSATUR seed: deterministic single-pass coloring that establishes
    # a strong upper bound on K. The subsequent random search uses this as
    # the early-abort threshold, so most random orderings die after a few
    # vertices on large graphs instead of running to completion. ---
    dsatur_colors_ig = G_cooccur.vertex_coloring_greedy(method='dsatur')
    dsatur_colors = list(dsatur_colors_ig)
    best_clique = max(dsatur_colors) + 1
    color_arr = np.full(n_patterns, -1, dtype=np.int32)
    for i, name in enumerate(names):
        if name in pattern_to_idx:
            color_arr[pattern_to_idx[name]] = dsatur_colors[i]
    c1 = color_arr[p1_idx]
    c2 = color_arr[p2_idx]
    mask = (c1 == c2) & (c1 >= 0)
    best_sum = float(w_arr[mask].sum())
    best_coloring = {name: dsatur_colors[i] for i, name in enumerate(names)}
    print(f"  DSATUR seed: colors={best_clique}, weight={best_sum}")

    total_iterations = n * 2500
    batch_size = 1000
    patience = max(50000, n * 50)
    iters_since_improvement = 0
    total_processed = 0
    batch_idx = 0
    # Pre-extract adjacency list for fast coloring in the hot loop
    adj = [G_cooccur.neighbors(v) for v in range(n)]
    while total_processed < total_iterations:
        iters_this_round = min(batch_size, total_iterations - total_processed)
        # Deterministic per-batch seed derived from the global RANDOM_SEED, the
        # number of vertices, and the batch index, so the random-ordering search
        # is fully reproducible across runs (fixes the prior random.Random()
        # that pulled from system entropy).
        batch_seed = (RANDOM_SEED * 1_000_003 + n * 9973 + batch_idx) & 0x7FFFFFFF
        batch_clique, batch_sum, batch_coloring = _coloring_batch(
            adj, n, names, pattern_to_idx, p1_idx, p2_idx, w_arr,
            n_patterns, iters_this_round, best_clique, best_sum,
            rng_seed=batch_seed
        )
        batch_idx += 1
        total_processed += iters_this_round
        improved = False
        if batch_coloring is not None:
            if (batch_clique < best_clique or
                    (batch_clique == best_clique and batch_sum > best_sum)):
                best_clique = batch_clique
                best_sum = batch_sum
                best_coloring = batch_coloring.copy()
                improved = True
        if improved:
            iters_since_improvement = 0
            print(f"  Improved: colors={best_clique}, weight={best_sum} "
                  f"(at iter {total_processed})")
        else:
            iters_since_improvement += iters_this_round
        if iters_since_improvement >= patience:
            print(f"  Early termination: no improvement in {patience} iterations")
            break
    print(f"  Final: colors={best_clique}, weight={best_sum}, "
          f"total iterations={total_processed}")
    return best_coloring, best_clique, best_sum


# =========================================================================
# EXPRESSION OUTPUT HELPER: build locus + UMI matrices from a coloring
# =========================================================================


def _build_locus_and_umi(cell_index, binary_arr, all_patterns_arr,
                         coloring, n_colors, nUMI_adjusted):
    """Build locus assignment DataFrame and parallel UMI DataFrame."""
    n_cells = len(cell_index)
    col_names = list(range(n_colors))
    color_lookup = np.full(len(all_patterns_arr), -1, dtype=np.int32)
    for i, p in enumerate(all_patterns_arr):
        if p in coloring:
            color_lookup[i] = coloring[p]
    locus_arr = np.full((n_cells, n_colors), "", dtype=object)
    umi_arr = np.zeros((n_cells, n_colors), dtype=np.float64)
    cell_indices, pattern_indices = np.where(binary_arr)
    colors = color_lookup[pattern_indices]
    valid = colors >= 0
    cell_indices = cell_indices[valid]
    pattern_indices = pattern_indices[valid]
    colors = colors[valid]

    # Collision check: a (cell, color) slot written by more than one pattern
    # means two patterns at the SAME inferred locus in the SAME cell — a
    # biological contradiction (a locus carries one edit state per cell).
    # numpy fancy-index assignment keeps only the last write, silently dropping
    # the others. This can happen when a pattern pair co-occurs in exactly one
    # cell (the cell-level augmentation requires >=2 cells to add the edge that
    # would force different colors). Warn so the user knows it occurred.
    flat_slots = cell_indices.astype(np.int64) * n_colors + colors.astype(np.int64)
    n_collisions = len(flat_slots) - len(np.unique(flat_slots))
    if n_collisions > 0:
        n_cells_affected = len(np.unique(
            flat_slots[pd.Series(flat_slots).duplicated(keep=False).values]
            // n_colors
        )) if len(flat_slots) else 0
        print(f"  Warning: {n_collisions} pattern-collision overwrite(s) in "
              f"locus assignment across {n_cells_affected} cell(s) — two patterns "
              f"assigned the same locus in the same cell (last-written kept). "
              f"Possible doublet, residual noise, or a once-co-occurring pair.")

    locus_arr[cell_indices, colors] = all_patterns_arr[pattern_indices]
    if nUMI_adjusted is not None:
        nUMI_vals = nUMI_adjusted.values
        nUMI_rows = {c: i for i, c in enumerate(nUMI_adjusted.index)}
        nUMI_cols = {p: j for j, p in enumerate(nUMI_adjusted.columns)}
        for ci, pi, color in zip(cell_indices, pattern_indices, colors):
            cell = cell_index[ci]
            pat = all_patterns_arr[pi]
            if cell in nUMI_rows and pat in nUMI_cols:
                umi_arr[ci, color] = nUMI_vals[nUMI_rows[cell], nUMI_cols[pat]]
    locus_df = pd.DataFrame(locus_arr, index=cell_index, columns=col_names)
    umi_df = pd.DataFrame(umi_arr, index=cell_index, columns=col_names)
    return locus_df, umi_df


# =========================================================================
# Main pipeline functions
# =========================================================================

def gen_patterns(df_filter, tapebc="", min_umi=2, plot=False, outdir="", name=""):
    """
    Returns either:
      - ("early", locus_df, umi_df)         for edge cases
      - ("normal", binary, log_umi, has_ety, nUMI_adjusted)  for normal flow
    """
    print("Tape BC: {}".format(tapebc))
    test = df_filter[df_filter['TargetBC'] == tapebc].copy()
    cells = list(test['Cell'].values)

    print(test.shape)
    print(test)

    test_real = test[test['is_real']].copy()
    test_noise = test[~test['is_real']].copy()
    print(f"  Real rows: {len(test_real)}, Noise rows: {len(test_noise)}")

    if len(test_real) == 0:
        print("  No real sequences found")
        locus_df = pd.DataFrame(
            "", index=np.unique(cells), columns=[tapebc + "-0"]
        )
        umi_df = pd.DataFrame(
            0.0, index=np.unique(cells), columns=[tapebc + "-0"]
        )
        return ("early", locus_df, umi_df)

    print("Redistributing noise UMIs to real patterns...")
    test_real = redistribute_noise_to_real(test_real, test_noise)
    test_real = test_real[test_real['nUMI'] >= min_umi].copy()

    if len(test_real) == 0:
        print("  No sequences above UMI cutoff after redistribution")
        locus_df = pd.DataFrame(
            "", index=np.unique(cells), columns=[tapebc + "-0"]
        )
        umi_df = pd.DataFrame(
            0.0, index=np.unique(cells), columns=[tapebc + "-0"]
        )
        return ("early", locus_df, umi_df)

    print("Computing consensus patterns...")
    pattern_counts = test_real.groupby('Pattern')['nUMI'].sum().sort_values(ascending=False)
    unique_patterns = list(pattern_counts.index)

    if len(unique_patterns) == 0:
        locus_df = pd.DataFrame(
            "", index=np.unique(cells), columns=[tapebc + "-0"]
        )
        umi_df = pd.DataFrame(
            0.0, index=np.unique(cells), columns=[tapebc + "-0"]
        )
        return ("early", locus_df, umi_df)

    print(f"  Unique patterns (including ETY): {len(unique_patterns)}")
    # for p in unique_patterns:
    #     print(f"    {p}: {pattern_counts[p]} UMIs")

    print("Counting nUMI per cell per pattern...")
    entries = test_real[test_real['Pattern'].isin(unique_patterns)]
    nUMI = entries.groupby(['Cell', 'Pattern'])['nUMI'].sum().unstack(fill_value=0)
    for p in unique_patterns:
        if p not in nUMI.columns:
            nUMI[p] = 0

    non_ety_patterns = [p for p in unique_patterns if p != "ETY"]
    if len(non_ety_patterns) == 0:
        all_cells = np.unique(cells)
        col_name = tapebc + "-0"
        locus_df = pd.DataFrame("ETY", index=all_cells, columns=[col_name])
        umi_df = pd.DataFrame(0.0, index=all_cells, columns=[col_name])
        if "ETY" in nUMI.columns:
            for cell in all_cells:
                if cell in nUMI.index:
                    umi_df.loc[cell, col_name] = nUMI.loc[cell, "ETY"]
        return ("early", locus_df, umi_df)

    if len(non_ety_patterns) == 1:
        all_cells = np.unique(cells)
        col_name = tapebc + "-0"
        locus_df = pd.DataFrame("", index=all_cells, columns=[col_name])
        umi_df = pd.DataFrame(0.0, index=all_cells, columns=[col_name])
        p = non_ety_patterns[0]
        for cell in locus_df.index:
            if cell in nUMI.index:
                if nUMI.loc[cell, p] > 0:
                    locus_df.loc[cell, col_name] = p
                    umi_df.loc[cell, col_name] = nUMI.loc[cell, p]
                elif "ETY" in nUMI.columns and nUMI.loc[cell, "ETY"] > 0:
                    locus_df.loc[cell, col_name] = "ETY"
                    umi_df.loc[cell, col_name] = nUMI.loc[cell, "ETY"]
        return ("early", locus_df, umi_df)

    nUMI_for_binary = nUMI[non_ety_patterns].copy()
    row_sums = nUMI_for_binary.sum(axis=1)
    ety_only_cells = row_sums[row_sums == 0].index
    if len(ety_only_cells) > 0:
        print(f"  Removing {len(ety_only_cells)} ETY-only cells from binarization")
        nUMI_for_binary = nUMI_for_binary.loc[row_sums > 0].copy()

    print("Pass 1: Computing preliminary binarization threshold...")
    if nUMI_for_binary.shape[1] > 1:
        log_umi_pass1 = nUMI_for_binary.div(nUMI_for_binary.sum(axis=1), axis=0)
    else:
        log_umi_pass1 = nUMI_for_binary.copy()
    log_umi_pass1 = np.log(log_umi_pass1 + 0.1) - np.log(0.1)

    threshold_pass1, kde_x_1, kde_smooth_1 = compute_kde_threshold(log_umi_pass1)
    print(f"  Pass 1 threshold: {threshold_pass1}")

    print("Redistributing shadow transcript UMIs...")
    nUMI_for_binary = redistribute_shadow_umis(nUMI_for_binary, log_umi_pass1, threshold_pass1)

    print("Pass 2: Recomputing binarization threshold after redistribution...")
    if nUMI_for_binary.shape[1] > 1:
        log_umi = nUMI_for_binary.div(nUMI_for_binary.sum(axis=1), axis=0)
    else:
        log_umi = nUMI_for_binary.copy()
    log_umi = np.log(log_umi + 0.1) - np.log(0.1)

    threshold_pass2, kde_x_2, kde_smooth_2 = compute_kde_threshold(log_umi)
    print(f"  Pass 2 threshold: {threshold_pass2}")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        data1 = np.ndarray.flatten(log_umi_pass1.values)
        axes[0].hist(data1, density=True, bins=10, alpha=0.7)
        axes[0].plot(kde_x_1, kde_smooth_1, color='red')
        axes[0].axvline(threshold_pass1, color='blue', linestyle='--',
                        label=f'threshold={threshold_pass1:.2f}')
        axes[0].set_title('Pass 1: Before redistribution')
        axes[0].set_ylabel("Density")
        axes[0].set_xlabel("Log-counts")
        axes[0].legend()
        data2 = np.ndarray.flatten(log_umi.values)
        axes[1].hist(data2, density=True, bins=10, alpha=0.7)
        axes[1].plot(kde_x_2, kde_smooth_2, color='red')
        axes[1].axvline(threshold_pass2, color='blue', linestyle='--',
                        label=f'threshold={threshold_pass2:.2f}')
        axes[1].set_title('Pass 2: After redistribution')
        axes[1].set_ylabel("Density")
        axes[1].set_xlabel("Log-counts")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, name, "stdout", "tape_expression.svg"), dpi=300)
        plt.show()
        plt.clf()

    binary = np.where(log_umi > threshold_pass2, 1.0, 0.0)

    # Drop pattern columns that are above threshold in zero cells. These
    # patterns can never be assigned to a locus (they're never 'present' in
    # any cell), so they'd only show up as isolated vertices in the full
    # graph and as wasted pairwise-weight computations. Removing them keeps
    # nUMI_adjusted intact (used for ETY filling) but cleans up the matrices
    # passed to find_graph_and_partition.
    col_sums_binary = binary.sum(axis=0)
    keep_cols = col_sums_binary > 0
    n_dropped = int((~keep_cols).sum())
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} pattern(s) with no cells above threshold "
              f"({int(keep_cols.sum())} kept)")
        binary = binary[:, keep_cols]
        log_umi = log_umi.loc[:, keep_cols]

    # has_ety covers ALL cells in the original nUMI (including ETY-only ones
    # that were dropped from nUMI_for_binary), so we can re-insert them later
    has_ety = None
    if "ETY" in nUMI.columns:
        has_ety = nUMI["ETY"] > 0

    # EXPRESSION OUTPUT: build adjusted UMI matrix
    # nUMI_for_binary now has shadow-redistributed values for non-ETY patterns
    # Combine with ETY from original nUMI. Index over ALL cells (not just
    # nUMI_for_binary.index) so ETY-only cells dropped from binarization are
    # still represented in nUMI_adjusted for downstream ETY filling.
    nUMI_adjusted = nUMI_for_binary.reindex(nUMI.index, fill_value=0)
    if "ETY" in nUMI.columns:
        nUMI_adjusted["ETY"] = nUMI["ETY"].values

    return ("normal", binary, log_umi, has_ety, nUMI_adjusted)


def find_graph_and_partition(binary, log_umi, nUMI_adjusted=None,
                             min_cells=5, plot=False,
                             outdir="", name=""):
    binary = pd.DataFrame(index=log_umi.index, columns=log_umi.columns, data=binary)
    all_patterns = list(binary.columns)
    all_patterns_arr = np.array(all_patterns)
    binary_arr = binary.values > 0.5
    n_patterns = len(all_patterns)
    print(f"  Total patterns: {n_patterns}")

    print("Precomputing pairwise similarity weights...")
    pattern_to_idx, p1_idx, p2_idx, w_arr = precompute_pair_weights(all_patterns)
    print(f"  Nonzero weight pairs: {len(w_arr)}")

    pattern_cell_counts = binary.sum(axis=0).astype(int)
    core_patterns = [p for p in all_patterns if pattern_cell_counts[p] >= min_cells]
    peripheral_patterns = [p for p in all_patterns if pattern_cell_counts[p] < min_cells]
    print(f"  Core patterns (>= {min_cells} cells): {len(core_patterns)}")
    print(f"  Peripheral patterns (< {min_cells} cells): {len(peripheral_patterns)}")

    has_core = len(core_patterns) > 0

    if has_core:
        # ---- Pass 1: Core coloring to find K ----
        print("--- Pass 1: Core graph coloring ---")
        binary_core = binary[core_patterns].copy()

        row_sums = binary_core.sum(axis=1)
        nonzero_rows = row_sums[row_sums > 0].index
        zero_rows = row_sums[row_sums == 0].index
        print(f"  Cells with core signal: {len(nonzero_rows)}, "
              f"All-zero cells: {len(zero_rows)}")

        binary_for_clustering = binary_core.loc[nonzero_rows].copy()

        # Group identical binary vectors. With the original DBSCAN settings
        # (eps=0.5, integer Hamming distance, min_samples=min_cells), the only
        # cells that ever clustered together were ones with exactly identical
        # binary vectors, and groups of size < min_cells were labeled noise.
        # This groupby reproduces those semantics exactly in O(N*D) time and
        # O(N*D) memory, vs DBSCAN's O(N^2) on both axes.
        binary_arr_clust = binary_for_clustering.values.astype(np.uint8)
        cell_index_clust = binary_for_clustering.index
        group_map = defaultdict(list)
        for cell_pos, row in enumerate(binary_arr_clust):
            group_map[row.tobytes()].append(cell_pos)

        labels = np.full(len(binary_arr_clust), -1, dtype=np.int64)
        next_label = 0
        for indices in group_map.values():
            if len(indices) >= min_cells:
                for i in indices:
                    labels[i] = next_label
                next_label += 1

        assign = pd.Series(labels, index=cell_index_clust).astype(str)
        cluster_unique = [c for c in np.unique(assign.values) if c != '-1']
        n_noise = int((assign == '-1').sum())
        print(f"  Number of clusters (identical-vector groups): {len(cluster_unique)}")
        print(f"  Noise cells (vector seen in < min_cells cells): {n_noise}")

        cluster_sig = {}
        for i in cluster_unique:
            index = assign[assign == i].index
            a = np.mean(binary_for_clustering.loc[index, :], axis=0)
            cluster_sig[i] = a.values

        print("Constructing core co-occurrence graph...")
        G_cooccur_core = build_cooccurrence_graph_from_clusters(
            core_patterns, cluster_sig, binary_for_clustering.columns
        )
        n_cluster_edges = G_cooccur_core.ecount()
        print(f"  Core co-occurrence graph (from clusters): "
              f"{G_cooccur_core.vcount()} nodes, {n_cluster_edges} edges")
        cell_edge_counts = Counter()
        for i in range(len(binary_for_clustering.index)):
            present = [p for p, val in zip(core_patterns, binary_for_clustering.values[i]) if val > 0.5]
            if len(present) >= 2:
                for a, b in combinations(present, 2):
                    cell_edge_counts[(min(a, b), max(a, b))] += 1
        n_augmented = 0
        for (a, b), count in cell_edge_counts.items():
            if count >= 2 and G_cooccur_core.get_eid(a, b, error=False) == -1:
                G_cooccur_core.add_edge(a, b)
                n_augmented += 1
        if n_augmented > 0:
            print(f"  Augmented core graph with {n_augmented} cell-level co-occurrence edges (>=2 cells)")
        print(f"  Final core co-occurrence graph: {G_cooccur_core.vcount()} nodes, {G_cooccur_core.ecount()} edges")

        core_pattern_to_idx, core_p1, core_p2, core_w = precompute_pair_weights(core_patterns)
        n_core = len(core_patterns)

        core_coloring, K, core_weight = run_coloring(
            G_cooccur_core, core_pattern_to_idx, core_p1, core_p2, core_w,
            n_core, label="core"
        )

        if core_coloring is None:
            print("  Core coloring failed, returning single locus")
            locus_df = pd.DataFrame("", index=binary.index, columns=["0"])
            umi_df = pd.DataFrame(0.0, index=binary.index, columns=["0"])
            return locus_df, None, umi_df

        print(f"  Core loci (K): {K}")

        if len(peripheral_patterns) == 0:
            print("  No peripheral patterns, using core coloring directly")
            locus_df, umi_df = _build_locus_and_umi(
                binary.index, binary_arr, all_patterns_arr,
                core_coloring, K, nUMI_adjusted
            )
            cliques = [set() for _ in range(K)]
            for node, color in core_coloring.items():
                cliques[color].add(node)
            return locus_df, cliques, umi_df

        # ---- Pass 2: Full graph coloring ----
        print("--- Pass 2: Full graph coloring ---")
        print("Building full co-occurrence graph...")

        G_cooccur_full = G_cooccur_core.copy()
        G_cooccur_full.add_vertices(peripheral_patterns)

        # Collect peripheral-incident co-occurrence edges in a set first so we
        # don't add duplicate edges to the graph (which would inflate ecount()
        # and the logged edge count). Edges already in the core graph are
        # skipped via get_eid.
        peripheral_set = set(peripheral_patterns)
        new_edges = set()
        for i in range(len(binary.index)):
            present = all_patterns_arr[binary_arr[i]]
            if len(present) < 2:
                continue
            has_peripheral = any(p in peripheral_set for p in present)
            if not has_peripheral:
                continue
            for a, b in combinations(present, 2):
                if a in peripheral_set or b in peripheral_set:
                    new_edges.add((a, b) if a < b else (b, a))
        edges_to_add = [
            (a, b) for (a, b) in new_edges
            if G_cooccur_full.get_eid(a, b, error=False) == -1
        ]
        if edges_to_add:
            G_cooccur_full.add_edges(edges_to_add)

        print(f"  Full co-occurrence graph: {G_cooccur_full.vcount()} nodes, "
              f"{G_cooccur_full.ecount()} edges")

        full_coloring, full_n_colors, full_weight = run_coloring(
            G_cooccur_full, pattern_to_idx, p1_idx, p2_idx, w_arr,
            n_patterns, label="full"
        )

        if full_coloring is None:
            print("  Full coloring failed, falling back to core coloring")
            locus_df, umi_df = _build_locus_and_umi(
                binary.index, binary_arr, all_patterns_arr,
                core_coloring, K, nUMI_adjusted
            )
            cliques = [set() for _ in range(K)]
            for node, color in core_coloring.items():
                cliques[color].add(node)
            return locus_df, cliques, umi_df

        print(f"  Full coloring uses {full_n_colors} colors, core K={K}")

        if full_n_colors <= K:
            print(f"  Full coloring fits within K={K}, using directly")
            final_coloring = full_coloring
            final_n_colors = full_n_colors
        else:
            print(f"  Full coloring needs {full_n_colors} > K={K}, "
                  f"keeping {K} largest color groups")
            color_groups = defaultdict(set)
            for node, color in full_coloring.items():
                color_groups[color].add(node)
            sorted_groups = sorted(color_groups.items(), key=lambda x: len(x[1]), reverse=True)
            kept_colors = sorted_groups[:K]
            discarded_groups = sorted_groups[K:]
            final_coloring = {}
            for new_color, (old_color, nodes) in enumerate(kept_colors):
                for node in nodes:
                    final_coloring[node] = new_color
            final_n_colors = K
            discarded_patterns = []
            for old_color, nodes in discarded_groups:
                discarded_patterns.extend(nodes)
            print(f"  Discarded {len(discarded_patterns)} patterns from "
                  f"{len(discarded_groups)} extra color groups")
            if len(discarded_patterns) <= 20:
                print(f"  Discarded: {discarded_patterns}")

    else:
        print("No core patterns found. Building graph from all cells...")
        edge_counts = Counter()
        for i in range(len(binary.index)):
            present = all_patterns_arr[binary_arr[i]]
            for a, b in combinations(present, 2):
                edge_key = (min(a, b), max(a, b))
                edge_counts[edge_key] += 1
        for min_cooccur in [2, 1]:
            G_cooccur_all = ig.Graph(directed=False)
            G_cooccur_all.add_vertices(all_patterns)
            edges_to_add = [(a, b) for (a, b), count in edge_counts.items()
                            if count >= min_cooccur]
            if edges_to_add:
                G_cooccur_all.add_edges(edges_to_add)
            if G_cooccur_all.ecount() > 0:
                print(f"  Using min_cooccur={min_cooccur}: "
                      f"{G_cooccur_all.ecount()} edges")
                break
            else:
                print(f"  min_cooccur={min_cooccur} gives 0 edges, trying lower...")
        print(f"  Co-occurrence graph: {G_cooccur_all.vcount()} nodes, "
              f"{G_cooccur_all.ecount()} edges")
        full_coloring, full_n_colors, full_weight = run_coloring(
            G_cooccur_all, pattern_to_idx, p1_idx, p2_idx, w_arr,
            n_patterns, label="full (no core)"
        )
        if full_coloring is None:
            print("  Coloring failed, returning single locus")
            locus_df = pd.DataFrame("", index=binary.index, columns=["0"])
            umi_df = pd.DataFrame(0.0, index=binary.index, columns=["0"])
            return locus_df, None, umi_df
        final_coloring = full_coloring
        final_n_colors = full_n_colors
        K = full_n_colors

    # ---- Build output using helper ----
    print(f"Building locus matrix with {K} loci...")
    locus_df, umi_df = _build_locus_and_umi(
        binary.index, binary_arr, all_patterns_arr,
        final_coloring, K, nUMI_adjusted
    )

    cliques = [set() for _ in range(K)]
    for node, color in final_coloring.items():
        cliques[color].add(node)
    cliques = [c for c in cliques if len(c) > 0]

    print(f"  Loci sizes: {[len(c) for c in cliques]}")
    print("Done!")
    return locus_df, cliques, umi_df


def analyze_tape(df_filter, tapebc="", min_umi=2, min_cells=5, plot=False,
                 outdir="", name=""):
    time1 = timeit.default_timer()
    run_dir = os.path.join(outdir, name)
    log_dir = os.path.join(run_dir, "stdout")
    logfile = os.path.join(log_dir, "{}.txt".format(tapebc))

    with open(logfile, 'w') as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            try:
                results = gen_patterns(
                    df_filter, tapebc, min_umi, plot,
                    outdir=outdir, name=name
                )

                if results[0] == "normal":
                    _, binary, log_umi, has_ety, nUMI_adjusted = results

                    partition_results = find_graph_and_partition(
                        binary, log_umi,
                        nUMI_adjusted=nUMI_adjusted,
                        min_cells=min_cells, plot=plot,
                        outdir=outdir, name=name
                    )

                    locus_df = partition_results[0]
                    # partition_results[1] is cliques
                    umi_df = partition_results[2]

                    locus_df.columns = [
                        tapebc + "-" + str(i) for i in locus_df.columns
                    ]
                    umi_df.columns = [
                        tapebc + "-" + str(i) for i in umi_df.columns
                    ]

                    # Re-insert ETY-only cells that were dropped from binarization.
                    # These cells had this TapeBC but only as unedited (ETY) — they
                    # belong in the output as ETY at every locus.
                    if has_ety is not None:
                        missing_cells = has_ety.index.difference(locus_df.index)
                        if len(missing_cells) > 0:
                            empty_locus = pd.DataFrame(
                                "", index=missing_cells, columns=locus_df.columns
                            )
                            empty_umi = pd.DataFrame(
                                0.0, index=missing_cells, columns=umi_df.columns
                            )
                            locus_df = pd.concat([locus_df, empty_locus])
                            umi_df = pd.concat([umi_df, empty_umi])

                    # Fill unassigned loci with ETY + evenly distributed ETY UMI
                    if has_ety is not None:
                        for cell in locus_df.index:
                            if cell in has_ety.index and bool(has_ety.loc[cell]):
                                empty_cols = [ci for ci, col in enumerate(locus_df.columns)
                                              if locus_df.loc[cell, col] == ""]
                                n_empty = len(empty_cols)
                                if n_empty == 0:
                                    continue
                                for ci in empty_cols:
                                    col = locus_df.columns[ci]
                                    locus_df.loc[cell, col] = "ETY"
                                    if (nUMI_adjusted is not None and
                                            cell in nUMI_adjusted.index and
                                            "ETY" in nUMI_adjusted.columns):
                                        umi_df.loc[cell, umi_df.columns[ci]] = \
                                            nUMI_adjusted.loc[cell, "ETY"] / n_empty

                else:
                    # Early return: results = ("early", locus_df, umi_df)
                    _, locus_df, umi_df = results

                time2 = timeit.default_timer()
                print("Tape analyzed in: {}".format(
                    str(timedelta(seconds=time2 - time1))
                ))
                print("-" * 67)

            except Exception as e:
                print(f"Error processing tape {tapebc}: {e}")
                traceback.print_exc()
                return (pd.DataFrame(), pd.DataFrame())

    return (locus_df, umi_df)


def main(csv_path="", umi_cutoff=2, min_umi=2, min_cells=5, plot=False,
         outdir=".", name="", edit_key="GGAT", use_rpu=False,
         num_sites=6, seed=RANDOM_SEED):
    global EDIT_KEY, EDIT_KEY_LEN, RANDOM_SEED
    EDIT_KEY = edit_key
    EDIT_KEY_LEN = len(EDIT_KEY)
    # Reset all RNGs at run start so results are reproducible regardless of
    # import timing or repeated main() calls in one process.
    RANDOM_SEED = seed
    set_seeds(seed)

    run_dir = os.path.join(outdir, name)
    log_dir = os.path.join(run_dir, "stdout")
    os.makedirs(log_dir, exist_ok=True)

    df_filter = read_csv(csv_path, umi_cutoff=1)
    site_cols = [i for i in df_filter.columns if 'Site' in i]

    print("Parsing edit patterns from site columns...")
    df_filter['Pattern'] = df_filter.apply(
        lambda row: parse_sites(row, site_cols), axis=1
    )

    if use_rpu:
        print("Computing reads/UMI threshold...")
        log_rpu = np.log(df_filter['nRead'].values / df_filter['nUMI'].values + 0.01)
        log_rpu_df = pd.DataFrame(log_rpu, columns=['log_rpu'])
        rpu_threshold, kde_x, kde_smooth = compute_kde_threshold(log_rpu_df, log=False)

        rpu_natural = np.exp(rpu_threshold)
        rpu_natural = np.clip(rpu_natural, RPU_FLOOR, RPU_CEILING)
        rpu_threshold = np.log(rpu_natural)

        print(f"  Reads/UMI threshold (log scale): {rpu_threshold:.3f}")
        print(f"  Reads/UMI threshold (natural scale): {rpu_natural:.2f}")
        print(f"  i.e., requiring >= ~{rpu_natural:.1f} reads per UMI")

        if plot:
            plt.hist(log_rpu, density=True, bins=50, alpha=0.7)
            plt.plot(kde_x, kde_smooth, color='red')
            plt.axvline(rpu_threshold, color='blue', linestyle='--',
                        label=f'threshold={rpu_threshold:.2f}')
            plt.title('Reads per UMI distribution')
            plt.ylabel("Density")
            plt.xlabel("Log(reads/UMI)")
            plt.legend()
            plt.savefig(os.path.join(log_dir, "reads_per_umi.svg"), dpi=300)
            plt.show()
            plt.clf()

        log_rpu_values = np.log(df_filter['nRead'].values / df_filter['nUMI'].values + 0.01)
        df_filter['is_real'] = (log_rpu_values >= rpu_threshold) & (df_filter['nUMI'] >= umi_cutoff)
    else:
        print("RPU filtering disabled, using UMI cutoff only...")
        df_filter['is_real'] = df_filter['nUMI'] >= umi_cutoff

    n_real = df_filter['is_real'].sum()
    n_noise = (~df_filter['is_real']).sum()
    print(f"  Real rows: {n_real}, Noise rows: {n_noise}")

    tape_list = np.unique(df_filter['TargetBC'].values)
    print(f"Processing {len(tape_list)} TapeBCs in parallel...")

    actual_n_jobs = int(os.environ.get('NSLOTS', os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    results_list = Parallel(n_jobs=actual_n_jobs, prefer="processes")(
        delayed(analyze_tape)(
            df_filter, i, min_umi=min_umi,
            min_cells=min_cells, plot=plot,
            outdir=outdir, name=name
        )
        for i in tape_list
    )

    # Separate locus and UMI DataFrames
    locus_list = [r[0] for r in results_list if len(r[0]) > 0]
    umi_list = [r[1] for r in results_list if len(r[1]) > 0]

    full_df = pd.concat(locus_list, axis='columns', join='outer')
    full_df = full_df.fillna("")
    full_df = full_df.replace("", np.nan)
    full_df = full_df.dropna(axis=1, how='all')
    full_df = full_df.fillna("")

    # EXPRESSION OUTPUT: combine UMI DataFrames
    expr_df = pd.concat(umi_list, axis='columns', join='outer')
    expr_df = expr_df.fillna(0.0)
    # Keep only columns that exist in full_df (same loci)
    # Column names match: TapeBC-LocusIdx
    # expr_df columns are TapeBC-LocusIdx, full_df columns are TapeBC-LocusIdx
    # After site decomposition, full_df columns become TapeBC-LocusIdx-SiteN
    # But expr_df stays at locus level — one value per locus
    # Align to same column order as full_df (locus-level names)
    locus_level_cols = full_df.columns.tolist()
    expr_df = expr_df.reindex(columns=locus_level_cols, fill_value=0.0)
    expr_df = expr_df.reindex(index=full_df.index, fill_value=0.0)

    # Decompose patterns into per-site columns (vectorized)
    max_sites = num_sites

    site_df = pd.DataFrame(index=full_df.index)
    for j in full_df.columns:
        split_series = full_df[j].apply(
            lambda v: v.split(WORD_DELIMITER) if v not in ("", "ETY") else [v]
        )
        for i in range(max_sites):
            column = j + "-Site" + str(i + 1)
            site_df[column] = split_series.apply(
                lambda words: words[i] if i < len(words) and words[0] not in ("", "ETY")
                else ("ETY" if i == 0 and len(words) > 0 and words[0] == "ETY" else "")
            )
    site_df = site_df.replace("", "None")

    return site_df, expr_df


if __name__ == "__main__":
    time1 = timeit.default_timer()
    sys.setrecursionlimit(100000)

    parser = argparse.ArgumentParser(description="DNA Typewriter locus inference pipeline")
    parser.add_argument('-i', '--input_csv', help="Input csv path", required=True)
    parser.add_argument('-uc', '--umi_cutoff', help="UMI cutoff", required=False, default=2, type=int)
    parser.add_argument('-mu', '--min_umi', help="Minimum UMI for pattern inclusion",
                        required=False, default=2, type=int)
    parser.add_argument('-mc', '--min_cells', help="Minimum cells for core patterns",
                        required=False, default=5, type=int)
    parser.add_argument('-imp', '--impute', help="For backwards compatibility",
                        required=False, default=False, type=bool)
    parser.add_argument('-o', '--output_csv', help="Output csv filename", required=True)
    parser.add_argument('-d', '--outdir', help="Output directory (default: current directory)",
                        required=False, default=".")
    parser.add_argument('-p', '--plot', help="Produce plots (boolean)",
                        required=False, default="False")
    parser.add_argument('-n', '--folder_name', help="Name for log subfolder",
                        required=True)
    parser.add_argument('--use_rpu', help="Use reads-per-UMI filtering (default: False)",
                        required=False, default="False")
    parser.add_argument('-ns', '--num_sites', help="Number of sites per tape",
                        required=False, default=6, type=int)
    parser.add_argument('-ek', '--edit_key', help="Edit key suffix to strip from sites (default: GGAT)",
                        required=False, default="GGAT")
    # EXPRESSION OUTPUT: new CLI argument
    parser.add_argument('-e', '--expression_csv',
                        help="Output expression (adjusted UMI) CSV filename",
                        required=False, default="")

    argument = parser.parse_args()

    csv_path = str(argument.input_csv)
    umi_cutoff = int(argument.umi_cutoff)
    min_umi = int(argument.min_umi)
    min_cells = int(argument.min_cells)
    outdir = str(argument.outdir)
    output_csv = str(argument.output_csv)
    name = str(argument.folder_name)

    if str(argument.plot).lower() == "true":
        plot = True
    else:
        plot = False

    if str(argument.use_rpu).lower() == "true":
        use_rpu = True
    else:
        use_rpu = False

    EDIT_KEY = str(argument.edit_key)
    EDIT_KEY_LEN = len(EDIT_KEY)
    print(f"Edit key: {EDIT_KEY} (length {EDIT_KEY_LEN})")

    os.makedirs(outdir, exist_ok=True)
    run_dir = os.path.join(outdir, name)
    out_path = os.path.join(run_dir, output_csv)

    site_df, expr_df = main(
        csv_path=csv_path, umi_cutoff=umi_cutoff,
        min_umi=min_umi, min_cells=min_cells,
        plot=plot, outdir=outdir,
        name=name, edit_key=EDIT_KEY,
        use_rpu=use_rpu, num_sites=int(argument.num_sites)
    )

    site_df.to_csv(out_path)
    print(f"Output written to: {out_path}")

    # EXPRESSION OUTPUT: save expression matrix
    if argument.expression_csv:
        expr_path = os.path.join(run_dir, str(argument.expression_csv))
        expr_df.to_csv(expr_path)
        print(f"Expression matrix written to: {expr_path}")
        print(f"  Shape: {expr_df.shape}")
        print(f"  Non-zero entries: {(expr_df > 0).sum().sum()}")
        print(f"  Mean UMI (non-zero): {expr_df.values[expr_df.values > 0].mean():.1f}")

    time2 = timeit.default_timer()
    total_time = time2 - time1

    print("Script executed in: {}".format(str(timedelta(seconds=total_time))))

    os._exit(0)
