import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import combinations, groupby
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
import seaborn as sns
from sklearn.neighbors import KernelDensity, BallTree
from scipy.signal import find_peaks
import networkx as nx
import os
import contextlib
import traceback
from sklearn.neighbors import KNeighborsClassifier
import sys
import argparse
import timeit
from datetime import timedelta
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
from multiprocessing import cpu_count

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 500
plt.rcParams['svg.fonttype'] = 'none'

import random
import functools
print = functools.partial(print, flush=True)

np.random.seed(42)
random.seed(42)

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


def compute_kde_threshold(log_umi, bw=0.15, rel_height=0.25, log=True):
    data = np.ndarray.flatten(log_umi.values)
    kde = KernelDensity(bandwidth=bw)
    kde.fit(data.reshape(-1, 1))
    x_bin = np.histogram(data, bins=10)[1]
    kde_x = np.linspace(min(x_bin) - 0.50, max(x_bin) + 0.50, 500)
    if log:
        kde_density = kde.score_samples(kde_x.reshape(-1, 1))
    else:
        kde_density = np.exp(kde.score_samples(kde_x.reshape(-1, 1)))
    peaks, properties = find_peaks(kde_density, width=(None, None), rel_height=rel_height)
    # Use the first two peaks (sorted by position) to find the valley between
    # absence (zero mode) and presence (first signal mode).
    # This correctly separates noise from signal regardless of expression level.
    modes = np.sort(peaks[:2]) if len(peaks) >= 2 else peaks
    try:
        new_range = kde_density[modes[0]:modes[1]]
        new_min = np.argmin(new_range) + modes[0]
        new_min = kde_x[new_min]
    except:
        new_min = 0.0
    return new_min, kde_x, kde_density


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


def redistribute_noise_to_real(test_real, test_noise):
    if len(test_noise) == 0 or len(test_real) == 0:
        print("  No noise redistribution needed")
        return test_real
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
    G = nx.Graph()
    G.add_nodes_from(patterns)
    for i in cluster_sig:
        present = [col for col, val in zip(binary_columns, cluster_sig[i]) if val > 0.5]
        present_in_graph = [p for p in present if p in patterns]
        for a, b in combinations(present_in_graph, 2):
            G.add_edge(a, b)
    return G


# =========================================================================
# Graph coloring
# =========================================================================

def _coloring_batch(G_cooccur_adj, nodes, pattern_to_idx, p1_idx, p2_idx, w_arr,
                    n_patterns, batch_size, current_best_clique, current_best_sum):
    import networkx as nx
    G_cooccur = nx.from_dict_of_dicts(G_cooccur_adj)
    batch_best_clique = current_best_clique
    batch_best_sum = current_best_sum
    batch_best_coloring = None
    for _ in range(batch_size):
        coloring = nx.greedy_color(G_cooccur, strategy='random_sequential', interchange=False)
        n_colors = max(coloring.values()) + 1 if coloring else 0
        if n_colors <= batch_best_clique:
            color_arr = np.full(n_patterns, -1, dtype=np.int32)
            for node_str, c in coloring.items():
                if node_str in pattern_to_idx:
                    color_arr[pattern_to_idx[node_str]] = c
            c1 = color_arr[p1_idx]
            c2 = color_arr[p2_idx]
            mask = (c1 == c2) & (c1 >= 0)
            weight = float(w_arr[mask].sum())
            if n_colors < batch_best_clique or weight > batch_best_sum:
                batch_best_clique = n_colors
                batch_best_sum = weight
                batch_best_coloring = dict(coloring)
    return batch_best_clique, batch_best_sum, batch_best_coloring


def run_coloring(G_cooccur, pattern_to_idx, p1_idx, p2_idx, w_arr, n_patterns, label=""):
    print(f"Finding best graph partition ({label})...")
    nodes = list(G_cooccur.nodes())
    n = len(nodes)
    if n == 0:
        return None, 0, 0
    if n == 1:
        return {nodes[0]: 0}, 1, 0
    total_iterations = n * 2500
    batch_size = 1000
    best_clique = n + 1
    best_sum = 0
    best_coloring = None
    patience = max(50000, n * 500)
    iters_since_improvement = 0
    total_processed = 0
    G_cooccur_adj = nx.to_dict_of_dicts(G_cooccur)
    while total_processed < total_iterations:
        iters_this_round = min(batch_size, total_iterations - total_processed)
        batch_clique, batch_sum, batch_coloring = _coloring_batch(
            G_cooccur_adj, nodes, pattern_to_idx, p1_idx, p2_idx, w_arr,
            n_patterns, iters_this_round, best_clique, best_sum
        )
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
            index=np.unique(cells), columns=[tapebc + "-0"]
        ).fillna("")
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
            index=np.unique(cells), columns=[tapebc + "-0"]
        ).fillna("")
        umi_df = pd.DataFrame(
            0.0, index=np.unique(cells), columns=[tapebc + "-0"]
        )
        return ("early", locus_df, umi_df)

    print("Computing consensus patterns...")
    pattern_counts = test_real.groupby('Pattern')['nUMI'].sum().sort_values(ascending=False)
    unique_patterns = list(pattern_counts.index)

    if len(unique_patterns) == 0:
        locus_df = pd.DataFrame(
            index=np.unique(cells), columns=[tapebc + "-0"]
        ).fillna("")
        umi_df = pd.DataFrame(
            0.0, index=np.unique(cells), columns=[tapebc + "-0"]
        )
        return ("early", locus_df, umi_df)

    print(f"  Unique patterns (including ETY): {len(unique_patterns)}")
    for p in unique_patterns:
        print(f"    {p}: {pattern_counts[p]} UMIs")

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
        locus_df = pd.DataFrame(index=all_cells, columns=[col_name]).fillna("ETY")
        umi_df = pd.DataFrame(0.0, index=all_cells, columns=[col_name])
        if "ETY" in nUMI.columns:
            for cell in all_cells:
                if cell in nUMI.index:
                    umi_df.loc[cell, col_name] = nUMI.loc[cell, "ETY"]
        return ("early", locus_df, umi_df)

    if len(non_ety_patterns) == 1:
        all_cells = np.unique(cells)
        col_name = tapebc + "-0"
        locus_df = pd.DataFrame(index=all_cells, columns=[col_name]).fillna("")
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

    has_ety = None
    if "ETY" in nUMI.columns:
        has_ety = nUMI["ETY"] > 0

    # EXPRESSION OUTPUT: build adjusted UMI matrix
    # nUMI_for_binary now has shadow-redistributed values for non-ETY patterns
    # Combine with ETY from original nUMI
    nUMI_adjusted = nUMI_for_binary.copy()
    if "ETY" in nUMI.columns:
        nUMI_adjusted["ETY"] = nUMI.loc[nUMI_for_binary.index, "ETY"].values

    return ("normal", binary, log_umi, has_ety, nUMI_adjusted)


def find_graph_and_partition(binary, log_umi, nUMI_adjusted=None,
                             min_cells=5, plot=False,
                             outdir="", name="", impute=False):
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

        dist = pairwise_distances(binary_for_clustering, metric='hamming')
        dist = dist * binary_for_clustering.shape[1]
        dist = dist.astype(int)
        clustering = DBSCAN(
            eps=0.5, min_samples=min_cells, metric='precomputed', n_jobs=-1
        ).fit_predict(dist)

        assign = pd.Series(clustering, index=binary_for_clustering.index).astype(str)
        cluster_unique = [c for c in np.unique(assign.values) if c != '-1']
        n_noise = (assign == '-1').sum()
        print(f"  Number of DBSCAN clusters: {len(cluster_unique)}")
        print(f"  DBSCAN noise cells: {n_noise}")

        cluster_sig = {}
        for i in cluster_unique:
            index = assign[assign == i].index
            a = np.mean(binary_for_clustering.loc[index, :], axis=0)
            cluster_sig[i] = a.values

        print("Constructing core co-occurrence graph...")
        G_cooccur_core = build_cooccurrence_graph_from_clusters(
            core_patterns, cluster_sig, binary_for_clustering.columns
        )
        n_cluster_edges = len(G_cooccur_core.edges())
        print(f"  Core co-occurrence graph (from clusters): "
              f"{len(G_cooccur_core.nodes())} nodes, {n_cluster_edges} edges")
        cell_edge_counts = Counter()
        for i in range(len(binary_for_clustering.index)):
            present = [p for p, val in zip(core_patterns, binary_for_clustering.values[i]) if val > 0.5]
            if len(present) >= 2:
                for a, b in combinations(present, 2):
                    cell_edge_counts[(min(a, b), max(a, b))] += 1
        n_augmented = 0
        for (a, b), count in cell_edge_counts.items():
            if count >= 2 and not G_cooccur_core.has_edge(a, b):
                G_cooccur_core.add_edge(a, b)
                n_augmented += 1
        if n_augmented > 0:
            print(f"  Augmented core graph with {n_augmented} cell-level co-occurrence edges (>=2 cells)")
        print(f"  Final core co-occurrence graph: {len(G_cooccur_core.nodes())} nodes, {len(G_cooccur_core.edges())} edges")

        core_pattern_to_idx, core_p1, core_p2, core_w = precompute_pair_weights(core_patterns)
        n_core = len(core_patterns)

        core_coloring, K, core_weight = run_coloring(
            G_cooccur_core, core_pattern_to_idx, core_p1, core_p2, core_w,
            n_core, label="core"
        )

        if core_coloring is None:
            print("  Core coloring failed, returning single locus")
            locus_df = pd.DataFrame(index=binary.index, columns=["0"]).fillna("")
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
        G_cooccur_full.add_nodes_from(peripheral_patterns)

        peripheral_set = set(peripheral_patterns)
        for i in range(len(binary.index)):
            present = all_patterns_arr[binary_arr[i]]
            if len(present) < 2:
                continue
            has_peripheral = any(p in peripheral_set for p in present)
            if not has_peripheral:
                continue
            for a, b in combinations(present, 2):
                if a in peripheral_set or b in peripheral_set:
                    G_cooccur_full.add_edge(a, b)

        print(f"  Full co-occurrence graph: {len(G_cooccur_full.nodes())} nodes, "
              f"{len(G_cooccur_full.edges())} edges")

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
            G_cooccur_all = nx.Graph()
            G_cooccur_all.add_nodes_from(all_patterns)
            for (a, b), count in edge_counts.items():
                if count >= min_cooccur:
                    G_cooccur_all.add_edge(a, b)
            if len(G_cooccur_all.edges()) > 0:
                print(f"  Using min_cooccur={min_cooccur}: "
                      f"{len(G_cooccur_all.edges())} edges")
                break
            else:
                print(f"  min_cooccur={min_cooccur} gives 0 edges, trying lower...")
        print(f"  Co-occurrence graph: {len(G_cooccur_all.nodes())} nodes, "
              f"{len(G_cooccur_all.edges())} edges")
        full_coloring, full_n_colors, full_weight = run_coloring(
            G_cooccur_all, pattern_to_idx, p1_idx, p2_idx, w_arr,
            n_patterns, label="full (no core)"
        )
        if full_coloring is None:
            print("  Coloring failed, returning single locus")
            locus_df = pd.DataFrame(index=binary.index, columns=["0"]).fillna("")
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
                 outdir="", name="", impute=False):
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
                        outdir=outdir, name=name, impute=impute
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

                    # Fill unassigned loci with ETY + evenly distributed ETY UMI
                    if has_ety is not None:
                        for cell in locus_df.index:
                            if cell in has_ety.index and has_ety.get(cell, False):
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
         outdir=".", name="", impute=False, edit_key="GGAT", use_rpu=False):
    global EDIT_KEY, EDIT_KEY_LEN
    EDIT_KEY = edit_key
    EDIT_KEY_LEN = len(EDIT_KEY)

    run_dir = os.path.join(outdir, name)
    log_dir = os.path.join(run_dir, "stdout")
    os.makedirs(log_dir, exist_ok=True)

    df_filter = read_csv(csv_path, umi_cutoff=1)
    site_cols = [i for i in df_filter.columns if 'Site' in i]

    print("Parsing edit patterns from site columns...")
    df_filter['Pattern'] = df_filter.apply(
        lambda row: parse_sites(row, site_cols), axis=1
    )

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

    n_real = df_filter['is_real'].sum()
    n_noise = (~df_filter['is_real']).sum()
    print(f"  Real rows: {n_real}, Noise rows: {n_noise}")

    tape_list = np.unique(df_filter['TargetBC'].values)
    print(f"Processing {len(tape_list)} TapeBCs in parallel...")

    actual_n_jobs = cpu_count()
    results_list = Parallel(n_jobs=actual_n_jobs, prefer="processes")(
        delayed(analyze_tape)(
            df_filter, i, min_umi=min_umi,
            min_cells=min_cells, plot=plot,
            outdir=outdir, name=name, impute=impute
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
    max_sites = full_df.apply(
        lambda col: col[~col.isin(["", "ETY"])].apply(
            lambda v: len(v.split(WORD_DELIMITER))
        ).max()
    ).max()
    max_sites = int(max_sites) if not pd.isna(max_sites) else 6

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
    parser.add_argument('-o', '--output_csv', help="Output csv filename", required=True)
    parser.add_argument('-d', '--outdir', help="Output directory (default: current directory)",
                        required=False, default=".")
    parser.add_argument('-p', '--plot', help="Produce plots (boolean)",
                        required=False, default="False")
    parser.add_argument('-n', '--folder_name', help="Name for log subfolder",
                        required=True)
    parser.add_argument('--impute', help="Impute orphan cells via KNN (boolean)",
                        required=False, default="False")
    parser.add_argument('-ek', '--edit_key', help="Edit key suffix to strip from sites (default: GGAT)",
                        required=True, default="GGAT")
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

    if str(argument.impute).lower() == "true":
        impute = True
    else:
        impute = False

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
        name=name, impute=impute
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
