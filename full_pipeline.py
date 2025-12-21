import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from statistics import mode
from itertools import zip_longest
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from scipy.spatial.distance import hamming
import networkx as nx
import os
from itertools import groupby
from sklearn.neighbors import KNeighborsClassifier
import sys
import argparse
import timeit
from datetime import timedelta
from sklearn.manifold import TSNE

from DBSCANPP import DBSCANPP
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from joblib import Parallel, delayed
from multiprocessing import cpu_count

#sys.stdout = sys.__stdout__

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 500
plt.rcParams['svg.fonttype'] = 'none'


import functools
print = functools.partial(print, flush=True)

sns.set_style("white")


def read_csv(csv_path = "", umi_cutoff = 3):
    df = pd.read_csv(csv_path, header=0)
    #df.columns[-1] = "nUMI"
    #columns = list(df.columns)
    #columns[-1] = "nUMI"
    #df.columns = columns
    df_filter = df[df['nUMI']>=umi_cutoff]
    df_filter.fillna("", inplace=True)
    return df_filter


def gen_patterns(df_filter, tapebc = "", min_umi=100, plot=False, outdir="", name=""):
    #print(min_umi)
    print("Tape BC: {}".format(tapebc))
    test = df_filter[df_filter['TargetBC']==tapebc]
    seqs_orig = test[[i for i in test.columns if 'Site' in i]].apply(lambda row: ''.join(row.values.astype(str)), axis=1).values
    cells = list(test['Cell'].values)
    
    print(test.shape)
    print(test)
    print(seqs_orig[:10])
    
    # Clustering sequences and detecting consensus patterns
    seqs = []
    for i in range(len(seqs_orig)):
        seqs += [seqs_orig[i]]*test['nUMI'].values[i]

    '''
    ex = np.array([list(i) for i in seqs_orig])
    ex_num = ex.copy()
    ex_num[ex_num=="A"] = 0
    ex_num[ex_num=="T"] = 1
    ex_num[ex_num=="G"] = 2
    ex_num[ex_num=="C"] = 3
    ex_num[ex_num=="N"] = 4 #maybe change this later?
    ex_num = ex_num.astype(float)
    print(ex_num.shape)

    print("Computing pairwise hamming distance matrix...") 
    dist = pairwise_distances(ex_num, metric='hamming', n_jobs=-1)
    dist = dist*ex_num.shape[1]
    dist = dist.astype(int)

    ####
    #dist = csr_matrix(dist)
    ####
    '''

    '''
    print("One-hot encoding strings...")
    print(len(seqs_orig))
    print(len(seqs))
    char_array = np.array([list(string) for string in seqs])
    print(char_array.shape)
    encoder = OneHotEncoder(dtype="double", sparse_output=False)
    ex_num = encoder.fit_transform(char_array)
    print(type(ex_num))
    print(ex_num.shape)
    '''
    
    
    #print(dist)

    print("Clustering UMI sequences...")
    '''
    weights = test['nUMI'].values
    clustering = DBSCAN(eps=0.5, min_samples=min_umi, metric='precomputed', n_jobs=-1).fit_predict(dist, sample_weight=weights)
    '''

    '''
    dbscanpp = DBSCANPP(p=0.1, eps_density=0.0, eps_clustering=np.sqrt(2)+0.01, minPts=min_umi)
    clustering = dbscanpp.fit_predict(ex_num, init="k-centers", cluster_outliers=False)
    '''

    seq_counts = pd.DataFrame(pd.Series(Counter(seqs)).sort_values(ascending=False))
    seq_counts['cluster'] = list(range(seq_counts.shape[0]))
    seq_counts.loc[seq_counts[seq_counts[0]<min_umi].index, "cluster"] = -1
    print(len(seqs_orig))
    print(len(seqs))
    seq_counts['cluster'] = seq_counts['cluster'].astype(str)
    clustering = seq_counts.loc[seqs_orig,'cluster'].values
    clustering_expand = seq_counts.loc[seqs,'cluster'].values

    
    ######
    #clustering = clustering.astype(str)
    print("Number of noisy sequences: {}".format(list(clustering_expand).count("-1")))
    cluster_names = list(np.unique(clustering))
    try:
        cluster_names.remove('-1')
    except:
        pass

    ####
    '''
    clustering_expand = clustering.copy()
    clust_series = pd.Series(clustering_expand, index=seqs)
    clust_series = clust_series[~clust_series.index.duplicated(keep='first')]
    print("length clust_series: {}".format(len(clust_series)))
    '''
    print("length seqs_orig: {}".format(len(seqs_orig)))
    print("length seqs: {}".format(len(seqs)))
    print("length unique seqs: {}".format(len(np.unique(seqs_orig))))
    #clustering = clust_series.loc[seqs_orig].values
    print("length clustering: {}".format(len(clustering)))
    '''
    clustering_expand = []
    for i in range(len(seqs_orig)):
        clustering_expand += [clustering[i]]*test['nUMI'].values[i]
    '''
    
    print("Computing consensus per cluster...")
    consensus = {}
    for j in cluster_names:
        cluster_num = np.array(seqs)[np.where(np.array(clustering_expand)==j)[0]]
        maxOccurs = "".join(mode("".join(chars)) for chars in zip_longest(*cluster_num, fillvalue=""))
        consensus[j] = maxOccurs
        ######
        #print(cluster_num)
        #print(maxOccurs)
        cluster_num = list(cluster_num)
        cons_count = cluster_num.count(maxOccurs)
        purity = cons_count/len(cluster_num)
        print("Cluster {} purity: {}".format(j, purity))
        if purity < 0.5:
            print(maxOccurs)
            print(Counter(cluster_num))
        #break

    for j in consensus:
        try:
            sub = consensus[j][:consensus[j].rindex("GGAT")]
            sub = sub.split("GGAT")
            sub = [i for i in sub if len(i)==3]
            #sub = "|".join(sub)
            sub = "".join(sub)
        except:
            sub = "ETY"
        consensus[j] = sub
    
    if plot:
        print(consensus)
        print(np.unique(clustering_expand))
        consensus_new = consensus.copy()
        consensus_new["-1"] = "Noise"
        
        ex_plot = np.array([list(i) for i in seqs])
        ex_num_plot = ex_plot.copy()
        ex_num_plot[ex_num_plot=="A"] = 0
        ex_num_plot[ex_num_plot=="T"] = 1
        ex_num_plot[ex_num_plot=="G"] = 2
        ex_num_plot[ex_num_plot=="C"] = 3
        ex_num_plot[ex_num_plot=="N"] = 4 #maybe change this later?
        ex_num_plot = ex_num_plot.astype(float)
        dist_plot = pairwise_distances(ex_num_plot, metric='hamming', n_jobs=-1)
        dist_plot = dist_plot*ex_num_plot.shape[1]
        dist_plot = dist_plot.astype(int)

        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3, metric='precomputed').fit_transform(dist_plot)
        plot_df = pd.DataFrame(X_embedded, columns=['TSNE1','TSNE2'])
        plot_df['Consensus'] = clustering_expand
        plot_df['Consensus'] = plot_df['Consensus'].map(consensus_new)
        sns.scatterplot(data=plot_df, x="TSNE1", y="TSNE2", hue="Consensus", s=30)
        plt.legend(bbox_to_anchor=(1.0, 0.65), loc='upper left', borderaxespad=0, frameon=False)
        plt.title("Tape UMI clustering")
        plt.savefig("{}/stdout_{}/umi_clustering.svg".format(outdir,name), dpi=300)
        plt.show()
        plt.clf()
        
    
    test['Clustering'] = clustering
    test['Pattern'] = test['Clustering'].map(consensus)
    entries = test.dropna()

    #print(test)
    #print(entries)

    print("Counting nUMI per cell per pattern...")
    # Determining loci
    nUMI = pd.DataFrame(index=np.unique(entries['Cell']), columns = np.unique(entries['Pattern']))
    nUMI.fillna(0, inplace=True)
    for i in entries.index:
        nUMI.loc[entries.loc[i,'Cell'],entries.loc[i,'Pattern']] += entries.loc[i,'nUMI']
    if nUMI.shape[1] > 1:
        log_umi = nUMI.div(nUMI.sum(axis=1), axis=0)
    else:
        log_umi = nUMI.copy()
    log_umi = np.log(log_umi+0.1)-np.log(0.1)

    #print(log_umi)

    if nUMI.shape == (0,0) or nUMI.shape == (1,1) or nUMI.shape == (1,0):
        locus_df = pd.DataFrame(index=np.unique(cells), columns=[tapebc+"-0"]).fillna("")
        return locus_df
    
    data = np.ndarray.flatten(log_umi.values)
    '''
    if plot:
        plt.hist(data, bins=10)
        plt.show()
        plt.clf()
    '''
    bw = 0.15
    rel_height=0.25
    
    print("Binarizing matrix with KDE...")
    kde = KernelDensity(bandwidth=bw)#kernel='exponential')
    kde.fit(data.reshape(-1, 1))
    x_bin = np.histogram(data, bins=10)[1]
    kde_x = np.linspace(min(x_bin)-0.50,max(x_bin)+0.50,500)
    kde_smooth = kde.score_samples(kde_x.reshape(-1, 1))#np.exp(kde.score_samples(kde_x.reshape(-1, 1)))
    
    if plot:
        plt.hist(data, density=True, bins=10)
        plt.plot(kde_x, np.exp(kde_smooth), color='red')
        plt.title('Tape Expression')
        plt.ylabel("Log-density")
        plt.xlabel("Log-counts")
        plt.yscale('log')
        plt.savefig("{}/stdout_{}/tape_expression.svg".format(outdir,name), dpi=300)
        plt.show()
        plt.clf()
    
    peaks, properties = find_peaks(kde_smooth, width=(None,None), rel_height=rel_height) #height=0.005
    #print(properties)
    
    prop_array = properties['widths']#/(properties['peak_heights']+1)
    #print(prop_array)
    
    modes = np.argsort(prop_array)[-2:]
    modes = peaks[modes]
    modes = np.sort(modes)

    try:
        new_range = kde_smooth[modes[0]:modes[1]]
        new_min = np.argmin(new_range) + modes[0]
        new_min = kde_x[new_min]
    except:
        new_min = 0.0
    print(new_min)

    binary = np.where(log_umi>new_min, 1.0, 0.0)
    
    return binary, log_umi


def find_graph(binary, log_umi, min_cells=5, plot=False, outdir="", name=""):
    print(binary.shape)
    binary = pd.DataFrame(index=log_umi.index, columns=log_umi.columns, data=binary)
    #print(binary)
    keep_index = np.where(binary.apply(sum, axis=0)>=min_cells)[0]
    binary = binary.iloc[:,keep_index].copy()
    print(binary.shape)

    log_umi = log_umi.loc[binary.index, binary.columns].copy()

    '''
    godsnot_64 = [
    # "#000000",  # remove the black, as often, we have black colored annotation
    "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF",
    "#997D87", "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF",
    "#4A3B53", "#FF2F80", "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92",
    "#FF90C9", "#B903AA", "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299",
    "#300018", "#0AA6D8", "#013349", "#00846F", "#372101", "#FFB500", "#C2FFED",
    "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09", "#00489C", "#6F0062",
    "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66", "#885578",
    "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F",
    "#938A81", "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757",
    "#C8A1A1", "#1E6E00", "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C",
    "#772600", "#D790FF", "#9B9700", "#549E79", "#FFF69F", "#201625", "#72418F",
    "#BC23FF", "#99ADC0", "#3A2465", "#922329", "#5B4534", "#FDE8DC", "#404E55",
    "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C"
    ]
    '''
    
    dist = pairwise_distances(binary, metric='hamming')
    dist = dist*binary.shape[1]
    dist = dist.astype(int)
    #print(dist)
    clustering = DBSCAN(eps=0.5, min_samples=min_cells, metric='precomputed', n_jobs=-1).fit_predict(dist)
    
    #colormap = dict(zip(range(len(plt.cm.tab20.colors)), plt.cm.tab20.colors))
    #colormap = pd.Series(colormap)
    #colors = pd.Series(godsnot_64).iloc[clustering]#colormap.iloc[clustering]

    
    if plot:
        n_colors = len(np.unique(clustering))
        lut = dict(zip(set(clustering), sns.hls_palette(n_colors, l=0.5, s=0.8)))
        row_colors = pd.DataFrame(clustering)[0].map(lut)
        #print(row_colors)

        #print(lut)
        
        try:
            hmp = sns.clustermap(binary, cmap='Reds', figsize=(8,6), row_colors=row_colors.values, linewidth=.3)
        except:
            hmp = sns.clustermap(binary, cmap='Reds', figsize=(8,6), row_colors=row_colors.values, linewidth=.3, col_cluster=False)
        plt.savefig("{}/stdout_{}/binary_heatmap.svg".format(outdir,name), dpi=300)
        plt.show()
        plt.clf()

    
    assign = pd.Series(clustering, index=log_umi.index)
    assign = assign.astype(str)
    clustering = clustering.astype(str)
    
    cluster_unique = list(np.unique(clustering))
    try:
        cluster_unique.remove('-1')
    except:
        pass
    
    cluster_sig = {}
    for i in cluster_unique:
        index = assign[assign==i].index
        a = np.mean(binary.loc[index,:], axis=0)
        cluster_sig[i] = a.values
    
    # Impute on per-tape basis
    assign = assign.replace("-1",-1)
    #print(assign)
    pos = assign[assign!=-1].index
    neg = assign[assign==-1].index
    k_best = len(neg)
    if k_best>0:
        print("Imputing noisy data...")
        KNN = KNeighborsClassifier(n_neighbors=5, weights='distance')

        KNN.fit(log_umi.loc[pos,:], assign.loc[pos])
        cluster_pred = KNN.predict(log_umi.loc[neg,:])
        assign.loc[neg] = cluster_pred

        #print(assign)
        #del lut[-1]
        
        n_colors = len(np.unique(clustering))
        lut = dict(zip(set(clustering), sns.hls_palette(n_colors, l=0.5, s=0.8)))
        print(lut)
        
        row_colors = pd.DataFrame(assign.values).astype(str)[0].map(lut)
        print(row_colors)
        
        #print(assign)
        if plot:
            try:
                sns.clustermap(log_umi.iloc[hmp.dendrogram_row.reordered_ind,hmp.dendrogram_col.reordered_ind], 
                               cmap='Reds', figsize=(8,6), col_cluster=False, row_cluster=False, linewidth=.3, 
                               row_colors=row_colors.values[hmp.dendrogram_row.reordered_ind])
                #row_colors=row_colors.iloc[assign.values+1].values[hmp.dendrogram_row.reordered_ind])
            except:
                sns.clustermap(log_umi.iloc[hmp.dendrogram_row.reordered_ind,:], 
                               cmap='Reds', figsize=(8,6), col_cluster=False, row_cluster=False, linewidth=.3,
                               row_colors=row_colors.values[hmp.dendrogram_row.reordered_ind])
                #row_colors=row_colors.iloc[assign.values+1].values[hmp.dendrogram_row.reordered_ind])
            plt.savefig("{}/stdout_{}/exp_heatmap.svg".format(outdir,name), dpi=300)
            plt.show()
            plt.clf()
    else:
        if plot:
            n_colors = len(np.unique(clustering))
            lut = dict(zip(set(clustering), sns.hls_palette(n_colors, l=0.5, s=0.8)))
            print(lut)

            row_colors = pd.DataFrame(assign.values).astype(str)[0].map(lut)
            print(row_colors)
            
            try:
                sns.clustermap(log_umi.iloc[hmp.dendrogram_row.reordered_ind,hmp.dendrogram_col.reordered_ind], 
                               cmap='Reds', figsize=(8,6), col_cluster=False, row_cluster=False, linewidth=.3, 
                               row_colors=row_colors.values[hmp.dendrogram_row.reordered_ind])
                #row_colors=row_colors.iloc[assign.values+1].values[hmp.dendrogram_row.reordered_ind])
            except:
                sns.clustermap(log_umi.iloc[hmp.dendrogram_row.reordered_ind,:], 
                               cmap='Reds', figsize=(8,6), col_cluster=False, row_cluster=False, linewidth=.3,
                               row_colors=row_colors.values[hmp.dendrogram_row.reordered_ind])
            plt.savefig("{}/stdout_{}/exp_heatmap.svg".format(outdir,name), dpi=300)
            plt.show()
            plt.clf()

    #print(assign)
    cluster_sig = pd.DataFrame(cluster_sig)
    #print(cluster_sig)
    keep_index = np.where(cluster_sig.apply(sum, axis=1)>0)[0]
    cluster_sig = cluster_sig.iloc[keep_index,:].copy()
    cluster_sig = cluster_sig.to_dict(orient='list')
    binary = binary.iloc[:,keep_index].copy()
    cluster_sig = {i:np.array(cluster_sig[i]) for i in cluster_sig}

    print("Constructing graphs per cluster...")
    # Determine optimal graph structure of the patterns
    graph_list = []
    for i in cluster_sig:
        v = cluster_sig[i]
        v1 = pairwise_distances(v.reshape(-1,1), metric='cosine')
        G1 = nx.from_numpy_array(v1)
        G1 = nx.relabel_nodes(G1, dict(zip(G1.nodes(), binary.columns)))
        graph_list.append(G1)

    G = nx.intersection_all(graph_list)

    print("Nodes: {}".format(len(G.nodes())))
    
    edge_dict={}
    for e in G.edges():
        edge = [e[0].replace("|",""),e[1].replace("|","")]
        edge = tuple(edge)
        common = os.path.commonprefix(edge)
        edge_dict[e] = int(len(common)/3)
    
    nx.set_edge_attributes(G, edge_dict, 'similarity')

    '''
    print(list(G.nodes())[:10])
    node_map = {i:i.replace("|","") for i in G.nodes()}
    print(node_map)
    nx.relabel_nodes(G, node_map, copy=False) #### new
    print(list(G.nodes())[:10])
    '''
    
    if plot:
        plt.margins(x=0.4)
        try:
            pos = nx.spring_layout(G, k=0.5)#, k=0.85)
        except:
            pos = nx.spectral_layout(G)#, k=0.85)
        nx.draw(G, pos, with_labels=True)
        labels = nx.get_edge_attributes(G,'similarity')
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        plt.savefig("{}/stdout_{}/orig_graph.svg".format(outdir,name), dpi=300)
        plt.show()
        plt.clf()

    return G, assign, binary, cluster_sig


def find_partition_coloring(G, assign, binary, cluster_sig, plot=False, outdir="", name=""):
    print("Finding best graph partition...")

    '''
    Gc = nx.complement(G)
    set_list = set()
    graph_names = {}
    graph_attrs = {}
    best_clique = 1000
    best_sum = 0
    n = len(G.nodes())
    graph_ideal = None
    
    for k in range(n*1000):#2500):#int(n**np.e)):
        #print(k)
        coloring = nx.greedy_color(Gc, strategy='random_sequential', interchange=False)
        res = {i: [j[0] for j in j] for i, j in groupby(sorted(coloring.items(), key = lambda x : x[1]), lambda x : x[1])}
        res_set = list(res.values())
        res_set = frozenset({frozenset(i) for i in res_set})
        clique_num = len(res_set)
        #print(clique_num)
    
        if clique_num <= best_clique:# and size >= best_sum:
            G_ideal = nx.union_all([G.subgraph(i) for i in res_set])
            size = G_ideal.size("similarity")
            best_clique = clique_num
            if size > best_sum:
                best_sum = size
                graph_ideal = G_ideal.copy()
                #print("Best params: {},{}".format(clique_num,size))
            #graph_names[str(k)] = G_ideal.copy()
            #graph_attrs[str(k)] = [clique_num, size]
    '''

    #####################

    
    def process_single_iteration(Gc, G, k, best_clique, best_sum):
        """Process a single iteration and only return if it's better than current best"""
        coloring = nx.greedy_color(Gc, strategy='random_sequential', interchange=False)
        res = {i: [j[0] for j in j] for i, j in groupby(sorted(coloring.items(), key=lambda x: x[1]), lambda x: x[1])}
        res_set = list(res.values())
        res_set = frozenset({frozenset(i) for i in res_set})
        clique_num = len(res_set)
        
        # Only process further if this could be a better result
        if clique_num <= best_clique:
            G_ideal = nx.union_all([G.subgraph(i) for i in res_set])
            size = G_ideal.size("similarity")
            
            if size > best_sum:
                return clique_num, size, G_ideal
        
        return None
    
    def process_batch(Gc, G, start_k, batch_size, current_best_clique, current_best_sum):
        """Process a batch of iterations and return only the best result"""
        batch_best_clique = current_best_clique
        batch_best_sum = current_best_sum
        batch_best_graph = None
        
        for k in range(start_k, start_k + batch_size):
            result = process_single_iteration(Gc, G, k, batch_best_clique, batch_best_sum)
            if result is not None:
                clique_num, size, G_ideal = result
                if clique_num <= batch_best_clique and size > batch_best_sum:
                    batch_best_clique = clique_num
                    batch_best_sum = size
                    batch_best_graph = G_ideal.copy()
        
        return batch_best_clique, batch_best_sum, batch_best_graph
    
    def find_best_clique_parallel(G, batch_size=1000, n_jobs=-1):
        """
        Parallel implementation with batching to manage memory
        
        Parameters:
        - G: input graph
        - batch_size: number of iterations per batch
        - n_jobs: number of parallel jobs
        """
        Gc = nx.complement(G)
        n = len(G.nodes())
        total_iterations = 500000#n * 5000
        
        # Get actual number of jobs
        actual_n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        
        best_clique = 1000
        best_sum = 0
        graph_ideal = None
        
        # Process in batches
        for batch_start in range(0, total_iterations, batch_size * actual_n_jobs):
            remaining_iters = total_iterations - batch_start
            current_n_jobs = min(actual_n_jobs, (remaining_iters + batch_size - 1) // batch_size)
            
            # Create batch jobs
            batch_results = Parallel(n_jobs=actual_n_jobs)(
                delayed(process_batch)(
                    Gc, G, 
                    batch_start + i * batch_size,
                    min(batch_size, remaining_iters - i * batch_size),
                    best_clique,
                    best_sum
                )
                for i in range(current_n_jobs)
            )
            print("Batch size: {}".format(len(batch_results)))
            
            # Update best results from this batch
            for batch_clique, batch_sum, batch_graph in batch_results:
                if batch_graph is not None:  # Only if this batch found a better result
                    if batch_clique <= best_clique and batch_sum > best_sum:
                        best_clique = batch_clique
                        best_sum = batch_sum
                        graph_ideal = batch_graph
            
            # Optional: Print progress
            print(f"Processed up to iteration {min(batch_start + batch_size * actual_n_jobs, total_iterations)}/{total_iterations}")
        
        return graph_ideal, best_clique, best_sum

    graph_ideal, best_clique, best_sum = find_best_clique_parallel(G, batch_size=1000, n_jobs=-1)

    ###########################
    
    if plot:
        plt.margins(x=0.4)
        #pos = nx.spring_layout(graph_ideal, weight=None, k=0.85)
        #pos = nx.planar_layout(graph_ideal, scale=3.0)

        try:
            pos = nx.spring_layout(G, k=0.5)#, k=0.85)
        except:
            pos = nx.spectral_layout(G)#, k=0.85)
        
        #plt.figure(3,figsize=(12,12)) 
        #nx.draw(graph_ideal, pos, with_labels=True,node_size=120,font_size=10,width=0.5)
        nx.draw(graph_ideal, pos, with_labels=True)#, width=0.5)
        labels = nx.get_edge_attributes(graph_ideal,'similarity')
        nx.draw_networkx_edge_labels(graph_ideal,pos,edge_labels=labels)
        plt.savefig("{}/stdout_{}/color_graph.svg".format(outdir,name), dpi=300)
        plt.show()
        plt.clf()

    print("Best clique, best sum {},{}".format(best_clique, best_sum))
    print("Finding cliques...")
    cliques = list(nx.find_cliques(graph_ideal))
    print("Number of loci: {}".format(len(cliques)))
    print("Creating final matrix...")
    # Create cell X locus matrix
    locus_names = list(range(len(cliques)))
    locus_df = pd.DataFrame(index=assign.index, columns = locus_names)
    
    cluster_patterns = {}
    for i in cluster_sig:
        cluster_patterns[i] = list(binary.columns[np.array(cluster_sig[i].astype(bool))])
    
    patterns_df = assign.map(cluster_patterns)
    for i in locus_df.index:
        patterns = patterns_df[i]
        vector = []
        for j in cliques:
            vector.append(set(j).intersection(set(patterns)))
        vector = [list(i) for i in vector]
        vector = [i[0] if len(i)>0 else "" for i in vector]
        locus_df.loc[i,:] = vector
    
    #locus_df.columns = [tapebc + "-" + str(i) for i in locus_df.columns]
    print("Done!")

    return locus_df


def analyze_tape(df_filter, tapebc = "", min_umi=100, min_cells=5, plot=False, outdir="", name=""):
    time1 = timeit.default_timer()
    
    sys.stdout = open('{}/stdout_{}/{}.txt'.format(outdir,name,tapebc),'wt')
    
    results = gen_patterns(df_filter, tapebc, min_umi, plot, outdir=outdir, name=name)
    if type(results)==tuple:
        binary = results[0]
        log_umi = results[1]
        try:
            G, assign, binary, cluster_sig = find_graph(binary, log_umi, min_cells, plot, outdir=outdir, name=name)
            locus_df = find_partition_coloring(G, assign, binary, cluster_sig, plot, outdir=outdir, name=name)
            locus_df.columns = [tapebc + "-" + str(i) for i in locus_df.columns]
        except:
            return pd.DataFrame()
    else:
        locus_df = results

    time2 = timeit.default_timer()
    total_time = time2-time1
    
    print("Tape analyzed in: {}".format(str(timedelta(seconds=total_time))))
    print("-------------------------------------------------------------------")

    sys.stdout.close()
    return locus_df


def main(csv_path = "", umi_cutoff = 3, min_umi=100, min_cells=5, plot=False, outdir="", name=""):

    os.mkdir(outdir+"/stdout_{}/".format(name))
    
    df_filter = read_csv(csv_path, umi_cutoff)
    #print(df_filter)
    tape_list = np.unique(df_filter['TargetBC'].values)
    
    df_list = [analyze_tape(df_filter, i, min_umi=min_umi, 
                            min_cells=min_cells, plot=plot, 
                            outdir=outdir, name=name) for i in tape_list]

    sys.stdout = sys.__stdout__
    
    full_df = pd.concat(df_list, axis='columns', join='outer')
    full_df = full_df.fillna("")

    full_df = full_df.replace("", np.nan)
    full_df = full_df.dropna(axis=1, how='all')
    full_df = full_df.fillna("")

    site_df = pd.DataFrame(index=full_df.index)
    for j in full_df.columns:
        for i in range(6):
            column = j+"-Site"+str(i+1)
            for k in site_df.index:
                try:
                    site_df.loc[k,column] = full_df.loc[k,j][i*3:i*3+3]
                except:
                    site_df.loc[k,column] = ""
    site_df = site_df.replace("","None")
    
    return site_df


if __name__ == "__main__":
    time1 = timeit.default_timer()
    sys.setrecursionlimit(100000)

    parser = argparse.ArgumentParser(description = "Description for my parser")
    parser.add_argument('-i', '--input_csv', help = "Input csv path", required = True, default = "")
    parser.add_argument('-uc', '--umi_cutoff', help = "UMI cutoff", required = False, default = 3)
    parser.add_argument('-mu', '--min_umi', help = "Minimum UMI for clustering", required = False, default = 100)
    parser.add_argument('-mc', '--min_cells', help = "Minimum cells for clustering", required = False, default = 5)
    parser.add_argument('-o', '--output_csv', help = "Output csv path", required = True, default = "")
    parser.add_argument('-p', '--plot', help = "Produce plots (boolean)", required = False, default = False)
    parser.add_argument('-n', '--folder_name', help = "Output folder name", required = True, default = "")

    
    argument = parser.parse_args()
    
    csv_path=str(argument.input_csv)
    umi_cutoff=int(argument.umi_cutoff)
    min_umi=int(argument.min_umi)
    min_cells=int(argument.min_cells)
    out=str(argument.output_csv)
    name=str(argument.folder_name)
    
    if str(argument.plot).lower() == "true":
        plot=True
    else:
        plot=False

    site_df = main(csv_path=csv_path, umi_cutoff=umi_cutoff, 
                   min_umi=min_umi, min_cells=min_cells, 
                   plot=plot, outdir='/'.join(out.split('/')[:-1]),
                  name=name)
    
    site_df.to_csv(out)
    time2 = timeit.default_timer()
    total_time = time2-time1
    
    print("Script executed in: {}".format(str(timedelta(seconds=total_time))))
    






