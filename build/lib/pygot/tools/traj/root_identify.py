import ot as pot
import torch
from sklearn.neighbors import KNeighborsRegressor
from cellrank.kernels import CytoTRACEKernel
import scanpy as sc
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import minmax_scale
from copy import deepcopy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import pygot.external.palantir as palantir


def scale(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def highlight_extrema(adata, basis='umap', **kwargs):
    extrema_names = adata.uns['extrema']
    fig,ax = plt.subplots(1,1)
    sc.pl.embedding(adata, show=False, ax=ax, basis=basis, **kwargs)

    ax.scatter(adata[extrema_names].obsm['X_'+basis][:,0], adata[extrema_names].obsm['X_'+basis][:,1], label='extrema')
    plt.legend()

def calcu_w(D, sdv=None):
    if sdv is None:
        sdv = np.std(np.ravel(D)) * 1.06 * len(np.ravel(D)) ** (-1 / 5)
    W = np.exp(-0.5 * np.power((D / sdv), 2))
    return W, sdv


def _max_min_sampling(data, num_waypoints, seed=None):
    """Function for max min sampling of waypoints

    :param data: Data matrix along which to sample the waypoints,
                 usually diffusion components
    :param num_waypoints: Number of waypoints to sample
    :param seed: Random number generator seed to find initial guess.
    :return: pandas Series reprenting the sampled waypoints
    """

    waypoint_set = list()
    no_iterations = int((num_waypoints) / data.shape[1])
    if seed is not None:
        np.random.seed(seed)

    # Sample along each component
    N = data.shape[0]
    for ind in range(data.shape[1]):
        # Data vector
        vec = np.ravel(data[:,ind])

        # Random initialzlation
        iter_set = [
            np.random.randint(N),
        ]

        # Distances along the component
        dists = np.zeros([N, no_iterations])
        dists[:, 0] = abs(vec - data[iter_set, ind])
        for k in range(1, no_iterations):
            # Minimum distances across the current set
            min_dists = dists[:, 0:k].min(axis=1)

            # Point with the maximum of the minimum distances is the new waypoint
            new_wp = np.where(min_dists == min_dists.max())[0][0]
            iter_set.append(new_wp)

            # Update distances
            dists[:, k] = abs(vec - data[new_wp, ind])

        # Update global set
        waypoint_set = waypoint_set + iter_set

    # Unique waypoints
    waypoints = np.unique(waypoint_set)

    return waypoints

def fast_palantir(dist_matrix, start_cell, waypoints, waypoints_D, waypoints_W, sdv, max_iterations = 25):
    waypoints_idx = waypoints != start_cell
    start_D = dist_matrix[[start_cell]]
    D = np.concatenate([start_D, waypoints_D[waypoints_idx]])
    start_W, _ = calcu_w(start_D, sdv)
    W = np.concatenate([start_W, waypoints_W[waypoints_idx]])
    norm_c = W.sum(axis=0)
    zero_idx = norm_c == 0
    norm_c[zero_idx] = 1.
    W = W / norm_c
    W[:, zero_idx] = 1. / W.shape[0]
    # Initalize pseudotime to start cell distances
    pseudotime = D[0]
    
    converged = False

    # Iteratively update perspective and determine pseudotime
    iteration = 1

    while not converged and iteration < max_iterations:
        P = deepcopy(D)
        # Perspective matrix by alinging to start distances
        for i,wp in enumerate(waypoints[waypoints_idx]):

            # Position of waypoints relative to start
            idx_val = pseudotime[wp]

            # Convert all cells before starting point to the negative
            before_indices = np.where(pseudotime < idx_val)[0]
            P[i+1, before_indices] = -D[i+1, before_indices]

            # Align to start
            P[i+1, :] = P[i+1, :] + idx_val
        
        # Weighted pseudotime
        new_traj = (P*W).sum(axis=0)

        # Check for convergence
        corr = pearsonr(pseudotime, new_traj)[0]

        
        if corr > 0.9999:
            converged = True

        # If not converged, continue iteration
        pseudotime = new_traj
        iteration += 1

    pseudotime -= np.min(pseudotime)
    pseudotime /= np.max(pseudotime)
    return pseudotime




def diffmap_extrema(adata, diffmap_key='X_diffmap', ):
    extrema = []
    
    eigenvectors = adata.obsm[diffmap_key]
    for dcomp in range(eigenvectors.shape[1]):
        ec = eigenvectors[:, dcomp].argmax()
        extrema.append(ec)
        ec = eigenvectors[:, dcomp].argmin()
        extrema.append(ec)
    extrema_names = adata.obs.index[extrema]
    adata.uns['extrema'] = extrema_names

def init_candidiates(adata, diffmap_key, cell_type_key=None):
    diffmap_extrema(adata, diffmap_key=diffmap_key)
    if cell_type_key is None:
        return
    candidates = []
    for cell_type in pd.unique(adata.obs[cell_type_key]):
        candidates.append(adata.obs.loc[adata.obs[cell_type_key] == cell_type].sample(n=1).index)
    adata.uns['extrema'] = np.concatenate([adata.uns['extrema'], np.concatenate(candidates)])
    adata.uns['extrema'] = np.unique(adata.uns['extrema'])

def _connect_graph(adj, data, start_cell):
    # Create graph and compute distances
    
    graph = nx.Graph(adj)
    
    dists = pd.Series(nx.single_source_dijkstra_path_length(graph, start_cell))
    
    dists = pd.Series(dists.values, index=data.index[dists.index])

    # Idenfity unreachable nodes
    unreachable_nodes = data.index.difference(dists.index)
    if len(unreachable_nodes) > 0:
        warnings.warn(
            "Some of the cells were unreachable. Consider increasing the k for \n \
            nearest neighbor graph construction."
        )

    # Connect unreachable nodes
    while len(unreachable_nodes) > 0:
        farthest_reachable = np.where(data.index == dists.idxmax())[0][0]

        # Compute distances to unreachable nodes
        unreachable_dists = pairwise_distances(
            data.iloc[farthest_reachable, :].values.reshape(1, -1),
            data.loc[unreachable_nodes, :],
        )
        unreachable_dists = pd.Series(
            np.ravel(unreachable_dists), index=unreachable_nodes
        )

        # Add edge between farthest reacheable and its nearest unreachable
        add_edge = np.where(data.index == unreachable_dists.idxmin())[0][0]
        adj[farthest_reachable, add_edge] = unreachable_dists.min()

        # Recompute distances to early cell
        graph = nx.Graph(adj)
        dists = pd.Series(nx.single_source_dijkstra_path_length(graph, start_cell))
        dists = pd.Series(dists.values, index=data.index[dists.index])

        # Idenfity unreachable nodes
        unreachable_nodes = data.index.difference(dists.index)

    return adj



def compute_spdist(adata, embedding_key='X_pca', n_neighbors=20, start_cells=None, scale=True):
    
    X = pd.DataFrame(adata.obsm[embedding_key], index=adata.obs.index)
    if scale:
        X =pd.DataFrame(minmax_scale(X), index=adata.obs.index)
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X)
   
    knn_graph = neighbors.kneighbors_graph(X, mode="distance")
    if start_cells is not None:
        intersection_index = pd.Index(start_cells).intersection(adata.obs.index)
        print('Convert into connected graph')
        for start_cell in tqdm(intersection_index):
            knn_graph = _connect_graph(knn_graph, X, np.where(adata.obs.index == start_cell)[0][0])
    dist_matrix = dijkstra(csgraph=knn_graph, directed=False, )
    return knn_graph, dist_matrix

def calcu_ot_loss(adata, embedding_key,  pseudo_group_key='pseudobin', p=2):
    def _ot_loss(M, a=None, b=None):
        if a is None:
            a = torch.ones(M.shape[0]) / M.shape[0]
        if b is None:
            b = torch.ones(M.shape[1]) / M.shape[1]
        pi = pot.emd(a, b, M)
        #pi = pot.sinkhorn_unbalanced(a, b, M, 0., [1., 10.])
        
        return torch.sum(pi * M).item()
    loss = 0.
    for i in range(int(np.max(adata.obs[pseudo_group_key]))):
        M = torch.cdist(torch.tensor(adata[adata.obs.loc[adata.obs[pseudo_group_key] == i].index].obsm[embedding_key]),
                torch.tensor(adata[adata.obs.loc[adata.obs[pseudo_group_key] == i+1].index].obsm[embedding_key]), p=p)
        
        loss += _ot_loss(M,)
    return loss

def calcu_got_loss(adata, pseudo_group_key='pseudobin', dist_matrix = None):
    adata.obs['idx'] = range(len(adata))
    def _ot_loss(M, a=None, b=None):
        if a is None:
            a = torch.ones(M.shape[0]) / M.shape[0]
        if b is None:
            b = torch.ones(M.shape[1]) / M.shape[1]
        pi = pot.emd(a, b, M)
        #pi = pot.sinkhorn_unbalanced(a, b, M, 0., [1., 10.])
        
        return torch.sum(pi * M).item()
    loss = 0.
    for i in range(int(np.max(adata.obs[pseudo_group_key]))):
        idx1 = adata.obs.loc[adata.obs[pseudo_group_key] == i].idx.tolist()
        idx2 = adata.obs.loc[adata.obs[pseudo_group_key] == i+1].idx.tolist()
        M = torch.tensor(dist_matrix[idx1,:])
        M = M[:, idx2]
        
        loss += _ot_loss(M,)
    return loss

def greedy_search_best_source(adata, embedding_key, kernel='dpt', split_k=None, graph_dist=False, n_neighbors=20, connect_anchor=None, n_waypoints=1200):
    assert (kernel=='dpt') | ((kernel != 'dpt') & (graph_dist == True))
    time_key = kernel + '_pseudotime'
    if split_k is None:
        split_k = int(len(adata) / 100)
    res = {}
    if graph_dist:
        
        knn_graph, dist_matrix = compute_spdist(adata, embedding_key, n_neighbors=n_neighbors, start_cells=connect_anchor)
        if kernel == 'palantir':
            waypoints = _max_min_sampling(adata.obsm[embedding_key], num_waypoints=n_waypoints, seed=20)
            waypoints_D = dist_matrix[waypoints]
            waypoints_W, sdv = calcu_w(waypoints_D)

    adata.obs['root_loss'] = np.nan
    for i in tqdm(range(len(adata))):
        filtered_idx = range(len(adata))

        if kernel == 'dpt':
            adata.uns['iroot'] = i
            sc.tl.dpt(adata)

        elif kernel == 'sp':
            adata.obs[time_key] = dist_matrix[i,:]

        elif kernel == 'palantir':
            adata.obs[time_key] = fast_palantir(dist_matrix, i, waypoints, waypoints_D, waypoints_W, sdv)

        elif kernel == 'euclidean':
            adata.obs[time_key] = np.mean((adata.obsm[embedding_key] - adata.obsm[embedding_key][i])**2, axis=1)
            
        
        if np.sum(np.isinf(adata.obs[time_key])) > 0:
            
            filtered_idx = np.where(~np.isinf(adata.obs[time_key]))[0].tolist()
            
            data = adata[~np.isinf(adata.obs[time_key])].copy()
        else:
            data = adata
        data.obs[time_key] = scale(data.obs[time_key])
        if len(data) < 100:
            res[adata.obs.index[i]] = np.nan
            continue
        
        generate_time_points(data, k=split_k, pseudotime_key=time_key, sigma=0.)
        
        with torch.no_grad():
            if graph_dist == False:
                loss = calcu_ot_loss(data, embedding_key, pseudo_group_key='pseudobin',)
            else:
            
                cost_matrix = dist_matrix[filtered_idx,:]
                cost_matrix = cost_matrix[:, filtered_idx]
                loss = calcu_got_loss(data, pseudo_group_key='pseudobin', dist_matrix=cost_matrix )

        res[adata.obs.index[i]] = loss
    res = pd.DataFrame([list(res.keys()), list(res.values())], index=['cell', 'raw_root_loss']).T
    res['idx'] = range(len(res))
    res = res.sort_values('raw_root_loss')
    

    return res
    
def generate_time_points(adata, k=4, pseudotime_key = 'dpt_pseudotime', time_key='pseudobin', sigma=.0, ):
    adata.obs[time_key] = -1
    adata.obs[time_key+'_noise'] = adata.obs[pseudotime_key] + np.random.rand(len(adata)) * sigma
    sorted_idx = adata.obs.sort_values(time_key+'_noise').index
    
    bin_idxs = np.array_split(sorted_idx, k)
    for i in range(k):
        adata.obs.loc[bin_idxs[i], time_key] = i
    adata.obs[time_key] = adata.obs[time_key].astype(float)

def smoothe_score(X, y, n_neighbors=5):
    idx = ~np.isnan(y)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X[idx], y[idx])
    y_smoothed = knn.predict(X)
    return y_smoothed, knn
 
def determine_source_state(adata, embedding_key, graph_dist=True, n_neighbors=30, 
                            split_m=30, kernel='dpt', n_comps=15, down_sampling=True, n_obs=3000, 
                            cytotrace=True, alpha = 0.1, smooth_k=5,
                            connect_anchor=False) :
    """Determine souce cell for snapshot data

    In most developing biological scenario, source cells will develop into multiple different cells.
    
    By setting cell :math:`r` as start cell, the pseudotime :math:`\hat{t}(x_i)` can be computed, and 
    the empirical distribution can be divided into :math:`m` portions that :math:`X_1, X_2, ..., X_m`, according to time :math:`\hat{t}(x_i)`.  
    The transport cost of this time-vary distribution :math:`p_t(x|r)` can be quantified by optimal transport with graphical metrics.
    
    
    .. math::

        W_2^2(r)=\sum_{i=1}^{m-1}\inf_{\pi}\sum_{x \in X_i}\sum_{y \in X_{i+1}}c(x,y | G)\pi(x,y)

    where :math:`c(x,y|G)` is the shorest path distance between two cells :math:`x,y` in graph :math:`G`.
    According to the energy-saving hypothesis, the defined transport cost of real source cell will be smallest, that

    .. math::

        {r}^* = arg \min_{r} W_2^2(r)

    .. note::
        This assumption may *fails* in the case of *linear progression* that souce cell only developing in one direction.
        In that case, the transport cost of real source cell and terminate cell will be very close. So this function 
        will detect linear progression and compute cytotrace score with very low weight (default 0.1) to choose the optimal source cell.
        
        \\
            
        To accelerate the computation, we suggest to down sample the dataset to 3000 cells (default) and use the down sampled data to compute the transport cost.
        
    Arguments:
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    embedding_key: `str`
        Name of latent space, in adata.obsm
    graph_dist: `bool` (default: True)
        Using shorest path distance or euclidean distance
    n_neighbors: `int` (default: 30)
        Number of neighbors of kNN which is used to compute shortest path distance
    split_m: int (default: 30)
        Number of split. This number should NOT be too small
    kernel: 'dpt' or 'palantir' or 'euclidean' (default: 'dpt')
        Pseudotime method, 'dpt' is recommended
    n_comps: `int` (default: 15)
        Number of diffmap components, which is used for DPT computation
    down_sampling: `bool` (default: True)
        Down sampling dataset to accelerate computation
    n_obs: `int` (default: 3000)
        Number of down sampling size
    cytotrace: `bool` (default: True)
        Use cytotrace to help. Note cytorace is implemented by Cellrank2
    alpha: `float` (default: 0.1)
        Weight of cytotrace. We do NOT suggest increase the weight
    smooth_k: `int` (default: 5)
        Number of neighbors which is used to smoothes the final score
    time_key: `str` (default: None)
        Name of time label, in adata.obs, use if the model input contains time label
    
    Returns
    -------
    ot_root (.uns): `int`
        best source cell index using transport cost only
    ot_ct_root (.uns): `int`
        best source cell index using both transport cost and cytotrace
    root_score (.obs): `np.ndarray`
        source cell score (higher score higher probability to be source)
    ot_root_score (.obs): `np.ndarray`
        source cell score + alpha * cytotrace score (higher score higher probability to be source)
    """
    if kernel == 'palantir':
        if not ('DM_EigenVectors' in adata.obsm.keys()):
            palantir.run_diffusion_maps(adata, n_components=n_comps)
    else:
        if not ('X_diffmap' in adata.obsm.keys()):
            sc.tl.diffmap(adata, n_comps=n_comps)

    if down_sampling and len(adata) > n_obs:
        print('Down sampling')
        sub_adata = adata.copy()
        sc.pp.subsample(sub_adata, n_obs=n_obs)
        sc.pp.neighbors(sub_adata, use_rep=embedding_key)    
    else:
        sub_adata = adata
    
    
    if connect_anchor:
        if kernel == 'palantir':
            init_candidiates(sub_adata, diffmap_key='DM_EigenVectors')
        else:
            init_candidiates(sub_adata, diffmap_key='X_diffmap')
        connect_anchor = sub_adata.uns['extrema']
    else:
        connect_anchor = None
    res = greedy_search_best_source(sub_adata, embedding_key, split_k=split_m, kernel=kernel,
                             graph_dist=graph_dist, n_neighbors=n_neighbors, connect_anchor=connect_anchor,
                             )
    
    res['root_loss'], knn = smoothe_score(sub_adata[res['cell']].obsm[embedding_key], 
                                             res['raw_root_loss'].to_numpy().astype(float),
                                             smooth_k)
    res['root_score'] = 1 - scale(res['root_loss'])
    res = res.sort_values('root_score', ascending=False)
    print("optimal transport root cell write in adata.uns['ot_root']")
    adata.uns['ot_root'] = np.where(adata.obs.index == res['cell'].tolist()[0])[0][0]
    adata.obs['root_score'] = 1 - scale(knn.predict(adata.obsm[embedding_key]))
    
    #optianal
    if cytotrace:
        if (not 'spliced' in adata.layers.keys()) or (not 'unspliced' in adata.layers.keys()):
            adata.layers['Ms'] = adata.X
            adata.layers['Mu'] = adata.X

        CytoTRACEKernel(adata).compute_cytotrace()
        adata.obs['ct_root_score'] = (adata.obs['root_score'].to_numpy() + alpha * (1 - adata.obs['ct_pseudotime'].to_numpy())) / (1+alpha)
        res['ct_root_score'] = adata[res['cell']].obs['ct_root_score'].tolist()
        res = res.sort_values('ct_root_score', ascending=False)
        print("optimal transport + cytotrace root cell write in adata.uns['ot_ct_root']")
        adata.uns['ot_ct_root'] = np.where(adata.obs.index == res['cell'].tolist()[0])[0][0]
        adata.uns['cytotrace_alpha'] = alpha
    return res


