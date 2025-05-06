"""
Core functions for running Palantir
"""
from typing import Union, Optional, List, Dict
import numpy as np
import pandas as pd
import networkx as nx
import copy

from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from scipy.sparse.linalg import eigs

from scipy.sparse import csr_matrix, find, csgraph
from scipy.stats import pearsonr, norm
from copy import deepcopy

import scanpy as sc

import warnings




def run_palantir(
    data: Union[pd.DataFrame, sc.AnnData],
    early_cell,
    terminal_states: Optional[Union[List, Dict, pd.Series]] = None,
    knn: int = 30,
    num_waypoints: int = 1200,
    n_jobs: int = -1,
    scale_components: bool = True,
    use_early_cell_as_start: bool = False,
    max_iterations: int = 25,
    eigvec_key: str = "DM_EigenVectors_multiscaled",
    pseudo_time_key: str = "palantir_pseudotime",
    seed: int = 20,
) :
    """
    Executes the Palantir algorithm to derive pseudotemporal ordering of cells, their fate probabilities, and
    state entropy based on the multiscale diffusion map results.

    Parameters
    ----------
    data : Union[pd.DataFrame, sc.AnnData]
        Either a DataFrame of multiscale space diffusion components or a Scanpy AnnData object.
    early_cell : str
        Start cell for pseudotime construction.
    terminal_states : List/Series/Dict, optional
        User-defined terminal states structure in the format {terminal_name:cell_name}. Default is None.
    knn : int, optional
        Number of nearest neighbors for graph construction. Default is 30.
    num_waypoints : int, optional
        Number of waypoints to sample. Default is 1200.
    n_jobs : int, optional
        Number of jobs for parallel processing. Default is -1.
    scale_components : bool, optional
        If True, components are scaled. Default is True.
    use_early_cell_as_start : bool, optional
        If True, the early cell is used as start. Default is False.
    max_iterations : int, optional
        Maximum number of iterations for pseudotime convergence. Default is 25.
    eigvec_key : str, optional
        Key to access multiscale space diffusion components from obsm of the AnnData object. Default is 'DM_EigenVectors_multiscaled'.
    pseudo_time_key : str, optional
        Key to store the pseudotime in obs of the AnnData object. Default is 'palantir_pseudotime'.
    entropy_key : str, optional
        Key to store the entropy in obs of the AnnData object. Default is 'palantir_entropy'.
    fate_prob_key : str, optional
        Key to store the fate probabilities in obsm of the AnnData object. Default is 'palantir_fate_probabilities'.
        If save_as_df is True, the fate probabilities are stored as pandas DataFrame with terminal state names as columns.
        If False, the fate probabilities are stored as numpy array and the terminal state names are stored in uns[fate_prob_key + "_columns"].
    save_as_df : bool, optional
        If True, the fate probabilities are saved as pandas DataFrame. If False, the data is saved as numpy array.
        The option to save as DataFrame is there due to some versions of AnnData not being able to
        write h5ad files with DataFrames in ad.obsm. Default is palantir.SAVE_AS_DF = True.
    waypoints_key : str, optional
        Key to store the waypoints in uns of the AnnData object. Default is 'palantir_waypoints'.
    seed : int, optional
        The seed for the random number generator used in waypoint sampling. Default is 20.

    
    
    """


    if isinstance(terminal_states, dict):
        terminal_states = pd.Series(terminal_states)
    if isinstance(terminal_states, pd.Series):
        terminal_cells = terminal_states.index.values
    else:
        terminal_cells = terminal_states
    if isinstance(data, sc.AnnData):
        ms_data = pd.DataFrame(data.obsm[eigvec_key], index=data.obs_names)
    else:
        ms_data = data

    if scale_components:
        data_df = pd.DataFrame(
            preprocessing.minmax_scale(ms_data),
            index=ms_data.index,
            columns=ms_data.columns,
        )
    else:
        data_df = copy.copy(ms_data)

    # ################################################
    # Determine the boundary cell closest to user defined early cell
    dm_boundaries = pd.Index(set(data_df.idxmax()).union(data_df.idxmin()))
    dists = pairwise_distances(
        data_df.loc[dm_boundaries, :], data_df.loc[early_cell, :].values.reshape(1, -1)
    )
    start_cell = pd.Series(np.ravel(dists), index=dm_boundaries).idxmin()
    if use_early_cell_as_start:
        start_cell = early_cell

    # Sample waypoints
    

    # Append start cell
    if isinstance(num_waypoints, int):
        waypoints = _max_min_sampling(data_df, num_waypoints, seed)
    else:
        waypoints = num_waypoints
    waypoints = waypoints.union(dm_boundaries)
    if terminal_cells is not None:
        waypoints = waypoints.union(terminal_cells)
    waypoints = pd.Index(waypoints.difference([start_cell]).unique())

    # Append start cell
    waypoints = pd.Index([start_cell]).append(waypoints)

    # pseudotime and weighting matrix
    
    pseudotime, W = _compute_pseudotime(
        data_df, start_cell, knn, waypoints, n_jobs, max_iterations
    )



    if isinstance(data, sc.AnnData):
        data.obs[pseudo_time_key] = pseudotime
        



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
    for ind in data.columns:
        # Data vector
        vec = np.ravel(data[ind])

        # Random initialzlation
        iter_set = [
            np.random.randint(N),
        ]

        # Distances along the component
        dists = np.zeros([N, no_iterations])
        dists[:, 0] = abs(vec - data[ind].values[iter_set])
        for k in range(1, no_iterations):
            # Minimum distances across the current set
            min_dists = dists[:, 0:k].min(axis=1)

            # Point with the maximum of the minimum distances is the new waypoint
            new_wp = np.where(min_dists == min_dists.max())[0][0]
            iter_set.append(new_wp)

            # Update distances
            dists[:, k] = abs(vec - data[ind].values[new_wp])

        # Update global set
        waypoint_set = waypoint_set + iter_set

    # Unique waypoints
    waypoints = data.index[waypoint_set].unique()

    return waypoints


def _compute_pseudotime(data, start_cell, knn, waypoints, n_jobs, max_iterations=25):
    """Function for compute the pseudotime

    :param data: Multiscale space diffusion components
    :param start_cell: Start cell for pseudotime construction
    :param knn: Number of nearest neighbors for graph construction
    :param waypoints: List of waypoints
    :param n_jobs: Number of jobs for parallel processing
    :param max_iterations: Maximum number of iterations for pseudotime convergence
    :return: pseudotime and weight matrix
    """

    # ################################################
    # Shortest path distances to determine trajectories
    

    nbrs = NearestNeighbors(n_neighbors=knn, metric="euclidean", n_jobs=n_jobs if n_jobs != 1 else None).fit(
        data
    )
    adj = nbrs.kneighbors_graph(data, mode="distance")

    # Connect graph if it is disconnected
    adj = _connect_graph(adj, data, np.where(data.index == start_cell)[0][0])
    # Distances
    if n_jobs != 1:
        dists = Parallel(n_jobs=n_jobs, max_nbytes=None)(
            delayed(_shortest_path_helper)(np.where(data.index == cell)[0][0], adj)
            for cell in waypoints
        )
    else:
        dists = [ _shortest_path_helper(np.where(data.index == cell)[0][0], adj) for cell in waypoints]
        
    # Convert to distance matrix
    D = pd.DataFrame(0.0, index=waypoints, columns=data.index)
    for i, cell in enumerate(waypoints):
        D.loc[cell, :] = pd.Series(
            np.ravel(dists[i]), index=data.index[dists[i].index]
        )[data.index]
   

    # ###############################################
    # Determine the perspective matrix

    
    # Waypoint weights
    sdv = np.std(np.ravel(D)) * 1.06 * len(np.ravel(D)) ** (-1 / 5)
    W = np.exp(-0.5 * np.power((D / sdv), 2))
    # Stochastize the matrix
    W = W / W.sum()

    # Initalize pseudotime to start cell distances
    pseudotime = D.loc[start_cell, :]
    converged = False

    # Iteratively update perspective and determine pseudotime
    iteration = 1
    while not converged and iteration < max_iterations:
        # Perspective matrix by alinging to start distances
        
        P = deepcopy(D)
        
        for wp in waypoints[1:]:
            # Position of waypoints relative to start
            idx_val = pseudotime[wp]

            # Convert all cells before starting point to the negative
            before_indices = pseudotime.index[pseudotime < idx_val]
            P.loc[wp, before_indices] = -D.loc[wp, before_indices]

            # Align to start
            P.loc[wp, :] = P.loc[wp, :] + idx_val

        # Weighted pseudotime
        new_traj = P.multiply(W).sum()

        # Check for convergence
        corr = pearsonr(pseudotime, new_traj)[0]
        
        if corr > 0.9999:
            converged = True

        # If not converged, continue iteration
        pseudotime = new_traj
        iteration += 1

    pseudotime -= np.min(pseudotime)
    pseudotime /= np.max(pseudotime)

    return pseudotime, W


def identify_terminal_states(
    ms_data,
    early_cell,
    knn=30,
    num_waypoints=1200,
    n_jobs=-1,
    max_iterations=25,
    seed=20,
):
    # Scale components
    data = pd.DataFrame(
        preprocessing.minmax_scale(ms_data),
        index=ms_data.index,
        columns=ms_data.columns,
    )

    #  Start cell as the nearest diffusion map boundary
    dm_boundaries = pd.Index(set(data.idxmax()).union(data.idxmin()))
    dists = pairwise_distances(
        data.loc[dm_boundaries, :], data.loc[early_cell, :].values.reshape(1, -1)
    )
    start_cell = pd.Series(np.ravel(dists), index=dm_boundaries).idxmin()

    # Sample waypoints
    # Append start cell
    waypoints = _max_min_sampling(data, num_waypoints, seed)
    waypoints = waypoints.union(dm_boundaries)
    waypoints = pd.Index(waypoints.difference([start_cell]).unique())

    # Append start cell
    waypoints = pd.Index([start_cell]).append(waypoints)

    # Distance to start cell as pseudo pseudotime
    pseudotime, _ = _compute_pseudotime(
        data, start_cell, knn, waypoints, n_jobs, max_iterations
    )

    # Markov chain
    wp_data = data.loc[waypoints, :]
    T = _construct_markov_chain(wp_data, knn, pseudotime, n_jobs)

    # Terminal states
    terminal_states = _terminal_states_from_markov_chain(T, wp_data, pseudotime)

    # Excluded diffusion map boundaries
    dm_boundaries = pd.Index(set(wp_data.idxmax()).union(wp_data.idxmin()))
    excluded_boundaries = dm_boundaries.difference(terminal_states).difference(
        [start_cell]
    )
    return terminal_states, excluded_boundaries


def _construct_markov_chain(wp_data, knn, pseudotime, n_jobs):
    # Markov chain construction
    
    waypoints = wp_data.index

    # kNN graph
    n_neighbors = knn
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, metric="euclidean", n_jobs=n_jobs
    ).fit(wp_data)
    kNN = nbrs.kneighbors_graph(wp_data, mode="distance")
    dist, ind = nbrs.kneighbors(wp_data)

    # Standard deviation allowing for "back" edges
    adpative_k = np.min([int(np.floor(n_neighbors / 3)) - 1, 30])
    adaptive_std = np.ravel(dist[:, adpative_k])

    # Directed graph construction
    # pseudotime position of all the neighbors
    traj_nbrs = pd.DataFrame(
        pseudotime[np.ravel(waypoints.values[ind])].values.reshape(
            [len(waypoints), n_neighbors]
        ),
        index=waypoints,
    )

    # Remove edges that move backwards in pseudotime except for edges that are within
    # the computed standard deviation
    rem_edges = traj_nbrs.apply(
        lambda x: x < pseudotime[traj_nbrs.index] - adaptive_std
    )
    rem_edges = rem_edges.stack()[rem_edges.stack()]

    # Determine the indices and update adjacency matrix
    cell_mapping = pd.Series(range(len(waypoints)), index=waypoints)
    x = list(cell_mapping[rem_edges.index.get_level_values(0)])
    y = list(rem_edges.index.get_level_values(1))
    # Update adjacecy matrix
    kNN[x, ind[x, y]] = 0

    # Affinity matrix and markov chain
    x, y, z = find(kNN)
    aff = np.exp(
        -(z**2) / (adaptive_std[x] ** 2) * 0.5
        - (z**2) / (adaptive_std[y] ** 2) * 0.5
    )
    W = csr_matrix((aff, (x, y)), [len(waypoints), len(waypoints)])

    # Transition matrix
    D = np.ravel(W.sum(axis=1))
    x, y, z = find(W)
    T = csr_matrix((z / D[x], (x, y)), [len(waypoints), len(waypoints)])

    return T


def _terminal_states_from_markov_chain(T, wp_data, pseudotime):
    

    # Identify terminal statses
    waypoints = wp_data.index
    dm_boundaries = pd.Index(set(wp_data.idxmax()).union(wp_data.idxmin()))
    n = min(*T.shape)
    vals, vecs = eigs(T.T, 10, maxiter=n * 50)

    ranks = np.abs(np.real(vecs[:, np.argsort(vals)[-1]]))
    ranks = pd.Series(ranks, index=waypoints)

    # Cutoff and intersection with the boundary cells
    cutoff = norm.ppf(
        0.9999,
        loc=np.median(ranks),
        scale=np.median(np.abs((ranks - np.median(ranks)))),
    )

    # Connected components of cells beyond cutoff
    cells = ranks.index[ranks > cutoff]

    # Find connected components
    T_dense = pd.DataFrame(T.todense(), index=waypoints, columns=waypoints)
    graph = nx.from_pandas_adjacency(T_dense.loc[cells, cells])
    cells = [pseudotime[list(i)].idxmax() for i in nx.connected_components(graph)]

    # Nearest diffusion map boundaries
    terminal_states = [
        pd.Series(
            np.ravel(
                pairwise_distances(
                    wp_data.loc[dm_boundaries, :],
                    wp_data.loc[i, :].values.reshape(1, -1),
                )
            ),
            index=dm_boundaries,
        ).idxmin()
        for i in cells
    ]

    terminal_states = np.unique(terminal_states)

    # excluded_boundaries = dm_boundaries.difference(terminal_states)
    return terminal_states



def _shortest_path_helper(cell, adj):
    return pd.Series(csgraph.dijkstra(adj, False, cell))


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


def compute_kernel(
    data: Union[pd.DataFrame, sc.AnnData],
    knn: int = 30,
    alpha: float = 0,
    pca_key: str = "X_pca",
    
) -> csr_matrix:
    """
    Compute the adaptive anisotropic diffusion kernel.

    Parameters
    ----------
    data : Union[pd.DataFrame, sc.AnnData]
        Data points (rows) in a feature space (columns) for pd.DataFrame.
        For sc.AnnData, it uses the .X attribute.
    knn : int
        Number of nearest neighbors for adaptive kernel calculation. Default is 30.
    alpha : float
        Normalization parameter for the diffusion operator. Default is 0.
    pca_key : str, optional
        Key to retrieve PCA projections from data if it is a sc.AnnData object. Default is 'X_pca'.
    kernel_key : str, optional
        Key to store the kernel in obsp of data if it is a sc.AnnData object. Default is 'DM_Kernel'.

    Returns
    -------
    csr_matrix
        Computed kernel matrix.
    """

    # If the input is sc.AnnData, convert it to a DataFrame
    if isinstance(data, sc.AnnData):
        data_df = pd.DataFrame(data.obsm[pca_key], index=data.obs_names)
    else:
        data_df = data

    N = data_df.shape[0]
    temp = sc.AnnData(data_df.values)
    sc.pp.neighbors(temp, n_pcs=0, n_neighbors=knn)
    kNN = temp.obsp["distances"]

    adaptive_k = int(np.floor(knn / 3))
    adaptive_std = np.zeros(N)
    for i in np.arange(N):
        adaptive_std[i] = np.sort(kNN.data[kNN.indptr[i] : kNN.indptr[i + 1]])[
            adaptive_k - 1
            
        ]

    x, y, dists = find(kNN)
    dists /= adaptive_std[x]
    W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])

    kernel = W + W.T

    if alpha > 0:
        D = np.ravel(kernel.sum(axis=1))
        D[D != 0] = D[D != 0] ** (-alpha)
        mat = csr_matrix((D, (range(N), range(N))), shape=[N, N])
        kernel = mat.dot(kernel).dot(mat)

    return kernel

def diffusion_maps_from_kernel(
    kernel: csr_matrix, n_components: int = 10, seed: Union[int, None] = 0
):
    """
    Compute the diffusion map given a kernel matrix.

    Parameters
    ----------
    kernel : csr_matrix
        Precomputed kernel matrix.
    n_components : int
        Number of diffusion components to compute. Default is 10.
    seed : Union[int, None]
        Seed for random initialization. Default is 0.

    Returns
    -------
    dict
        T-matrix (T), Diffusion components (EigenVectors) and corresponding eigenvalues (EigenValues).
    """
    N = kernel.shape[0]
    D = np.ravel(kernel.sum(axis=1))
    D[D != 0] = 1 / D[D != 0]
    T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(kernel)

    np.random.seed(seed)
    v0 = np.random.rand(min(T.shape))
    D, V = eigs(T, n_components, tol=1e-4, maxiter=1000, v0=v0)

    D = np.real(D)
    V = np.real(V)
    inds = np.argsort(D)[::-1]
    D = D[inds]
    V = V[:, inds]

    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

    return {"T": T, "EigenVectors": pd.DataFrame(V), "EigenValues": pd.Series(D)}

def run_diffusion_maps(
    data: Union[pd.DataFrame, sc.AnnData],
    n_components: int = 10,
    knn: int = 30,
    alpha: float = 0,
    seed: Union[int, None] = 0,
    pca_key: str = "X_pca",
    eigval_key: str = "DM_EigenValues",
    eigvec_key: str = "DM_EigenVectors",
):
    """
    Run Diffusion maps using the adaptive anisotropic kernel.

    Parameters
    ----------
    data : Union[pd.DataFrame, sc.AnnData]
        PCA projections of the data or adjacency matrix.
        If sc.AnnData is passed, its obsm[pca_key] is used and the result is written to
        its obsp[kernel_key], obsm[eigvec_key], and uns[eigval_key].
    n_components : int, optional
        Number of diffusion components. Default is 10.
    knn : int, optional
        Number of nearest neighbors for graph construction. Default is 30.
    alpha : float, optional
        Normalization parameter for the diffusion operator. Default is 0.
    seed : Union[int, None], optional
        Numpy random seed, randomized if None, set to an arbitrary integer for reproducibility.
        Default is 0.
    pca_key : str, optional
        Key to retrieve PCA projections from data if it is a sc.AnnData object. Default is 'X_pca'.
    eigval_key : str, optional
        Key to store the EigenValues in uns of data if it is a sc.AnnData object. Default is 'DM_EigenValues'.
    eigvec_key : str, optional
        Key to store the EigenVectors in obsm of data if it is a sc.AnnData object. Default is 'DM_EigenVectors'.

    Returns
    -------
    dict
        Diffusion components, corresponding eigen values and the diffusion operator.
        If sc.AnnData is passed as data, these results are also written to the input object
        and returned.
    """

    if isinstance(data, sc.AnnData):
        data_df = pd.DataFrame(data.obsm[pca_key], index=data.obs_names)
    else:
        data_df = data

    
    kernel = compute_kernel(data_df, knn, alpha)
    
    res = diffusion_maps_from_kernel(kernel, n_components, seed)

    res["kernel"] = kernel
    
    res["EigenVectors"].index = data_df.index

    if isinstance(data, sc.AnnData):
        data.obsm[eigvec_key] = res["EigenVectors"].values
        data.uns[eigval_key] = res["EigenValues"].values

    return res

def determine_multiscale_space(
    dm_res: Union[dict, sc.AnnData],
    n_eigs: Union[int, None] = None,
    eigval_key: str = "DM_EigenValues",
    eigvec_key: str = "DM_EigenVectors",
    out_key: str = "DM_EigenVectors_multiscaled",
) -> Union[pd.DataFrame, None]:
    """
    Determine the multi-scale space of the data.

    Parameters
    ----------
    dm_res : Union[dict, sc.AnnData]
        Diffusion map results from run_diffusion_maps.
        If sc.AnnData is passed, its uns[eigval_key] and obsm[eigvec_key] are used.
    n_eigs : Union[int, None], optional
        Number of eigen vectors to use. If None is specified, the number
        of eigen vectors will be determined using the eigen gap. Default is None.
    eigval_key : str, optional
        Key to retrieve EigenValues from dm_res if it is a sc.AnnData object. Default is 'DM_EigenValues'.
    eigvec_key : str, optional
        Key to retrieve EigenVectors from dm_res if it is a sc.AnnData object. Default is 'DM_EigenVectors'.
    out_key : str, optional
        Key to store the result in obsm of dm_res if it is a sc.AnnData object. Default is 'DM_EigenVectors_multiscaled'.

    Returns
    -------
    Union[pd.DataFrame, None]
        Multi-scale data matrix. If sc.AnnData is passed as dm_res, the result
        is written to its obsm[out_key] and None is returned.
    """
    if isinstance(dm_res, sc.AnnData):
        eigenvectors = dm_res.obsm[eigvec_key]
        if not isinstance(eigenvectors, pd.DataFrame):
            eigenvectors = pd.DataFrame(eigenvectors, index=dm_res.obs_names)
        dm_res_dict = {
            "EigenValues": dm_res.uns[eigval_key],
            "EigenVectors": eigenvectors,
        }
    else:
        dm_res_dict = dm_res

    if not isinstance(dm_res_dict, dict):
        raise ValueError("'dm_res' should be a dict or a sc.AnnData instance")
    if n_eigs is None:
        vals = np.ravel(dm_res_dict["EigenValues"])
        n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-1] + 1
        if n_eigs < 3:
            n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-2] + 1

    # Scale the data
    use_eigs = list(range(1, n_eigs))
    eig_vals = np.ravel(dm_res_dict["EigenValues"][use_eigs])
    data = dm_res_dict["EigenVectors"].values[:, use_eigs] * (eig_vals / (1 - eig_vals))
    data = pd.DataFrame(data, index=dm_res_dict["EigenVectors"].index)

    if isinstance(dm_res, sc.AnnData):
        dm_res.obsm[out_key] = data.values

    return data