import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
import pandas as pd
from collections import deque, defaultdict


def construct_mst(data, cluster_labels, times, start_node, time_constrain=True, end_nodes=[]):
    """
    Parameters:
        data: np.ndarray. The dimensionality-reduced data of shape (num_cells, 2)
        cluster_labels: np.ndarray. Cluster assignments of shape (num_cells).
        start_node: int. The starting node of the minimum spanning tree.
        end_nodes: list. Any terminal nodes.
    Returns:
        children: dict. A dictionary mapping clusters to the children of each cluster.
    """
    cluster_times = calculate_cluster_centres(times, cluster_labels)
    cluster_times[start_node] = 0
    cluster_label_indices, cluster_centres, num_clusters = preprocess_clusters(data, cluster_labels)
    emp_covs = calculate_empirical_covariances(data, cluster_label_indices, num_clusters)
    dists = calculate_mahalanobis_distances(cluster_centres, emp_covs, num_clusters)
    #dists = sp_dist_cluster
    #print(dists)
    mst_dists, index_mapping = prepare_mst_distances(dists, end_nodes, num_clusters)
    tree = minimum_spanning_tree(mst_dists)
    connections = build_connections(tree, index_mapping, dists, end_nodes)
    children = build_tree_bfs(connections, start_node, cluster_times, time_constrain)
    #children = build_tree_bfs(connections, start_node)
    return children, tree, mst_dists

def preprocess_clusters(data, cluster_labels):
    if isint(cluster_labels[0]):
        cluster_max = cluster_labels.max()
        cluster_label_indices = cluster_labels
    elif isstr(cluster_labels[0]):
        cluster_max = len(np.unique(cluster_labels))
        cluster_label_indices = np.array([np.where(np.unique(cluster_labels) == label)[0][0] for label in cluster_labels])
    else:
        raise ValueError("Unexpected cluster label dtype.")
    cluster_centres = [data[cluster_label_indices == k].mean(axis=0) for k in range(cluster_max + 1)]
    cluster_centres = np.stack(cluster_centres)
    return cluster_label_indices, cluster_centres, cluster_max + 1

def calculate_empirical_covariances(data, cluster_label_indices, num_clusters):
    return np.stack([np.cov(data[cluster_label_indices == i].T) for i in range(num_clusters)])

def calculate_mahalanobis_distances(cluster_centres, emp_covs, num_clusters):
    dists = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(i, num_clusters):
            dist = mahalanobis(
                cluster_centres[i],
                cluster_centres[j],
                emp_covs[i],
                emp_covs[j]
            )
            dists[i, j] = dist
            dists[j, i] = dist
    return dists

def prepare_mst_distances(dists, end_nodes, num_clusters):
    mst_dists = np.delete(np.delete(dists, end_nodes, axis=0), end_nodes, axis=1)
    index_mapping = np.array([c for c in range(num_clusters - len(end_nodes))])
    for i, end_node in enumerate(end_nodes):
        index_mapping[end_node - i:] += 1
    return mst_dists, index_mapping

def build_connections(tree, index_mapping, dists, end_nodes):
    connections = {k: list() for k in range(len(index_mapping) + len(end_nodes))}
    cx = tree.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        i = index_mapping[i]
        j = index_mapping[j]
        connections[i].append(j)
        connections[j].append(i)
    for end in end_nodes:
        i = np.argmin(np.delete(dists[end], end_nodes))
        connections[i].append(end)
        connections[end].append(i)
    return connections


def mahalanobis(x, y, cov1, cov2):
    """
    Custom Mahalanobis distance calculation function.
    This is a placeholder; you should replace it with the actual implementation.
    """
    return np.sqrt((x - y).T @ np.linalg.inv(cov1 + cov2) @ (x - y))

def isint(value):
    return isinstance(value, (int, np.integer))

def isstr(value):
    return isinstance(value, (str, np.str_))

def build_tree_bfs(connections, start_node, cluster_times, time_constrain=True):
    
    visited = [False for _ in range(len(connections))]
    queue = deque([start_node])
    children = {k: list() for k in range(len(connections))}
    ancestors = {k: None for k in range(len(connections))}  # 记录每个节点的祖先节点
    
    while queue:
        current_node = queue.popleft()
        visited[current_node] = True

        for child in connections[current_node]:
            if not visited[child]:
                if cluster_times[current_node] < cluster_times[child] + 0.01 or (not time_constrain):
                    children[current_node].append(child)
                    ancestors[child] = current_node
                    queue.append(child)
                else:
                    # 查找祖先节点是否符合条件
                    ancestor = ancestors[current_node]
                    while ancestor is not None and cluster_times[ancestor] >= cluster_times[child]:
                        ancestor = ancestors[ancestor]
                    if ancestor is not None:
                        children[ancestor].append(child)
                        ancestors[child] = ancestor
                        queue.append(child)
    return children

def find_sources_and_sinks(mst_children):
    in_degrees = defaultdict(int)
    out_degrees = defaultdict(int)

    for parent, children in mst_children.items():
        out_degrees[parent] += len(children)
        for child in children:
            in_degrees[child] += 1

    sources = [node for node in mst_children if in_degrees[node] == 0 and out_degrees[node] > 0]
    sinks = [node for node in mst_children if in_degrees[node] > 0 and out_degrees[node] == 0]
    
    return sources, sinks

def search_lineages(mst_children, sources, sinks):
    def dfs(current_node, path):
        if current_node in sinks:
            all_paths.append(path[:])
            return

        for child in mst_children[current_node]:
            path.append(child)
            dfs(child, path)
            path.pop()

    all_paths = []

    for source in sources:
        dfs(source, [source])
    
    return all_paths



def topological_tree(adata, embedding_key, cell_type_key, time_key, start_cell_type, time_constrain=True):
    
    adata.obs['int_cluster'] = pd.factorize(adata.obs[cell_type_key])[0]
    #convert to int
    start_cell_type = adata.obs.loc[adata.obs[cell_type_key] == start_cell_type].int_cluster.tolist()[0]
    mst_children, tree, mst_dists = construct_mst(adata.obsm[embedding_key], adata.obs['int_cluster'].to_numpy(), 
                                       adata.obs[time_key].to_numpy(), start_cell_type, time_constrain=time_constrain)
    return mst_children, tree, mst_dists

       
def calculate_cluster_centres(data, cluster_labels):

    unique_labels = np.unique(cluster_labels)
    
    
    cluster_centres = {}
    
    for label in unique_labels:
        cluster_points = data[cluster_labels == label]
        
        cluster_centre = cluster_points.mean(axis=0)
        cluster_centres[int(label)] = cluster_centre
    
    return cluster_centres

