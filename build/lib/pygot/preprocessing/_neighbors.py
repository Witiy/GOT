import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def mutual_nearest_neighbors(
    adata, 
    use_rep, 
    n_neighbors,
    mutual=False,
    sym=False,
    **kwargs
):
    
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors, **kwargs)
    graph = adata.obsp['distances']
    if mutual:
        graph = filter_mutual_neighbors(graph)
    if sym:
        graph = sym_neighbors(graph)
    return graph

def sym_neighbors(graph: csr_matrix) -> csr_matrix:
    """
    Filters a sparse adjacency matrix, keeping only mutual neighbors (Pij != 0 and Pji != 0).
    
    Parameters:
        graph (csr_matrix): Input sparse adjacency matrix.
    
    Returns:
        csr_matrix: Filtered sparse adjacency matrix.
    """
    # Ensure the input is a csr_matrix for efficient row-wise operations
    graph = graph.tocsr()
    
    # Compute the element-wise minimum of the matrix and its transpose
    mutual_neighbors = graph + graph.transpose()
    return mutual_neighbors


def filter_mutual_neighbors(graph: csr_matrix) -> csr_matrix:
    """
    Filters a sparse adjacency matrix, keeping only mutual neighbors (Pij != 0 and Pji != 0).
    
    Parameters:
        graph (csr_matrix): Input sparse adjacency matrix.
    
    Returns:
        csr_matrix: Filtered sparse adjacency matrix.
    """
    # Ensure the input is a csr_matrix for efficient row-wise operations
    graph = graph.tocsr()
    
    # Compute the element-wise minimum of the matrix and its transpose
    mutual_neighbors = graph.minimum(graph.transpose())
    isolated_node_num = sum(np.array(mutual_neighbors.sum(axis=1)).flatten() == 0)
    print('Isolated node: {:.4f}%'.format(100 * isolated_node_num / graph.shape[0]))
    return mutual_neighbors
