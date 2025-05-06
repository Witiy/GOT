import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

def scale(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def cosine(a, b):
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))

def inner_kernel(a, b):
    return np.exp(np.sum(a * b, axis=1))
    

def find_neighbors(sparse_matrix):
    neighbors = {}
    for i in range(sparse_matrix.shape[0]):
        neighbors[i] = sparse_matrix[i].indices.tolist()
    return neighbors




    
def split_negative_P(P):
    graph = coo_matrix(P).copy()
    graph_neg = graph.copy()

    graph.data = np.clip(graph.data, 0, 1)
    graph_neg.data = np.clip(graph_neg.data, -1, 0)

    graph.eliminate_zeros()
    graph_neg.eliminate_zeros()

    return graph.tocsr(), graph_neg.tocsr()
