
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import coo_matrix

class NeighborsSampler:
    def __init__(self, n_neighbors, X0):
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X0)
        self.n_neighbors = n_neighbors
    def sample(self, x1, min_radius=20):
        _, indices = self.nbrs.kneighbors(x1)
        x0_idx = indices[ range(len(indices)), [np.random.choice(range(min_radius, self.n_neighbors)) for i in range(len(indices))]]
        return x0_idx
    
def cosine(a, b):
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))
    

def find_neighbors(sparse_matrix, directed=False):
    if directed:
        neighbors = np.array([sparse_matrix[i].indices.tolist() for i in range(sparse_matrix.shape[0])])
    else:
        neighbors = {i:sparse_matrix[i].indices.tolist() for i in range(sparse_matrix.shape[0])}
    return neighbors

    
def split_negative_P(P):
    graph = coo_matrix(P).copy()
    graph_neg = graph.copy()

    graph.data = np.clip(graph.data, 0, 1)
    graph_neg.data = np.clip(graph_neg.data, -1, 0)

    graph.eliminate_zeros()
    graph_neg.eliminate_zeros()

    return graph.tocsr(), graph_neg.tocsr()