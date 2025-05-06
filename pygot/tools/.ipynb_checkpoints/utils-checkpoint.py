
from sklearn.neighbors import NearestNeighbors
import numpy as np
class NeighborsSampler:
    def __init__(self, n_neighbors, X0):
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X0)
        self.n_neighbors = n_neighbors
    def sample(self, x1, min_radius=20):
        _, indices = self.nbrs.kneighbors(x1)
        x0_idx = indices[ range(len(indices)), [np.random.choice(range(min_radius, self.n_neighbors)) for i in range(len(indices))]]
        return x0_idx
    