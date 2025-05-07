import numpy as np
import matplotlib.pyplot as plt
from pygot.tools.traj.mst import calculate_cluster_centres
import scanpy as sc

def plot_mst(adata, mst_children, basis='umap', color=None, ax=None):
        data = adata.obsm['X_' + basis]
        cluster_labels = adata.obs['int_cluster'].to_numpy()
        cluster_centres = calculate_cluster_centres(data, cluster_labels)
        if ax is None:
            fig, ax = plt.subplots(1, 1, )
        sc.pl.embedding(adata, basis=basis, ax=ax, show=False, color=color, frameon=False)
        start_node_indicator = np.array([True for i in range(len(mst_children)) ])
        for root, kids in mst_children.items():
            for child in kids:
                start_node_indicator[child] = False
                x_coords = [cluster_centres[root][0], cluster_centres[child][0]]
                y_coords = [cluster_centres[root][1], cluster_centres[child][1]]
                ax.plot(x_coords, y_coords, 'k-')