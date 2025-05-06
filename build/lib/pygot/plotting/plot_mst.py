import numpy as np
import matplotlib.pyplot as plt
from pygot.tools.traj.mst import calculate_cluster_centres
import scanpy as sc
def plot_mst(adata, mst_children, basis='umap', color=None):
        data = adata.obsm['X_' + basis]
        cluster_labels = adata.obs['int_cluster'].to_numpy()
        cluster_centres = calculate_cluster_centres(data, cluster_labels)
       
        fig, ax = plt.subplots(1, 1, )
        # 绘制散点图
        sc.pl.embedding(adata, basis=basis, ax=ax, show=False, color=color)
        start_node_indicator = np.array([True for i in range(len(mst_children)) ])
        # 绘制最小生成树
        for root, kids in mst_children.items():
            for child in kids:
                start_node_indicator[child] = False
                x_coords = [cluster_centres[root][0], cluster_centres[child][0]]
                y_coords = [cluster_centres[root][1], cluster_centres[child][1]]
                ax.plot(x_coords, y_coords, 'k-')
                ax.arrow(x_coords[-2], y_coords[-2], x_coords[-1] - x_coords[-2], y_coords[-1] -  y_coords[-2], head_width=0.9, fc='black', ec='black', length_includes_head=True,)

        start_nodes = np.where(start_node_indicator == True)[0]
        # 绘制source states
        for start_node in start_nodes:
             ax.scatter(cluster_centres[start_node][0], cluster_centres[start_node][1], marker='*', s=200,color='blue')

         # 绘制terminal states
        for root, kids in mst_children.items(): 
             if len(kids) == 0:
                 ax.scatter(cluster_centres[root][0], cluster_centres[root][1], marker='*', s=200, color='red')
           


        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('MST Visualization on Scatter Plot')
       
        plt.show()