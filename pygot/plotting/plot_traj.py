import matplotlib.pyplot as plt
import matplotlib
import scanpy as sc
import numpy as np
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import fcluster
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

def plot_cell_fate_embedding(adata, color, 
                             obsm_key='descendant', basis='umap', **kwargs):
    
    adata.obs[color] = adata.obsm[obsm_key][color]
    sc.pl.embedding(adata, color=color,basis=basis, **kwargs)
    #del adata.obs[color]

def cluster_series(data,  num_clusters = 6, method='average', metric='euclidean', ):
    
    
    cluster_grid = sns.clustermap(data, 
               col_cluster=False, standard_scale=0, cmap='Spectral_r', xticklabels = False, method=method, metric=metric)
    plt.close() 
    row_linkage = cluster_grid.dendrogram_row.linkage

    
    row_clusters = fcluster(row_linkage, num_clusters, criterion='maxclust')
    return row_clusters

def plot_dynamical_genes_clusetermap(
        adata,
        layer=None,
        pseudotime_key='pseudotime', 
        row_clusters=None, 
        n_clusters=4, 
        method='weighted', 
        metric='correlation', 
        gene_font_size=8,
        color=None,
        gene_color=None,
        cmap='Spectral_r',
        highlight_genes = None,
        show_gene=True,
        **kwargs
    ):
    sorted_idx = adata.obs[pseudotime_key].sort_values().index
    # preprocess

    if layer is None:
        data_scaled = minmax_scale(adata[adata.obs.sort_values(pseudotime_key).index].X.toarray()).T
    else:
        data_scaled = minmax_scale(adata[adata.obs.sort_values(pseudotime_key).index].layers[layer].toarray()).T
        
    data = pd.DataFrame(data_scaled, index=adata.var.index)
    
    # pca-kmeans-cluster
    data_pca = PCA(n_components=min(20, data.shape[0])).fit_transform(data_scaled)
    
    row_clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(data_pca)

    # order genes by peak time
    unique_clusters = np.unique(row_clusters)
    
    gene_peak_time = np.sum((data_scaled / data_scaled.sum(axis=1)[:,None]) * np.array(range(data_scaled.shape[1])), axis=1)
    label_order = np.array([np.median(gene_peak_time[np.where(row_clusters == i)[0]]) for i in range(len(unique_clusters))]).argsort()
    
    m = dict(zip(label_order, range(len(unique_clusters))))
    row_clusters = np.array([m[r] for r in row_clusters])

    gene_idx = []
    for i in range(len(unique_clusters)):
        sub_idx = np.where(row_clusters == i)[0]
        sub_idx[gene_peak_time[sub_idx].argsort()]
        gene_idx.append(sub_idx)
    gene_idx = np.concatenate(gene_idx)

    # coloring cells by pseudotime and user-definition
    cmap_t = sns.color_palette("viridis", as_cmap=True) 

    pseudotime_colors = [cmap_t(t) for t in np.linspace(0,1, data.shape[1])]

    if not color is None:
        cell_type_colors = [to_rgba(c) for c in adata.uns['{}_colors'.format(color)][adata[sorted_idx,:].obs[color].cat.codes]]
        col_colors = pd.DataFrame([pseudotime_colors, cell_type_colors], index=['', '']).T
    else:
        col_colors = pseudotime_colors

    # coloring genes by clusters and user-definition
    cmap_g = ListedColormap(sns.color_palette("RdBu_r", len(unique_clusters) ))
    cluster_colors = np.array([cmap_g(cluster) for cluster in row_clusters])

    if not gene_color is None:
        cmap_g = ListedColormap(sns.color_palette("Greys", len(unique_clusters) ))
        values = minmax_scale(adata.var[gene_color].to_numpy())
        g_colors = np.array([cmap_g(v) for v in values])
        
        row_colors = pd.DataFrame([list(g_colors[gene_idx]), list(cluster_colors[gene_idx])], index=[gene_color, 'cluster']).T
        row_colors['idx'] = adata.var.index[gene_idx]
        row_colors = row_colors.set_index('idx')
    else:
        row_colors = cluster_colors[gene_idx]

    
    # plotting
    cluster_grid = sns.clustermap(data.loc[data.index[gene_idx]], 
               col_cluster=False, 
               row_cluster=False,
               cmap=cmap,
               xticklabels = False, 
               yticklabels = True,
               method=method,
               metric=metric, 
               cbar_pos=None,     # 去掉 colorbar
               row_colors=row_colors, 
               col_colors=col_colors,
               **kwargs)

    # show gene setting
    if not show_gene:
        cluster_grid.ax_heatmap.yaxis.set_visible(False)

    else:
        if not highlight_genes is None:
            ax = cluster_grid.ax_heatmap

            yticks = ax.get_yticks()  
            yticklabels = [tick.get_text() for tick in ax.get_yticklabels()]  

            
            new_labels = [
                ""  
                for label in yticklabels
            ]
            ax.set_yticklabels(new_labels, fontsize=10)
            ax.tick_params(axis="y", left=False) 

            for gene in highlight_genes:
                if gene in yticklabels:
                    idx = yticklabels.index(gene)
                    y_pos = yticks[idx]
                    
                    right_bound = data.shape[1] + 0.5  
                    ax.plot([right_bound, right_bound + 10], [y_pos, y_pos], color='black', lw=1.5, clip_on=False)  
                    ax.text(right_bound + 10.5, y_pos, gene, ha="left", va="center", fontsize=gene_font_size)  

            ax.tick_params(axis="both", which="both", length=0)  
        else:
            cluster_grid.ax_heatmap.set_yticklabels(
            cluster_grid.ax_heatmap.get_yticklabels(),
            fontsize=gene_font_size,       
            rotation=0          
        )
    return cluster_grid, pd.DataFrame([row_clusters[gene_idx], adata.var.index[gene_idx]], index=['cluster', 'gene']).T



def plot_trajectory(adata, traj, basis='pumap', title='', ax=None, embedding_kw=None, **kwargs):
    
    marker = matplotlib.markers.MarkerStyle('o', fillstyle='none')
    
    embedding_kw = {} if embedding_kw is None else embedding_kw.copy()
    if ax is None:
        fig, ax = plt.subplots(1,1, **kwargs)
    ax.axis('off')
    sc.pl.embedding(adata, basis=basis, 
                    ax=ax, show=False, colorbar_loc=None, title=title,
                    **embedding_kw)
    for i in range(traj.shape[1]):
        ax.plot(traj[:,i,0], traj[:,i,1], color='black',alpha=.3, linewidth=1)
    ax.scatter(traj[0,:,0], traj[0,:,1], color='blue', s=20, label='start')
    ax.scatter(traj[-1,:,0], traj[-1,:,1], color='red', s=20, label='end', marker=marker)
    
