import scanpy as sc
import matplotlib.pyplot as plt
def plot_root_cell(adata, color=None, basis='umap', **kwargs):
    if 'ot_ct_root' in adata.uns.keys():
        fig, axes = plt.subplots(1,2, **kwargs)
        root = adata.uns['ot_root']
        sc.pl.embedding(adata, basis=basis, show=False, color=color, ax=axes[0], legend_loc='on data', title='OT root cell')
        axes[0].scatter(adata.obsm['X_'+basis][root,0], adata.obsm['X_'+basis][root,1],  color='red', marker='*', s=200, label='root cell')


        root = adata.uns['ot_ct_root']
        sc.pl.embedding(adata, basis=basis, show=False, color=color, ax=axes[1], legend_loc='on data', title='OT + CytoTrace(alpha:{}) root cell'.format(adata.uns['cytotrace_alpha']))
        axes[1].scatter(adata.obsm['X_'+basis][root,0], adata.obsm['X_'+basis][root,1],  color='red', marker='*', s=200)
    else:
        fig, ax = plt.subplots(1,1, **kwargs)
        root = adata.uns['ot_root']
        sc.pl.embedding(adata, basis=basis, show=False, color=color, ax=ax, legend_loc='on data', title='OT root cell')
        ax.scatter(adata.obsm['X_'+basis][root,0], adata.obsm['X_'+basis][root,1],  color='red', marker='*', s=200, label='root cell')

def highlight_extrema_with_score(adata, basis='umap', **kwargs):
    extrema_names = adata.uns['source_score'].extrema
    c = adata.uns['source_score'].score
    fig,ax = plt.subplots(1,1)
    sc.pl.embedding(adata, show=False, ax=ax, basis=basis, **kwargs)
    ax.scatter(adata[extrema_names].obsm['X_'+basis][:,0], 
               adata[extrema_names].obsm['X_'+basis][:,1], c=c)
    ax.scatter(adata[adata.uns['mc_source']].obsm['X_'+basis][:,0], 
               adata[adata.uns['mc_source']].obsm['X_'+basis][:,1],  color='red', marker='*', s=200)
    