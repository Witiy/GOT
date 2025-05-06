from functools import partial
from datetime import datetime

from .root_identify import determine_source_state, generate_time_points
from .mst import topological_tree
from .model_training import fit_velocity_model
from ...plotting import plot_root_cell, plot_mst
import pygot.external.palantir as palantir
import warnings
import scanpy as sc
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale

def compute_pseudotime(adata, root, embedding_key='X_pca', kernel='dpt'):
    if kernel == 'dpt':
        adata.uns['iroot'] = root
        sc.tl.dpt(adata)
        return adata.obs['dpt_pseudotime']
    elif kernel == 'palantir' or kernel == 'sp':
        palantir.run_palantir(adata,adata.obs.index[root],n_jobs=1,use_early_cell_as_start=True, eigvec_key="DM_EigenVectors")
        return adata.obs['palantir_pseudotime']
    else:
        adata.obs['euclidean_pseudotime'] = minmax_scale(np.mean((adata.obsm[embedding_key] - adata.obsm[embedding_key][root])**2, axis=1))
        return adata.obs['euclidean_pseudotime']

def current():
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


def _fit_velocity_model(adata, embedding_key, device, **kwargs):
   
    warnings.filterwarnings('ignore')
    print(current()+'\t Start to fit velocity model')

    if kwargs.get('v_centric_iter_n') is None:
        kwargs['v_centric_iter_n'] = 100
    if kwargs.get('x_centric_iter_n') is None:
        kwargs['x_centric_iter_n'] = 200

    v_net_train_func = partial(fit_velocity_model,
            adata=adata, time_key='pseudobin',device=device, embedding_key=embedding_key, 
            time_varying=False, **kwargs)

    model, history = v_net_train_func()
    
    return model, history

def single_branch_detection(tree):
    return bool(np.sum([len(tree[key]) > 1 for key in tree.keys()]) == 0)


def post_process(adata, embedding_key, kernel, pseudotime_key, cell_type_key=None, single_branch_detect=True):
    
    print(current()+'\t Determine linear progress or not..')
    
    pseuodtime_ot = compute_pseudotime(adata, adata.uns['ot_root'], embedding_key=embedding_key, kernel=kernel)
    
    if (cell_type_key is None) or single_branch_detect == False:
        adata.obs[pseudotime_key] = pseuodtime_ot
        return False, None
    
    ot_tree, _, _ = topological_tree(adata, embedding_key=embedding_key, cell_type_key=cell_type_key, time_key=pseudotime_key, start_cell_type=adata.obs[cell_type_key].tolist()[adata.uns['ot_root']])
    
    pseuodtime_ot_ct = compute_pseudotime(adata, adata.uns['ot_ct_root'], embedding_key=embedding_key, kernel=kernel)
    
    single_branch_progress = single_branch_detection(ot_tree)

    adata.obs[pseudotime_key] = pseuodtime_ot_ct

    print(current()+'\t Single Branch Progress : {}'.format(single_branch_progress))

    return single_branch_progress, ot_tree


    
def fit_velocity_model_without_time(
        adata, 
        embedding_key,
        precomputed_pseudotime=None,
        kernel='dpt', 
        connect_anchor=True,
        split_m=30, 
        single_branch_detect=True, 
        cytotrace=True, 
        cell_type_key=None,
        plot=False, 
        basis='umap',
        device=None,   
        **kwargs
):
    
    """Estimates velocities and fit trajectories in latent space WITHOUT time label.

    
    Arguments:
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    embedding_key: `str'
        Name of latent space to fit, in adata.obsm
    precomputed_pseudotime: `str` (default: None)
        Name of precomputed pseudotime (in adata.obs), if offers, skip the searching of source cell and use precomputed time as time label to train model
    kernel: 'dpt' or 'palantir' or 'euclidean' (default: 'dpt')
        Pseudotime method, 'dpt' is recommended
    connect_anchor: `bool` (default: False)
        Use extrema in diffusion map space to connect the whole graph
    split_m: `int` (default: 30)
        Number of split. This number should NOT be too small
    single_branch_detect: `bool` (default: True)
        Auto detect single branch so that auto determine use ct_root or ot_ct_root as source cell
    cytotrace: `bool` (default: True)
        Use cytotrace to help. Note cytorace is implemented by Cellrank2
    cell_type_key: `str` (default: None)
        Cell cluster name, in adata.obs. 
    plot: `bool` (default: False)
        Plot the intermediate process
    basis: `str` (default: 'umap')
        Visualization space
    device: :class:`~torch.device`
        torch device
    kwargs: 
        parameter of `pygot.tl.traj.fit_velocity_model`
    
    
       
    Returns
    -------
    model: :class`~ODEwrapper`
        velocity model
    ot_root (.uns): `int`
        best source cell index using transport cost only
    ot_ct_root (.uns): `int`
        best source cell index using both transport cost and cytotrace
    root_score (.obs): `np.ndarray`
        source cell score (higher score higher probability to be source)
    ot_root_score (.obs): `np.ndarray`
        source cell score + alpha * cytotrace score (higher score higher probability to be source)
    expectation (.obs): `np.ndarray`
        updated time
    """
    assert (not plot) or (plot and (not basis is None)), 'please offer `basis` (e.g. umap) if you set `plot` = True'
    assert (not single_branch_detect) or (single_branch_detect and (not cell_type_key is None)), 'please offer `cell_type_key` if you set `single_branch_detect` = True'
    if device is None:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    if precomputed_pseudotime is None:
        if connect_anchor:
            print(current()+'\t Using extrema in diffmap space to connect the whole graph')
        
        if kernel == 'dpt':
            sc.tl.diffmap(adata)
            diffmap_key = 'X_diffmap'
            pseudotime_key = 'dpt_pseudotime'
        elif kernel == 'palantir' or kernel == 'sp':
            palantir.run_diffusion_maps(adata)
            diffmap_key = 'DM_EigenVectors'
            pseudotime_key = 'palantir_pseudotime'
        else:
            palantir.run_diffusion_maps(adata)
            pseudotime_key = 'euclidean_pseudotime'
        '''
        init_candidiates(adata, diffmap_key=diffmap_key)
        if plot:
            highlight_extrema(adata, basis=basis)
            plt.show()
            plt.close()
        '''
        print(current()+'\t Search for the best source cell..')

        determine_source_state(adata, 
                                kernel=kernel, 
                                split_m=split_m, 
                                embedding_key=embedding_key, 
                                cytotrace=cytotrace,
                                connect_anchor=connect_anchor)
        
    
        single_branch_progress, tree = post_process(adata, embedding_key, kernel, pseudotime_key, 
                                                    cell_type_key=cell_type_key,
                                                    single_branch_detect=single_branch_detect)
        
        if plot:
            if cytotrace:
                sc.pl.embedding(adata, basis=basis, color=['root_score', 'ct_root_score', pseudotime_key])
            else:
                sc.pl.embedding(adata, basis=basis, color=['root_score',  pseudotime_key])
            plot_root_cell(adata, figsize=(12,5), basis=basis)
            plt.show()
            plt.close()

            plot_mst(adata, tree, basis=basis)
            plt.show()
            plt.close()

    else:
        pseudotime_key = precomputed_pseudotime


    generate_time_points(adata, k=split_m, pseudotime_key=pseudotime_key, sigma=.0)
    if plot:
        sc.pl.embedding(adata, basis=basis, color=[pseudotime_key, 'pseudobin'])
        plt.show()
        plt.close()
    
    model, history = _fit_velocity_model(adata, embedding_key,  device=device, **kwargs)
    return model, history
    


