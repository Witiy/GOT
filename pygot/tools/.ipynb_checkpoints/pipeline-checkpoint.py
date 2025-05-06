from scipy.stats import pearsonr
from functools import partial
from datetime import datetime
from pygot.tools.continuity import ContinuityModel
from pygot.tools.model_training import fit_velocity_model
from pygot.tools.flow import latent_velocity
from pygot.tools.beta_mixture import ks_test_beta_mixture
from pygot.plotting import velocity_embedding_stream, plot_root_cell, plot_mst
from pygot.tools.root_identify import determine_source_states, init_candidiates, highlight_extrema, generate_time_points

from pygot.tools.mst import topological_tree, adjust_time_by_structure



import pygot.external.palantir as palantir
import warnings
import scanpy as sc
import torch
import matplotlib.pyplot as plt
import numpy as np

def current():
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time



def iterative_fit_velocity_model(adata, embedding_key, pseudotime_key, split_k, device, velocity_key=None, plot=False, basis=None, cell_type_key=None, **kwargs):
    if velocity_key is None:
        velocity_key = 'velocity_' + embedding_key.split('_')[-1]
    warnings.filterwarnings('ignore')
    print(current()+'\t Start to iterative training')
    old_time = adata.obs[pseudotime_key]
    
    cm = ContinuityModel(adata, embedding_key=embedding_key, n_neighbors=30)
    v_net_train_func = partial(fit_velocity_model,
            adata=adata, time_key='pseudobin',device=device, embedding_key=embedding_key, time_varying=False, **kwargs)
    i = 0
    while True:
        #update pseudobin
        if i != 0:
            pseudotime_key = 'expectation'
        generate_time_points(adata, k=split_k, pseudotime_key=pseudotime_key, sigma=.25)
        #train velocity model
        print(current()+'\t {} iteration: Training velocity model'.format(i))
        model, _ = v_net_train_func()
        if plot:
            fig, axs = plt.subplots(1,3,figsize=(20,5))
            sc.pl.embedding(adata, basis=basis, color=pseudotime_key, ax=axs[0], show=False, title='Old Time', frameon=False)
        #update time
        print(current()+'\t {} iteration: Training time model'.format(i))
        cm.train_density_model(adata, time_key=pseudotime_key, v_net=model.func)
        cm.update_time(adata)
        #plot velocity field
        adata.obsm[velocity_key] = latent_velocity(adata, model.func.net, embedding_key=embedding_key, time_vary=False, time_key=None)
        adata.layers['velocity'] = adata.obsm[velocity_key] @ adata.varm['PCs'].T
        
        if plot:
            velocity_embedding_stream(adata, velocity_key, embedding_key, basis='X_'+basis, k=30,  norm=False, update=True, color=cell_type_key, ax=axs[1], show=False)
            sc.pl.embedding(adata, basis=basis, color='expectation', ax=axs[2], title='New Time',frameon=False)
            plt.show()
            plt.close()

        pcc = pearsonr(old_time, adata.obs['expectation'])[0]
        print(current()+'\t {} iteration: Pearson correlation between new and old time is {:.4f}'.format(i, pcc))
        if pcc > 0.95:
            print(current()+'\t {} iteration:  Convergence ! '.format(i))
            break
        old_time = adata.obs['expectation']
        i+=1
    return model

def single_branch_detection(tree):
    return bool(np.sum([len(tree[key]) > 1 for key in tree.keys()]) == 0)


def single_branch(adata, cell_type_key, kernel, pseudotime_key):
    if kernel == 'dpt':
        adata.uns['iroot'] = adata.uns['ot_root']
        sc.tl.dpt(adata)
        pseuodtime_ot = adata.obs[pseudotime_key]
        ot_tree, _, _ = topological_tree(adata, embedding_key='X_pca', cell_type_key=cell_type_key, time_key=pseudotime_key, start_cell_type=adata.obs[cell_type_key].tolist()[adata.uns['ot_root']])
        
        adata.uns['iroot'] = adata.uns['ot_ct_root']
        sc.tl.dpt(adata)
        pseuodtime_ot_ct = adata.obs[pseudotime_key]
        ot_ct_tree, _, _ = topological_tree(adata, embedding_key='X_pca', cell_type_key=cell_type_key, time_key=pseudotime_key, start_cell_type=adata.obs[cell_type_key].tolist()[adata.uns['ot_ct_root']])
    else:
        palantir.run_palantir(adata,adata.obs.index[adata.uns['ot_root']],n_jobs=1,use_early_cell_as_start=True, eigvec_key="DM_EigenVectors")
        pseuodtime_ot = adata.obs[pseudotime_key]
        ot_tree, _, _ = topological_tree(adata, embedding_key='X_pca', cell_type_key=cell_type_key, time_key=pseudotime_key, start_cell_type=adata.obs[cell_type_key].tolist()[adata.uns['ot_root']])
        
        palantir.run_palantir(adata,adata.obs.index[adata.uns['ot_ct_root']],n_jobs=1,use_early_cell_as_start=True, eigvec_key="DM_EigenVectors")
        pseuodtime_ot_ct = adata.obs[pseudotime_key]
        ot_ct_tree, _, _ = topological_tree(adata, embedding_key='X_pca', cell_type_key=cell_type_key, time_key=pseudotime_key, start_cell_type=adata.obs[cell_type_key].tolist()[adata.uns['ot_ct_root']])

    single_branch_progress = single_branch_detection(ot_tree) & single_branch_detection(ot_ct_tree)

    if single_branch_progress:
        adata.obs[pseudotime_key] = pseuodtime_ot_ct
        tree = ot_ct_tree
    else:
        adata.obs[pseudotime_key] = pseuodtime_ot
        tree = ot_tree

    return single_branch_progress, tree


    
def got_without_time_pipeline(adata, embedding_key, kernel='dpt', ot_split_k=30, 
                     cytotrace=True, cell_type_key=None,  
                     single_branch_detect=True,  scale=False, 
                     plot=False, basis='umap', fig_dir='./fig', split_k='auto', device=None, **kwargs):
    if device is None:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    print(current()+'\t Using extrema in diffmap space to connect the whole graph')
    
    if kernel == 'palantir' or kernel == 'sp':
        palantir.run_diffusion_maps(adata)
        diffmap_key = 'DM_EigenVectors'
        pseudotime_key = 'palantir_pseudotime'
    else:
        sc.tl.diffmap(adata)
        diffmap_key = 'X_diffmap'
        pseudotime_key = 'dpt_pseudotime'

    init_candidiates(adata, diffmap_key=diffmap_key)
    if plot:
        highlight_extrema(adata, basis=basis)
        plt.show()
        plt.close()
   
    print(current()+'\t Search for the best source cell..')
    determine_source_states(adata, kernel=kernel, split_k=ot_split_k, embedding_key=embedding_key, cytotrace=cytotrace, connect_anchor=adata.uns['extrema'])
    tree = None
    if single_branch_detect and cytotrace:
        if cell_type_key is not None:
            print(current()+'\t Determine linear progress or not..')
            single_branch_progress, tree = single_branch(adata, cell_type_key, kernel, pseudotime_key)
            print(current()+'\t Single Branch Progress : {}'.format(single_branch_progress))
        else:
            print('Please off `cell_type_key` to perform single branch detection')
            
    
    if tree is None:
        if kernel == 'dpt':
            adata.uns['iroot'] = adata.uns['ot_root']
            sc.tl.dpt(adata)
        
        else:
            palantir.run_palantir(adata,adata.obs.index[adata.uns['ot_root']],n_jobs=1,use_early_cell_as_start=True, eigvec_key=diffmap_key)
    
    if plot:
        if cytotrace:
            sc.pl.embedding(adata, basis=basis, color=['root_score', 'ct_root_score', pseudotime_key])
        else:
            sc.pl.embedding(adata, basis=basis, color=['root_score',  pseudotime_key])
        plot_root_cell(adata, figsize=(12,5), basis=basis)
        plt.show()
        plt.close()

    if scale and cell_type_key:
        print(current()+'\t Adjust pseudotime underlying infered structure..')
        if tree is None:
            tree, _, _ = topological_tree(adata, embedding_key=embedding_key, cell_type_key=cell_type_key, time_key=pseudotime_key, start_cell_type=adata.obs[cell_type_key].tolist()[adata.uns['ot_root']])
        if plot:
            plot_mst(adata, tree, basis=basis)
            plt.show()
            plt.close()
        adata.obs['scaled_time'] = adjust_time_by_structure(adata, tree, pseudotime_key)
        pseudotime_key = 'scaled_time'
    if split_k == 'auto':
        print(current()+'\t Search for the best split k..')
        split_k_df = ks_test_beta_mixture(adata, pseudotime_key)
        split_k = split_k_df['k'].tolist()[0]
        print('Best split k is {}'.format(split_k))
        
    generate_time_points(adata, k=split_k, pseudotime_key=pseudotime_key, sigma=.25)
    if plot:
        sc.pl.embedding(adata, basis=basis, color=[pseudotime_key, 'pseudobin'])
        plt.show()
        plt.close()
    
    model = iterative_fit_velocity_model(adata, embedding_key, pseudotime_key, split_k=split_k, device=device,
                               plot=plot, basis=basis, cell_type_key=cell_type_key,
                                **kwargs)
    return model
    


