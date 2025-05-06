import numpy as np
from tqdm import tqdm
import scanpy as sc
from scipy.sparse import csr_matrix
from pygot.utils import find_neighbors, cosine, inner_kernel, split_negative_P


def velocity_graph(data, embedding_key, velocity_key, basis='X_umap', k=30, split_negative=True, copy=False):
    if copy:
        adata = data.copy()
    else:
        adata = data
    P = _transition_matrix(adata, embedding_key, velocity_key, basis=basis, k=k, method='cosine', norm=False)
    if split_negative:
        P, P_neg = split_negative_P(P)
        adata.uns['velocity_graph'] = P
        adata.uns['velocity_graph_neg'] = P_neg
    else:
        adata.uns['velocity_graph'] = P
    #P = csr_matrix((data, (rows, cols)), shape=(adata.shape[0], adata.shape[0])).toarray() 
    
    return adata if copy else None

def _transition_matrix(adata, embedding_key, velocity_key, basis='X_umap', k=30, method='inner', norm=False):
    vt = adata.obsm[velocity_key]
    neighbors = find_neighbors(sc.pp.neighbors(adata, use_rep=basis, n_neighbors=k, copy=True).obsp['connectivities'])
    rows, cols, data = [], [], []
    if method=='inner':
        func = inner_kernel
    elif method == 'cosine':
        func = cosine
    else:
        raise NotImplementedError()
    
    for i in tqdm(range(adata.shape[0])):
        vt_tuple = adata.obsm[embedding_key][np.array(neighbors[i]),:] - adata.obsm[embedding_key][i]
        p = func(vt_tuple, vt[i:i+1,:]).astype(np.float64)
        if method=='inner':
            if sum(np.isinf(p)) > 0: 
                p[~np.isinf(p)] = 0.
                p[np.isinf(p)] = 1.
        if norm:
            p /= np.sum(p)        
        rows.append([i] * len(neighbors[i]))
        cols.append(neighbors[i])
        data.append(p)
    data = np.concatenate(data)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    P = csr_matrix((data, (rows, cols)), shape=(adata.shape[0], adata.shape[0]))
    return P


def coarse_markov_chain(data, embedding_key, velocity_key,  basis='X_umap', k=100, min_chi=2, max_chi=12, copy=False):
    try:
        import pygpcca as gp
    except ImportError:
        raise ImportError(
                "Please install the GPCCA algorithm: `https://github.com/msmdev/pyGPCCA`.")
    adata = data.copy() if copy else data
    P = _transition_matrix(adata, embedding_key, velocity_key, basis=basis, k=k, method='inner').toarray()
    
    gpcca = gp.GPCCA(P, z='LM', method='brandts')
    #gpcca.minChi(min_chi, max_chi)
    gpcca.optimize({'m_min':min_chi, 'm_max':max_chi})
    chi = gpcca.memberships
    P_c = gpcca.coarse_grained_transition_matrix
    adata.obsm['chi'] = chi
    adata.obs['membership'] = chi.argmax(axis=1).astype(str)
    adata.obs['ent_chi'] = -(chi * np.log(chi)).sum(axis=1)
    return adata if copy else None

def project_velocity(data, velocity_key='velocity_pca', embedding_key='X_pca', basis='X_umap', k=30, norm=False, copy=False):
    if copy:
        adata = data.copy()
    else:
        adata = data
    vt = adata.obsm[velocity_key]
    neighbors = find_neighbors(sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=k, copy=True).obsp['connectivities'])
    velocity = []

    for i in tqdm(range(adata.shape[0])):
        vt_tuple = adata.obsm[embedding_key][np.array(neighbors[i]),:] - adata.obsm[embedding_key][i]
        p = inner_kernel(vt_tuple, vt[i:i+1,:])
        p /= np.sum(p)
        vt_vis = adata.obsm[basis][np.array(neighbors[i]),:] - adata.obsm[basis][i]
        velocity.append(np.sum(p[:,None] * vt_vis, axis=0))
        
    velocity = np.array(velocity)
    
    if norm:
        velocity /= np.linalg.norm(velocity, axis=-1)[:, None]    
    adata.obsm['velocity_' + basis.split('_')[-1]] = velocity
    
    return adata if copy else None

