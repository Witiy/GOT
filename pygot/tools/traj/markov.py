import numpy as np

from scipy.sparse import csr_matrix

from .utils import find_neighbors, cosine, split_negative_P




def velocity_graph(data, embedding_key, velocity_key,  split_negative=True, copy=False, graph=None, kernel='cosine'):
    """Compute kNN in latent space using latent velocity

    The transition matrix P is computed based on the cosine similarities which is similar to scvelo.tl.velocity_graph.
    BUT the velocity and the cell state is all in the latent space which could reduce noise compared to gene space.
    :math:`\delta_{ij} = x_j - x_i, \quad j \in N_{basis}(x_i)` The transition matrix is 

    .. math::
    
        \pi_{ij} = \cos(v_i, \delta_{ij})
        
    Arguments:
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    embedding_key: `str`
        Name of latent space, in adata.obsm
    velocity_key: `str` 
        Name of latent velocity, in adata.obsm
    split_negative: `bool` (default: True)
        Split velocity graph into positive and negative
    copy: `bool` (default: False)
        return adata or not
    graph: :class`csr_matrix` (default: None)
        Nearest neighbors graph, if None, use adata.obsp['connectivities']
    
    Returns
    -------
    velocity_graph (.uns): :class`csr_matrix`
        velocity kNN graph, (n_cells, n_cells)
    """
    
    if copy:
        adata = data.copy()
    else:
        adata = data
    if graph is None:
        print("Use adata.obsp['connectivities'] as neighbors, please confirm it is computed in embedding space")
        graph = adata.obsp['connectivities']
    P = cosine_transition_matrix(adata, embedding_key, velocity_key,  graph, norm=False,)
    if split_negative:
        P, P_neg = split_negative_P(P)
        adata.uns['velocity_graph'] = P
        adata.uns['velocity_graph_neg'] = P_neg
    else:
        adata.uns['velocity_graph'] = P
    #P = csr_matrix((data, (rows, cols)), shape=(adata.shape[0], adata.shape[0])).toarray() 
    
    return adata if copy else None



def diffusion_graph(X, V, graph, D=1.):
    
    neighbors = find_neighbors(graph)
    P = []

    rows, cols, data = [], [], []
    
    for i in range(graph.shape[0]):
        if len(neighbors[i]) == 0:
            continue
        dx = X[neighbors[i]] - X[i]
        #scale_factor = np.linalg.norm(dx, axis=1).mean() / np.linalg.norm(V[i])
        p = np.sum(V[i] *dx, axis=1)
        rows.append([i] * len(neighbors[i]))
        cols.append(neighbors[i])
        data.append(p)
        
    data = np.concatenate(data)
    factor = np.median(abs(data))
    data = np.exp(data / (factor*D))
    print('Scale factor:',factor)
    data[np.isinf(data)] = np.max(data[~np.isinf(data)])
    data[np.isinf(data)] = 1. # if all inf
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    P = csr_matrix((data, (rows, cols)), shape=(graph.shape[0], graph.shape[0]))
    
    
    return P




def cosine_transition_matrix(adata, embedding_key, velocity_key, graph, norm=False):
    vt = adata.obsm[velocity_key]
    
    neighbors = find_neighbors(graph)
    rows, cols, data = [], [], []
    
    func = cosine
    
    for i in range(adata.shape[0]):
        if len(neighbors[i]) == 0:
            continue
        vt_tuple = adata.obsm[embedding_key][np.array(neighbors[i]),:] - adata.obsm[embedding_key][i]
        p = func(vt[i][None,:], vt_tuple)
        p[np.isinf(p)] = np.max(p[~np.isinf(p)])
        p[np.isinf(p)] = 1. # if all inf
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

