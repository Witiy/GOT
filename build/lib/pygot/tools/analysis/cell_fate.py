from pygot.tools.traj import velocity_graph, diffusion_graph
from pygot.preprocessing import mutual_nearest_neighbors
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import statsmodels.distributions.empirical_distribution as edf
from scipy.interpolate import interp1d
from cellrank._utils._linear_solver import _solve_lin_system
from datetime import datetime




def current():
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


class CellFate:
    """Cell fate prediction based on the markov chain.
    
    .. math:: 
        r = \\frac{1}{|C_k|}\sum_{i \in C_k}{R_{\\dot, i}} \\\\
        P = (I - Q)^{-1} r \\\\ 

    where :math:`R` represents non-target cells to target cells transition matrix, :math:`C_k` is the set of target cell type k.
    Besides, :math:`Q` is the transition matrix from the non-target cells to the non-target cells, and :math:`I` is the identity matrix.
    The final solution is :math:`P`, which represents the absorbing probabilities for the target cell type.

    Example:
    ----------
    
    ::

        cf = pygot.tl.analysis.CellFate()
        cf.fit(adata, embedding_key='X_pca', velocity_key='velocity_pca', cell_type_key='clusters', target_cell_types=['Beta', 'Alpha', 'Delta', 'Epsilon'])
        adata.obs[adata.obsm['descendant'].columns] = adata.obsm['descendant']
        sc.pl.umap(adata, color=adata.obsm['descendant'].columns, ncols=2)

    """
    def __init__(self):
        pass
    
    def fit(
        self, 
        adata, 
        embedding_key, 
        velocity_key, 
        cell_type_key, 
        target_cell_types=None, 
        target_cell_idx=None,
        n_neighbors=30,
        mutual=True,
        sde=True,
        D=1.
    ):
        """
        fit the cell fate prediction model and export the result into adata.obsm['descendant'] and adata.obsm['ancestor']
        
        Arguments:
        ----------
        adata: :class:`anndata.AnnData`
            AnnData object
        embedding_key: `str`
            The key of the embedding in adata.obsm
        velocity_key: `str`
            The key of the velocity in adata.obsm
        cell_type_key: `str`
            The key of the cell type in adata.obs
        target_cell_types: `list` (default: None)
            The list of target cell types
        target_cell_idx: `list` (default: None)
            The list of target cell indices
        n_neighbors: `int` (default: 30)
            The number of neighbors for the nearest neighbors graph
        mutual: `bool` (default: True)
            Whether to use mutual nearest neighbors graph. Might isolate some cells if set to True
        sde: `bool` (default: True)
            Whether to use inner product kernel or cosine kernel
        D: `float` (default: 1.)
            The diffusion factor. Larger D means larger diffusion.

        """

        
        
        assert (target_cell_types is not None) or (target_cell_idx is not None); 'Must offer target_cell_types or target_cell_idx'
        adata.obs['transition'] = 0
        
        if target_cell_types is not None:
            adata.obs.loc[(adata.obs[cell_type_key].isin(target_cell_types)), 'transition'] = 1
        
        model = TimeSeriesRoadmap(adata, embedding_key=embedding_key, velocity_key=velocity_key, time_key='transition',
                                  sde=sde, D=D)
        model.compute_state_coupling(cell_type_key=cell_type_key, n_neighbors=n_neighbors, mutual=mutual)
        model.export_result()
        self.model = model

    def get_cluster_transition_map(
        self,
        pvalue=1e-3,
        max_cutoff=0.45
    ):
        """

        Get the cluster transition map based on the cell fate prediction model.

        Arguments:
        ----------
        pvalue: `float` (default: 1e-3)
            The pvalue cutoff for the cluster transition map
        max_cutoff: `float` (default: 0.45)
            The maximum cutoff for the cluster transition map
        
        """
        
        transition_list = self.model.filter_state_coupling(pvalue=pvalue, max_cutoff=max_cutoff)
        return transition_list[0]


class TimeSeriesRoadmap:
    
    """Developmental tree inference based on the velocity graph.
    

    Example:
    ----------
    
    ::
    
        embedding_key = 'X_pca'
        velocity_key = 'velocity_pca'
        time_key = 'stage_numeric'
        cell_type_key = 'clusters'
        
        roadmap = pygot.tl.analysis.TimeSeriesRoadmap(adata, embedding_key, velocity_key, time_key)
        roadmap.fit(cell_type_key='clusters', n_neighbors=30)
        
        filtered_state_coupling_list = roadmap.filter_state_coupling(pvalue=0.001) #permutation test to fileter cell type coupling

    """
    def __init__(self, adata, embedding_key, velocity_key, time_key, sde=False, D=1.):
        self.adata = adata
        self.embedding_key = embedding_key
        self.velocity_key = velocity_key
        self.time_key = time_key
        self.ts = np.sort(np.unique(adata.obs[time_key]))
        self.state_map = {t:{} for t in self.ts[:-1]}
        self.sde = sde
        self.D = D
        
    def compute_state_coupling(
        self,
        cell_type_key='cell_type',
        n_neighbors=None,
        permutation_iter_n=100,
        mutual=True,
    ):
        ad = self.adata
        self.cell_type_key = cell_type_key
        print(current(), '\t Compute transition roadmap among', self.ts)
        
        ad.obs['idx'] = range(len(ad))
        for i in range(len(self.ts) - 1):
            start = self.ts[i]
            end = self.ts[i+1]
            print(current(), '\t Compute transition between {} and {}'.format(start, end))
            x0_obs = ad.obs.loc[ad.obs[self.time_key] == start]
            x1_obs = ad.obs.loc[ad.obs[self.time_key] == end]
            
            idx = pd.concat([x0_obs['idx'], x1_obs['idx']])
            embedding = ad.obsm[self.embedding_key][idx.tolist()]
            embedding_v = ad.obsm[self.velocity_key][idx.tolist()]
            x0x1_ad = sc.AnnData(obs=ad.obs.loc[idx.index])
            x0x1_ad.obsm[self.embedding_key] = embedding
            x0x1_ad.obsm[self.velocity_key] = embedding_v
            
            x0x1_ad = ad[np.concatenate([x0_obs.index, x1_obs.index])].copy()
            
            fwd, bwd, fbwd, null, descendant, ancestor = time_series_transition_map(
                x0x1_ad, 
                self.embedding_key, 
                self.velocity_key, 
                self.time_key, 
                start, end,
                norm=0, 
                n_neighbors=n_neighbors,
                cell_type_key=cell_type_key,
                permutation_iter_n=permutation_iter_n,
                mutual=mutual,
                sde=self.sde,
                D=self.D
            )
            self.state_map[start]['fwd'] = fwd
            self.state_map[start]['bwd'] = bwd
            self.state_map[start]['fbwd'] = fbwd
            self.state_map[start]['null'] = null
            self.state_map[start]['null_iedf'] = fit_null_distribution(null)
            self.state_map[start]['descendant'] = descendant
            self.state_map[start]['ancestor'] = ancestor
            
    def filter_state_coupling(
        self, 
        pvalue=0.001,
        max_cutoff=0.45
    ):
        filtered_fbwd_list = []
        for key in self.state_map.keys():
            if len(self.state_map[key]['fbwd']) > 1:
                cutoff = min(max_cutoff, self.state_map[key]['null_iedf'](1-pvalue))
            else:
                cutoff = max_cutoff
            self.state_map[key]['cutoff'] = cutoff
            filtered_fbwd_list.append((self.state_map[key]['fbwd'] > cutoff) * self.state_map[key]['fbwd'])
            self.state_map[key]['filtered_fbwd'] = filtered_fbwd_list[-1]
        return filtered_fbwd_list

    def export_result(
        self,
    ):
        print("Export result into adata.obsm['descendant'] and adata.obsm['ancestor']")
        descendant_col = np.unique(np.concatenate([self.state_map[key]['descendant'].columns for key in self.state_map.keys()]))
        ancestor_col = np.unique(np.concatenate([self.state_map[key]['ancestor'].columns for key in self.state_map.keys()]))
        
        self.adata.obsm['descendant'] = pd.DataFrame(np.zeros(shape=(len(self.adata.obs), len(descendant_col))), 
                                                     index=self.adata.obs.index, columns=descendant_col)
        self.adata.obsm['ancestor'] = pd.DataFrame(np.zeros(shape=(len(self.adata.obs), len(ancestor_col))), 
                                                index=self.adata.obs.index, columns=ancestor_col)

        for key in self.state_map.keys():
            m = self.state_map[key]['descendant']
            self.adata.obsm['descendant'].loc[m.index, m.columns] = m
            m = self.state_map[key]['ancestor']
            self.adata.obsm['ancestor'].loc[m.index, m.columns] = m
            
        start_cells = self.adata.obs.loc[self.adata.obs[self.time_key] == self.ts[0]].index
        end_cells = self.adata.obs.loc[self.adata.obs[self.time_key] == self.ts[-1]].index
        for cell_type in descendant_col:
            self.adata.obsm['descendant'].loc[end_cells, cell_type] = np.array(self.adata.obs.loc[end_cells][self.cell_type_key] == cell_type).astype(float)
        for cell_type in ancestor_col:
            self.adata.obsm['ancestor'].loc[start_cells, cell_type] = np.array(self.adata.obs.loc[start_cells][self.cell_type_key] == cell_type).astype(float)
            
        


def fit_null_distribution(sample):
    sample = sample[~np.isnan(sample)]
    sample_edf = edf.ECDF(sample)
    
    slope_changes = sorted(set(sample))
    
    sample_edf_values_at_slope_changes = [ sample_edf(item) for item in slope_changes]
    inverted_edf = interp1d(sample_edf_values_at_slope_changes, slope_changes)
    return inverted_edf

def split_list(lst, sizes):
    result = []
    start = 0
    for size in sizes:
        result.append(lst[start:start+size])
        start += size
    return result


def time_series_transition_map(
    x0x1_ad, 
    embedding_key, 
    velocity_key, 
    time_key, 
    current_stage, 
    next_stage, 
    cell_type_key = 'cell_type', 
    n_neighbors=None, 
    norm=0,
    permutation_iter_n = 100,
    mutual=True,
    sde = False,
    D=1.,
):
    
    print(current(), '\t Compute velocity graph')
    # Add an index column
    x0x1_ad.obs['idx'] = range(len(x0x1_ad))

    if n_neighbors is None:
        n_neighbors = min(50, max(15, int(len(x0x1_ad) * 0.0025)))
        print('{} to {} | Number of neighbors: {}'.format(current_stage,  next_stage, n_neighbors))
    # Compute neighbors based on the embedding
    
    if len(x0x1_ad) < 8192: #scanpy exact nn cutoff
        #symetric graph
        #sc.pp.neighbors(x0x1_ad, n_neighbors=n_neighbors, use_rep=embedding_key)
        #graph = x0x1_ad.obsp['connectivities']
        graph = mutual_nearest_neighbors(x0x1_ad, n_neighbors=n_neighbors, use_rep=embedding_key, mutual=False, sym=True)
    else:
        #mutual nearest neighbors graph
        if mutual:
            graph = mutual_nearest_neighbors(x0x1_ad, n_neighbors=n_neighbors, use_rep=embedding_key, mutual=True, sym=False)
        else:
            graph = mutual_nearest_neighbors(x0x1_ad, n_neighbors=n_neighbors, use_rep=embedding_key, mutual=False, sym=True)
    
    
    
    
    # Get indices for the current and next stage cells
    x0_idx = x0x1_ad.obs.loc[x0x1_ad.obs[time_key] == current_stage, 'idx'].to_numpy()
    x1_idx = x0x1_ad.obs.loc[x0x1_ad.obs[time_key] == next_stage, 'idx'].to_numpy()
    
    if sde == False:
        # Compute the velocity graph
        velocity_graph(x0x1_ad, embedding_key, velocity_key, graph=graph, split_negative=True)
        
        P_fwd = x0x1_ad.uns['velocity_graph']
        P_bwd = -x0x1_ad.uns['velocity_graph_neg']
    
    else:
        P_fwd = diffusion_graph(X=x0x1_ad.obsm[embedding_key], V=x0x1_ad.obsm[velocity_key], graph=graph, D=D)
        P_bwd = diffusion_graph(X=x0x1_ad.obsm[embedding_key], V=-x0x1_ad.obsm[velocity_key], graph=graph, D=D)
        
    print(current(), '\t Convert into markov chain')
    
    P_fwd /= P_fwd.sum(axis=1)
    P_fwd = csr_matrix(P_fwd)
    
    

    P_bwd /= P_bwd.sum(axis=1)
    P_bwd = csr_matrix(P_bwd)
    
    x0x1_markov = P_fwd[x0_idx][:, x1_idx]
    x1x0_markov = P_bwd[x1_idx][:, x0_idx]
    Q_bwd = P_bwd[x1_idx][:, x1_idx]
    Q_fwd = P_fwd[x0_idx][:, x0_idx]
    
    fixed_fwd = np.array(Q_fwd.sum(axis=1) == 0).flatten()
    fixed_bwd = np.array(Q_bwd.sum(axis=1) == 0).flatten()
    
    x0_obs = x0x1_ad.obs.iloc[x0_idx]
    x1_obs = x0x1_ad.obs.iloc[x1_idx]
    
    x0_cell_list = x0_obs[cell_type_key].unique()
    x0_cell_idx_list = [np.where(x0_obs[cell_type_key] == c)[0] for c in x0_cell_list]
    x0_cell_num_list = np.array([ len(n) for n in x0_cell_idx_list])
    
    
    x1_cell_list = x1_obs[cell_type_key].unique()
    x1_cell_idx_list = [np.where(x1_obs[cell_type_key] == c)[0] for c in x1_cell_list]
    x1_cell_num_list = np.array([ len(n) for n in x1_cell_idx_list])
    
    
    s_fwd = np.array([x0x1_markov[:, f].sum(axis=1) for f in x1_cell_idx_list])[:,:,0].T
    s_bwd = np.array([x1x0_markov[:, f].sum(axis=1) for f in x0_cell_idx_list])[:,:,0].T
    
    descendant = s_fwd / x1_cell_num_list
    ancestor = s_bwd / x0_cell_num_list

    
    descendant *= (np.array((1-Q_fwd.sum(axis=1))).flatten() / (descendant.sum(axis=1) + 1e-6))[:,None]
    ancestor *= (np.array((1-Q_bwd.sum(axis=1))).flatten() / (ancestor.sum(axis=1) + 1e-6))[:,None]
    print(current(), '\t Solve abosorbing probabilities')
    IQR_fwd = _solve_lin_system(Q_fwd[~fixed_fwd,:][:,~fixed_fwd], csr_matrix(descendant[~fixed_fwd,:]), use_eye=True, show_progress_bar=False) 
    IQR_bwd = _solve_lin_system(Q_bwd[~fixed_bwd,:][:,~fixed_bwd], csr_matrix(ancestor[~fixed_bwd,:]), use_eye=True, show_progress_bar=False)

    descendant[~fixed_fwd,:] = IQR_fwd
    ancestor[~fixed_bwd,:] = IQR_bwd
    
    ancestor[np.isnan(ancestor)] = 0.
    descendant[np.isnan(descendant)] = 0.
    
    
    def compute_state_coupling(norm=0,):
        
        

        # Compute the forward state coupling matrix
        fwd = np.zeros((len(x0_cell_list), len(x1_cell_list)))
        for i in range(len(x0_cell_list)):
            fwd[i, :] = descendant[x0_cell_idx_list[i], :].sum(axis=0)
        fwd = fwd / fwd.sum(axis=1)[:, None]  # Normalize by rows
        fwd[np.isnan(fwd)] = 0.
        fwd += 1e-3
        
        # Compute the backward state coupling matrix 
        bwd = np.zeros((len(x0_cell_list), len(x1_cell_list)))
        for j in range(len(x1_cell_list)):
                bwd[:, j] = ancestor[x1_cell_idx_list[j], :].sum(axis=0)
        bwd = bwd / bwd.sum(axis=0)  # Normalize by columns
        bwd[np.isnan(bwd)] = 0.
        bwd += 1e-3
        # Combine the forward and backward matrices
        state_coupling = fwd * bwd
        state_coupling = state_coupling / (state_coupling.sum(axis=0) if norm == 0 else state_coupling.sum(axis=1)[:, None])
        
        
        return fwd, bwd, state_coupling

    # Make sure cell type is treated as a string
    x0x1_ad.obs[cell_type_key] = x0x1_ad.obs[cell_type_key].astype(str)
    
    # Calculate the forward and backward state couplings
    state_coupling_fwd, state_coupling_bwd, state_coupling = compute_state_coupling(norm)
    
    state_coupling = pd.DataFrame(state_coupling, index=x0_cell_list, columns=x1_cell_list)
    state_coupling_fwd = pd.DataFrame(state_coupling_fwd, index=x0_cell_list, columns=x1_cell_list)
    state_coupling_bwd = pd.DataFrame(state_coupling_bwd, index=x0_cell_list, columns=x1_cell_list)
    
    bwd_cc = ancestor / ancestor.sum(axis=1)[:,None]
    fwd_cc = descendant / descendant.sum(axis=1)[:,None]
            
    bwd_cc = pd.DataFrame(bwd_cc, 
                        columns=state_coupling.index,
                        index=x1_obs.index)
    fwd_cc = pd.DataFrame(fwd_cc, 
                        columns=state_coupling.columns,
                        index=x0_obs.index)
    
    # Perform permutation testing
    permutation_list = []
    N, M = len(x0_obs), len(x1_obs)
    print(current(), '\t Generate NULL distribution')
    for k in range(permutation_iter_n):
        
        x0_cell_idx_list = split_list(np.random.permutation(N), x0_cell_num_list)
        x1_cell_idx_list = split_list(np.random.permutation(M), x1_cell_num_list)
        
        _, _, permu_state_coupling = compute_state_coupling(norm)
        permutation_list.append(permu_state_coupling.flatten())


    return state_coupling_fwd, state_coupling_bwd, state_coupling, np.concatenate(permutation_list, axis=0), \
            fwd_cc, bwd_cc
    