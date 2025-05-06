from tqdm import tqdm
import scanpy as sc
import numpy as np
import pandas as pd
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import statsmodels.distributions.empirical_distribution as edf
from scipy.interpolate import interp1d
from datetime import datetime
from .. import CellFate

def current():
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time
def split_list(lst, sizes):
    result = []
    start = 0
    for size in sizes:
        result.append(lst[start:start+size])
        start += size
    return result    
class TimeSeriesRoadmap:
    def __init__(self, adata, model, embedding_key, velocity_key, time_key):
        self.adata = adata
        self.embedding_key = embedding_key
        self.velocity_key = velocity_key
        self.time_key = time_key
        self.ts = np.sort(np.unique(adata.obs[time_key]))
        
        self.tps = [str(self.ts[i]) + '_' + str(self.ts[i+1]) for i in range(len(self.ts) - 1)]
        self.state_map = {tp:{} for tp in self.tps}
        self.model = model
        self.clf = None

    def set_cell_type_classifier(self, clf, cell_type_key='cell_type'):
        self.clf = clf
        self.cell_type_key = cell_type_key
        
    def fit_cell_type_classifier(self, cell_type_key='cell_type'):
        self.cell_type_key = cell_type_key
        _, self.clf = pygot.tl.analysis.learn_embed2class_map(self.adata, self.embedding_key, cell_type_key)
        
    def _state_coupling(
        self,
        n_neighbors=10,
        sample_size=100,
        scale=1.25,
        dt=None,
        check_step=50,
        appro=True,
        mode='bwd',
        
    ):
        assert self.clf, 'use `fit_cell_type_classifier` first!'
        assert mode in ['bwd', 'fwd']
        
        for i in range(len(self.ts) - 1):
            
            start = self.ts[i]
            end = self.ts[i+1]
            if dt is None:
                dt = (end - start) / check_step
                
            tp = self.tps[i]
            x0_obs = self.adata.obs.loc[self.adata.obs[self.time_key] == start]
            x1_obs = self.adata.obs.loc[self.adata.obs[self.time_key] == end]
            x0x1_ad = self.adata[np.concatenate([x0_obs.index, x1_obs.index])]
            
            state_coupling, state_coupling_dist, cell_coupling, meta = time_series_transition_map(
                x0x1_ad, 
                self.model,
                self.clf,
                start,
                end,
                time_key=self.time_key,
                embedding_key=self.embedding_key, 
                cell_type_key=self.cell_type_key,
                min_cell=5, backward= mode == 'bwd', 
                n_neighbors=n_neighbors,
                dt=dt, sample_size=sample_size, scale=scale,
                appro=appro, check_step=check_step,
                
            )
            
            self.state_map[tp][mode] = state_coupling
            self.state_map[tp][mode+'_dist'] = state_coupling_dist
            self.state_map[tp][mode+'_cc'] = cell_coupling
            self.state_map[tp][mode+'_meta'] = meta
            
    def _generate_null_dist(self, n_permutation=100):
        for tp in self.tps:
            null_dist = []
            x0_cell_list = self.state_map[tp]['fwd_meta'][self.cell_type_key].unique()
            x0_cell_idx_list = [np.where(self.state_map[tp]['fwd_meta'][self.cell_type_key] == c)[0] for c in x0_cell_list]
            x0_cell_num_list = np.array([ len(n) for n in x0_cell_idx_list])
    
            x1_cell_list = self.state_map[tp]['bwd_meta'][self.cell_type_key].unique()
            x1_cell_idx_list = [np.where(self.state_map[tp]['bwd_meta'][self.cell_type_key] == c)[0] for c in x1_cell_list]
            x1_cell_num_list = np.array([ len(n) for n in x1_cell_idx_list])
            N, M = len(self.state_map[tp]['fwd_meta']), len(self.state_map[tp]['bwd_meta'])
            fwd_cell_coupling = self.state_map[tp]['fwd_cc'][x1_cell_list].to_numpy()
            bwd_cell_coupling = self.state_map[tp]['bwd_cc'][x0_cell_list].to_numpy()
            
            for i in tqdm(range(n_permutation)):
                x0_cell_idx_list = split_list(np.random.permutation(N), x0_cell_num_list)
                x1_cell_idx_list = split_list(np.random.permutation(M), x1_cell_num_list)
                fwd_nsc = np.array([fwd_cell_coupling[idxs].mean(axis=0) for idxs in x0_cell_idx_list])
                bwd_nsc = np.array([bwd_cell_coupling[idxs].mean(axis=0) for idxs in x1_cell_idx_list]).T
                nsc = fwd_nsc + bwd_nsc
                nsc /= nsc.sum(axis=0)
                null_dist.append(nsc.flatten())
            null_dist = np.concatenate(null_dist)
            self.state_map[tp]['null'] = null_dist
    
    
         
    def compute_state_coupling(
        self,
        n_neighbors=10,
        sample_size=100,
        scale=1.25,
        dt=None,
        check_step=50,
        n_permutation=100,
        appro=True,
        **kwargs
    ):
        
            
        for mode in ['fwd', 'bwd']:
            print(current(), '\t Compute State Coupling {}'.format(mode))
            self._state_coupling(
                n_neighbors=n_neighbors,
                sample_size=sample_size,
                scale=scale,
                dt=dt,
                check_step=check_step,
                appro=appro,
                mode=mode,
                **kwargs
            )
            
        for tp in self.tps:
            fwd_chain, bwd_chain = self.state_map[tp]['fwd'], self.state_map[tp]['bwd']
            self.state_map[tp]['fbwd'] = aggregate_state_coupling(fwd_chain, bwd_chain)
            
        #print(current(), '\t Generate Null Distribution')
        #self._generate_null_dist(n_permutation)
        print(current(), '\t Done')
        
    def filter_state_coupling(
        self, 
        max_cutoff=0.4,
        pvalue=0.001,
        hard_cutoff=None
    ):
        filtered_fbwd_list = []
        for key in self.state_map.keys():
            if hard_cutoff is not None:
                cutoff = hard_cutoff
            else:
                if len(self.state_map[key]['fbwd']) > 1:
                    cutoff = min(max_cutoff, get_cutoff(self.state_map[key]['null'], pvalue=pvalue))
                else:
                    cutoff = max_cutoff
            print(cutoff)
            self.state_map[key]['cutoff'] = cutoff
            filtered_fbwd_list.append((self.state_map[key]['fbwd'] > cutoff) * self.state_map[key]['fbwd'])
            self.state_map[key]['filtered_fbwd'] = filtered_fbwd_list[-1]
        return filtered_fbwd_list


def get_cutoff(sample, pvalue=0.001):
    sample = sample[~np.isnan(sample)]
    sample_edf = edf.ECDF(sample)
    
    slope_changes = sorted(set(sample))
    
    sample_edf_values_at_slope_changes = [ sample_edf(item) for item in slope_changes]
    inverted_edf = interp1d(sample_edf_values_at_slope_changes, slope_changes)
    cutoff = inverted_edf(1-pvalue)
    return cutoff


def Null_state_coupling(
    cell_coupling, 
    cell_idx_list
):
    cell_idx_list = split_list(np.random.permutation(N), x0_cell_num_list)
    return cell_coupling[cell_idx_list].mean(axis=0)
    p = meta[cell_type_key].tolist()
    np.random.shuffle(p)    
    meta['permutated'] = p
    nsc = meta.groupby('permutated').apply(lambda x: cell_coupling.loc[x.index].mean(axis=0))
    return nsc
    
def aggregate_state_coupling(fwd_chain, bwd_chain):
    agg = fwd_chain.loc[bwd_chain.index][bwd_chain.columns] + bwd_chain
    agg = agg / agg.sum(axis=0)
            
    return agg   

def time_series_transition_map(
    stage_adata,
    model,
    clf,
    current_stage,
    next_stage,
    n_neighbors=10,
    sigma=0.1,
    embedding_key = 'X_pca',
    time_key = 'stage_numeric',
    cell_type_key='cell_state',  
    dt=0.01,
    check_step=50,
    scale = 1.0,
    min_cell = 20,
    backward=True,
    appro=False,
    sample_size=100,
    mode='stochastic'
    
):
    if backward:
        b, a = current_stage, next_stage
    else:
        a, b = current_stage, next_stage
    print('From {} to {}'.format(a, b))
    
    cell_state_counts = stage_adata.obs[cell_type_key].value_counts()
    
    stage_adata = stage_adata[stage_adata.obs[cell_type_key].isin(cell_state_counts[cell_state_counts > min_cell].index)]
    
    
    next_stage_cell_state = stage_adata.obs.loc[stage_adata.obs[time_key] == next_stage][cell_type_key].unique().tolist()
    current_stage_cell_state = stage_adata.obs.loc[stage_adata.obs[time_key] == current_stage][cell_type_key].unique().tolist()
    
    cf = CellFate(
        stage_adata,
        embedding_key=embedding_key, model=model, 
        
        dt=dt, check=True, appro=appro, check_step=check_step,
        n_neighbors=n_neighbors, sigma=sigma,
        
                 )
    
    next_meta = stage_adata.obs.loc[stage_adata.obs[time_key] == next_stage]
    next_meta[cell_type_key] = next_meta[cell_type_key].astype(str)
    current_meta = stage_adata.obs.loc[stage_adata.obs[time_key] == current_stage]
    current_meta[cell_type_key] = current_meta[cell_type_key].astype(str)
    t_diff = next_stage - current_stage
    if backward:
        cell_type_list = current_stage_cell_state
        meta = next_meta
        x0_adata = stage_adata[stage_adata.obs[time_key] == next_stage]
        x1_stage = next_stage - (t_diff * scale)
    else:
        cell_type_list = next_stage_cell_state
        meta = current_meta
        x0_adata = stage_adata[stage_adata.obs[time_key] == current_stage]
        x1_stage = current_stage + (t_diff * scale)
        
    cf.setup_cell_fate(cell_type_key=cell_type_key,cell_type_list=cell_type_list, 
                      trained=True, report=False, specified_model=clf)
    cell_coupling, cell_traj_dist = cf.pred_cell_fate(x0_adata, time_key, x1_stage, sample_size=sample_size, mode=mode)

    cc = cell_coupling.copy().to_numpy()
    cc[cc < (1./ cc.shape[1])] = np.nan
    cc[cc > (1./ cc.shape[1])] = 1.
    traj_dist = pd.DataFrame(cell_traj_dist['traj_dist'].to_numpy()[:,None] * cc, index=meta.index, columns = cell_coupling.columns)
    traj_dist = np.stack(meta.groupby(cell_type_key).apply(lambda x: np.nanmean(traj_dist.loc[x.index], axis=0)))
    traj_dist = traj_dist.T if backward else traj_dist
    
    state_coupling = meta.groupby(cell_type_key).apply(lambda x: cell_coupling.loc[x.index].mean(axis=0))
    state_coupling = state_coupling.T if backward else state_coupling
    state_coupling_dist = pd.DataFrame(traj_dist, columns=state_coupling.columns, index=state_coupling.index)
    
    
    del cf
    
    return state_coupling, state_coupling_dist, cell_coupling, meta
