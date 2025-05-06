from scipy.sparse.csgraph import dijkstra
import ot as pot
import numpy as np
import scanpy as sc
import torch
from tqdm import tqdm
import pandas as pd
from pygot.tools.utils import NeighborsSampler
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
import time
from interpolator import Interpolator
import random
# 1. umap knn graph -> sklearn knn(euclidean)
# 2. interpolated mulitple point in the path -> interpolated one point
# 3. cython version interpolator
def get_correct_x(X0, X1, idx):
    
    if idx >= len(X0):
        return X1[idx-len(X0)]
    else:
        return X0[idx]

def solve_ot(
        M : torch.Tensor, 
        a : torch.Tensor = None, 
        b : torch.Tensor = None
        ):
    """Compute optimal transport plan using earth moving distance algorithm (determinstic)
    
    Parameters
    ----------
        M: Pair-wise distance matrix (n, m)
        a: Distribution of X (n, )
        b: Distribution of Y (n, )

    Returns
    -------
        Optimal transport plan P (n, m), which is a sparse (n entries) matrix, all value is 1. 

    """
    if a == None:
        a = torch.Tensor(pot.unif(M.shape[0]))
    if b == None:
        b = torch.Tensor(pot.unif(M.shape[1]))
    res = pot.emd(a, b, M)
    return res



def sample_map(
        pi : np.array, 
        batch_size: int):
    """Draw source and target samples from pi  $(x,z) \sim \pi$

    Parameters
    ----------
        pi : numpy array, shape (bs, bs)
            represents the source minibatch
        batch_size : int
            represents the OT plan between minibatches

    Returns
    -------
        (i_s, i_j) : tuple of numpy arrays, shape (bs, bs)
            represents the indices of source and target data samples from $\pi$
    """
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size)
    return np.divmod(choices, pi.shape[1])




class GraphicalOTVelocitySampler:
    """graphical velocity sampler using optimal transport and graphical geodesic (i.e. shorest path in graph)

    Attributes
    ----------
        ts : numpy array, shape (t)
            represents the time series experimental array, record each time point
        dim : int
            input embedding dimensonal number
        X : list[numpy array], shape (t, n_t, dim)
            all input embedding of time series experiment
        index_list : list[numpy array], shape (t, n_t)
            all input index of time series experiment
        n_list : list, shape(t)
            all sample number of time series experiment
    """
    def __str__(self):
        item = {
            "Time Series": self.ts,
            "Date Set Size": self.n_list,
            'Time Key': self.time_key,
            "Graph Key": self.graph_key,
            "Embedding Key": self.embedding_key,
            "Device": self.device,
            "Data Dir": self.data_dir

        }
        return str(item)

    def __init__(
            self, 
            adata: sc.AnnData, 
            time_key : str,
            graph_key : str, 
            embedding_key : str,
            device : torch.device,
            path  : str  = '',
            linear : bool = False,
            neighbor_sampling : bool = False,
            n_neighbors : int = 50,
            dt : float = 0.01
            ) -> None:
        """ init sampler function

        Parameters
        ----------
            adata : scanpy.AnnData
                scRNA-seq experiment data
            time_key : str
                time series experiment recorded key, should in adata.obs.columns
            graph_key : str
                cost key for OT computation, should in adata.obsm
            embedding_key : str
                input embedding key, should in adata.obsm
            device : torch.device
                torch device (cpu or gpu)
            neighbor_sampling : bool
                sample X0 from the neigh of X1
            n_neighbors : int
                the number of neighbors when neighbor_sampling
        
        """
        self.ts = np.sort(np.unique(adata.obs[time_key]))
        self.dim = adata.obsm[embedding_key].shape[1]
        self.linear = linear
        
        self.index_list = [
            adata[adata.obs[time_key] == t].obs.index
            for t in np.sort(pd.unique(adata.obs[time_key]))
        ]
        self.n_list = [len(self.index_list[i]) for i in range(len(self.index_list))]
        self.time_key = time_key
        self.device = device
        self.adata = adata
        self.set_cost(graph_key)
        self.set_embedding(embedding_key)
        self.data_dir = path
        self.load_sp(self.data_dir)
        self.compute_shortest_path(n_neighbors)
        self.dt = dt
        if neighbor_sampling:
            self.neighbor_sampler = []
            for i, t in enumerate(self.ts):
                self.neighbor_sampler.append(NeighborsSampler(n_neighbors, self.X_cost[i]))
        self.neighbor_sampling = neighbor_sampling
        
        self.itps = [Interpolator(self.dist_map[i].astype(np.float64), self.X[i], self.X[i+1], dt) for i in range(len(self.ts)-1)]
    def set_cost(self, graph_key):
        self.X_cost = [
                self.adata.obsm[graph_key][self.adata.obs[self.time_key] == t]
                for t in self.ts
            ]
        self.graph_key = graph_key


    def set_embedding(self, embedding_key, n_components=None):
        if n_components == None:
            self.dim = self.adata.obsm[embedding_key].shape[1]
        else:
            self.dim = n_components
        self.X = [
                self.adata.obsm[embedding_key][self.adata.obs[self.time_key] == t][:, :self.dim].astype(np.float64)
                for t in self.ts
            ]
        self.embedding_key = embedding_key
        

    def compute_shortest_path(
            self, 
            n_neighbors : int =50):
        """ compute shortest path in constructed kNN graph

        Parameters
        ----------
            n_neighbors : int
                number of neighbors in kNN graph

        Return
        ------
            None, restore shorest path map (shape=(t, n_t, n_t+m_t)) as self.sp_map, and shorest path cost matrix ((shape=(t, n_t, n_t+m_t))) as self.dist_map
        """
        if self.sp_map != None and self.dist_map != None:
            return

        sp_map = []
        dist_map = []
        
        for i in range(len(self.ts)-1):
            print('calcu shortest path between {} to {}'.format(self.ts[i], self.ts[i+1]))
            
            # construct graph using data in time i and i+1
            #transition = sc.pp.neighbors(self.adata[np.concatenate([self.index_list[i], self.index_list[i+1]])], 
            #                             n_neighbors=n_neighbors, use_rep=self.graph_key, copy=True)
            ##graph = transition.obsp['distances']
            t0t1X = self.adata[np.concatenate([self.index_list[i], self.index_list[i+1]])].obsm[self.graph_key]
            neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(t0t1X)
            graph = neighbors.kneighbors_graph(t0t1X, mode="distance")
            
            predecessors_list, dist_list = [], []
            
            # compute each shorest path of cell in time i to cell in time {i, i+1}
            for j in tqdm(range(len(self.index_list[i]))):
                dist, predecessors = dijkstra(csgraph=graph, directed=True, indices=j, return_predecessors=True)
                predecessors_list.append(predecessors)
                dist[np.isinf(dist)] = 9999
                dist_list.append(dist)
        
            predecessors_list = np.array(predecessors_list).astype(int)
            dist_list = np.array(dist_list).astype('float32')
            sp_map.append(predecessors_list)
            dist_map.append(dist_list)
        self.sp_map = sp_map
        self.dist_map = dist_map
        # construct interpolator list
        if self.data_dir != '':
            if not os.path.exist(self.data_dir):
                os.makedirs(self.data_dir)
            self.save_sp(self.data_dir)



    def shortest_path_cost(self, 
                           graph ):
        M = dijkstra(csgraph=graph, directed=True, return_predecessors=False)
        return M

    
    '''
    def interpolate_one_point_shortest_path(
            self,
            X : list,
            t_start : int, 
            shortest_path_indices : np.array,
            dt=0.01 ,
            ):
        """ interpolate data point in the selected shortest path

        SP(k) is the k-th cell in the shortest path of cell i to cell j.
        Interpolated data point x_start equals to SP(int(a * len(SP) )),
        the next data point x_end equals to SP(int(a * len(SP)) + 1)
        
        Parameters
        ----------
            X : list
                embedding dataset
            t_start : int
                source cell time
            shortest_path_indices : np.array, shape=(l,)
                record the indices of shorest path of given source cell i to target cell j, e.g. [i, ..., j]
            dt : float
                control the norm of velocity

        Return
        ------
            xa : 
                interpolated data point in the shortest path
            va : 
                velocity of xa
            ti :
                interpolated time
            
        """
        # 需要保证dist_map是欧式距离
        interpolated_features = []
        total_dist = self.dist_map[t_start][shortest_path_indices[0], shortest_path_indices[-1]]
        ori_path_dist = np.array([self.dist_map[t_start][shortest_path_indices[0], shortest_path_indices[i]] for i in range(1, len(shortest_path_indices))])
        path_dist = ori_path_dist / total_dist
        ti = np.random.rand(1)
        s_idx = np.argmin(abs(path_dist - ti)) + 1
        td = path_dist[s_idx - 1]
        remain_t = ti - td
        
        if remain_t > 0:
            start_feature =  X[t_start + 1][shortest_path_indices[s_idx]-len(X[t_start])] if shortest_path_indices[s_idx] >= len(X[t_start]) else X[t_start][shortest_path_indices[s_idx]]
            end_feature = X[t_start + 1][shortest_path_indices[s_idx+1]-len(X[t_start])] if shortest_path_indices[s_idx+1] >= len(X[t_start]) else X[t_start][shortest_path_indices[s_idx+1]]
            va = (end_feature - start_feature)
            norm_va = ori_path_dist[s_idx] - max(ori_path_dist[s_idx - 1], 0) if s_idx > 0 else ori_path_dist[s_idx]
            xa =  start_feature +  (total_dist / norm_va) *  remain_t * va
                
        else:
            start_feature =  X[t_start + 1][shortest_path_indices[s_idx-1]-len(X[t_start])] if shortest_path_indices[s_idx-1] >= len(X[t_start]) else X[t_start][shortest_path_indices[s_idx-1]]
            end_feature = X[t_start + 1][shortest_path_indices[s_idx]-len(X[t_start])] if shortest_path_indices[s_idx] >= len(X[t_start]) else X[t_start][shortest_path_indices[s_idx]]
            va = (end_feature - start_feature)
            norm_va = ori_path_dist[s_idx - 1] - ori_path_dist[s_idx - 2] if s_idx -1 > 0 else ori_path_dist[s_idx - 1]
            xa = end_feature + (total_dist / norm_va) * remain_t * va
            
        va = ((total_dist / norm_va) * dt) * va
        
        return xa, va, ti

    '''
    def save_sp(self, path):
        for i in range(len(self.ts) - 1):
            save_path = path + '_' + str(self.ts[i]) + 'to' + str(self.ts[i+1]) 
            np.save(save_path + '_map.npy', self.sp_map[i])
            np.save(save_path + '_dist.npy', self.dist_map[i])
    
    
    def load_sp(self, path):
        self.sp_map = []
        self.dist_map = []
        print("loading saved shortest path profile")
        for i in range(len(self.ts) - 1):
            save_path = path + '/_' + str(self.ts[i]) + 'to' + str(self.ts[i+1]) 
            try:
                self.sp_map.append(np.load(save_path+ '_map.npy'))
                self.dist_map.append(np.load(save_path+ '_dist.npy'))
            except:
                self.sp_map = None
                self.dist_map = None
                print('Error in loading shortest path file')
                break    

    def get_path(
            self, 
            t : int, 
            target : int, 
            source : int):
        
        """ Get shorest path indices for given source cell and target cell
        
        Parameters
        ----------
            t : int
                source cell time
            target : int
                target cell indices
            source : int
                source cell indices

        Return
        ------
            path : list
                shortest path indices 
            flag : int 
                1 if exist path 0 if not
            
        """
        # get shorest path map
        path_map = self.sp_map[t][source]
        
        path = [target]
        next_node = target
        # search map to get wanted path
        while (True):
            if path_map[next_node] < 0:
                break
            path.append(path_map[next_node])
            
            next_node = path_map[next_node]
        path = np.array(path[::-1])
        if source == path[0]:
            
            return path, 1
        else:
            return path, 0
    

    def sample_pair(
            self,
            source_idx : np.array, 
            target_idx : np.array, 
            t_start : int,
            distance_matrices : str,
            t_end : int = None,
            outlier_filter: bool = True):
        """ sample source-target pair using OT
        
        Parameters
        ----------
            source_idx : np.array, shape=(batch_size, )
                source cell indices
            target_idx : np.array, shape=(batch_size, )
                target cell indices
            t_start : int
                source time 
            t_end : int
                target time 
            distance_matrices: str
                using 'SP' (i.e. shortest path distance) or 'L2'

        Return
        ------
            x0 : torch.Tensor, shape=(batch_size, dim)
                paired source cell
            x1 : torch.Tensor, shape=(batch_size, dim)
                paired source cell
            i : torch.Tensor, shape=(batch_size, )
                paired source cell indices
            j_map : torch.Tensor, shape=(batch_size, )
                paired target cell indices
        """
        if t_end == None:
            t_end = t_start + 1
        x0_c = self.X_cost[t_start][source_idx]
        x1_c = self.X_cost[t_end][target_idx]
            
        x0_c = (
                torch.from_numpy(x0_c)
                .float()
            )
        x1_c = (
                torch.from_numpy(
                    x1_c
                )
                .float()
            )
        # calcu the optimal transport plan and get \pi(x0, x1) as latent distribution
        if distance_matrices != 'SP':
            # L2 distance as optimal transport cost matrix
            M = torch.cdist(x0_c, x1_c)
        else:
            # Shortest path distance as optimal transport cost matrix
            M = self.dist_map[t_start][source_idx,:]
            M = M[:, self.n_list[t_start] + target_idx]
            
            
        M = torch.Tensor(M)
        pi = solve_ot(M)
            
        # re-sampling by OT plan
        i, j = sample_map(pi, x0_c.shape[0])
        i = source_idx[i]
        j = target_idx[j]
                
        x0_c = self.X_cost[t_start][i]
        x1_c = self.X_cost[t_end][j]
        if outlier_filter:
            # exclude the far pair
            if distance_matrices != 'SP':
                dist = np.sum((x0_c - x1_c)**2, axis=-1)
            else:
                dist = self.dist_map[t_start][i,:]
                dist = dist[range(len(dist)), self.n_list[t_start] + j]
            
            mean = np.mean(dist)
            std = np.std(dist)
            idx = np.where(dist < mean + 2*std)[0]

            i, j = i[idx], j[idx]
        
        x0 = self.X[t_start][i]
        x1 = self.X[t_end][j]

        # for shortest path indices adjustment
        j_map = j + self.n_list[t_start]

        return x0, x1, i, j_map

    
    def _sample_one_time_point(
            self, 
            t_start, 
            batch_size=256, 
            interpolate_num=100,
            distance_matrices='SP',
            ):        
        interpolate_num += 1
        if not self.neighbor_sampling or t_start == 0:
            source_idx = np.random.choice(self.n_list[t_start], size=batch_size)
            target_idx = np.random.choice(self.n_list[t_start + 1], size=batch_size)
        else:
            target_idx = np.random.choice(self.n_list[t_start + 1], size=batch_size)
            source_idx = self.neighbor_sampler[t_start].sample(self.X_cost[t_start + 1][target_idx])

        
        #start = time.perf_counter() #Python3.8不支持clock了，使用timer.perf_counter()
        
        x0, x1, i, j_map = self.sample_pair(source_idx, target_idx, t_start, distance_matrices)
        #end = time.perf_counter()
        
        #print('OT: %s Seconds'%(end-start))

        #start = time.perf_counter() #Python3.8不支持clock了，使用timer.perf_counter()
        '''
        paths = []
        linear_idxs = []
        for idx in range(len(x0)):
            path, flag = self.get_path(t_start, target=j_map[idx], source=i[idx])
            if flag == 0 or len(path) < 2:
                if not self.linear:
                    continue
                else:
                    linear_idxs.append(idx)        
            paths.append(path)
        
        t_sp = np.random.rand(len(paths))
        xa_t_sp, ua_t_sp = self.itps[t_start].mini_batch_interpolate(paths, t_sp)
            
        if len(linear_idxs) != 0:
            t_linear = np.random.rand(len(linear_idxs))
            xa_t_linear = t_linear*x0[linear_idxs] + (1-t_linear) * x1[linear_idxs]
            ua_t_linear = x1[linear_idxs] - x0[linear_idxs]
            xa_t = np.concatenate([xa_t_sp, xa_t_linear],axis=-1)
            ua_t = np.concatenate([ua_t_sp, ua_t_linear],axis=-1)
            ts = np.concatenate([t_sp, t_linear],axis=-1)
        else:
            xa_t, ua_t, ts = xa_t_sp, ua_t_sp, t_sp
                        
        '''
        xa_t, ua_t = [], []
        ts = []
        for idx in range(len(x0)):
            source = i[idx]
            target = j_map[idx]
            
            path, flag = self.get_path(t_start, target=target, source=source)
            if flag == 0 or len(path) < 2:
                if not self.linear:
                    continue             
                t = random.random()
                xa_t.append(t*x0[idx] + (1-t) * x1[idx])
                ua_t.append(x1[idx] - x0[idx])
                ts.append(t)
            
            else:
                t = random.random()
                xa, ua = self.itps[t_start].interpolate_one_point_shortest_path(path, ti=t)
                xa_t.append(xa)
                ua_t.append(ua)
                ts.append(t)
            
            
        #end = time.perf_counter()
        
        #print('interpolate: %s Seconds'%(end-start))    
        
        return np.array(xa_t), np.array(ua_t), np.array(ts)[:,None], x0, x1

    

    def sample_batch_path(
            self,
            sigma : float , 
            batch_size : int,
            distance_matrices : str = 'L2', 
            add_noise : bool = True,
            ):
        """ sample data point x_t and corresponding velocity u_t using OT and SP
        
        Parameters
        ----------
            sigma : float, belongs to [0,1]
                noise level of interpolated data point
            batch_size : int
            distance_matrices: str
                using 'SP' (i.e. shortest path distance) or 'L2'
            add_noise : bool
                xt add noise or not
            interpolate : bool
                linear interpolate between two node in the shortest path or not

        Return
        ------
            T : torch.Tensor, shape=(batch_size * t_max, )
                correponding t to x
            X : torch.Tensor, shape=(batch_size * t_max, dim)
                sampled interpolated data points
            U : torch.Tensor, shape=(batch_size * t_max, dim)
                corresponding velocity to x

        """
        X, U, T = [], [], []
        X0, X1 = [], []
        error_num = 10
        for t_start in range(len(self.ts) - 1):
            t_diff = (self.ts[t_start + 1] - self.ts[t_start])

            xa_t, ua_t, t, x0, x1 = self._sample_one_time_point(t_start, batch_size=batch_size, distance_matrices=distance_matrices,)
                
            if len(xa_t) == 0:
                raise Exception('low connection of graph, please increase `n_neighbors` or set `linear` into `True` ')
            X0.append(x0)
            X1.append(x1)
            
            #t = self.ts[t_start] + t
            t = t_start + t

            
            T.append(t)
            X.append(xa_t)
            U.append(ua_t)
            
        
        X, U, T = np.concatenate(X), np.concatenate(U)/self.dt, np.concatenate(T)
        X, U, T =  X.reshape(-1, X.shape[-1]), U.reshape(-1, U.shape[-1]), T.reshape(-1, T.shape[-1])
        if add_noise:
                X = X + sigma*np.random.randn(X.shape[0], X.shape[1])
        return T, X, U, X0, X1
    

    def filtered_sample_batch_path(
            self,
            sigma : float , 
            batch_size : int,
            distance_matrices : str = 'L2', 
            add_noise : bool = True,
            k=15,
            q=20,
           
            ):
        """ sample data point x_t and corresponding velocity u_t using OT and SP, filter outlier using gaussian dist with knn center
        
        Parameters
        ----------
            sigma : float, belongs to [0,1]
                noise level of interpolated data point
            batch_size : int
            distance_matrices: str
                using 'SP' (i.e. shortest path distance) or 'L2'
            add_noise : bool
                xt add noise or not
            interpolate : bool
                linear interpolate between two node in the shortest path or not
            k : int
                knn kernel neighbors number
            q : int
                cutoff, filter in q % data points.

        Return
        ------
            T : torch.Tensor, shape=(batch_size * t_max, )
                correponding t to x
            X : torch.Tensor, shape=(batch_size * t_max, dim)
                sampled interpolated data points
            U : torch.Tensor, shape=(batch_size * t_max, dim)
                corresponding velocity to x

        """
        T, X, U, X0, X1 = self.sample_batch_path(sigma, batch_size, distance_matrices, add_noise)

        filtered_idx = filter_outlier(torch.Tensor(X), torch.tensor(U), k=k, q=q)
        T = torch.Tensor(T[filtered_idx])
        X = torch.Tensor(X[filtered_idx])
        U = torch.Tensor(U[filtered_idx])
        
        return T, X, U, X0, X1

    


    
def outlier_mean_field_likelihood(x, mu, std=None):
    d = x.shape[1]
    if std == None or std == 1:
        l = torch.exp(torch.sum((-0.5)*(x - mu) ** 2, dim=1) + d*np.log(1/np.sqrt(2*torch.pi)))
    else:
        l = torch.exp(torch.sum((-0.5*std**2)*(x - mu) ** 2, dim=1) + d*np.log(1/np.sqrt(2*torch.pi) * std))
    return l


def filter_outlier(xt, ut, k=15, q=50):
    
    knn = KNeighborsRegressor(k)
    ut_norm = ut / np.linalg.norm(ut, axis=1)[:,np.newaxis]
    knn.fit(xt, ut_norm)
    ut_knn = knn.predict(xt)
    in_l = outlier_mean_field_likelihood(ut_norm, ut_knn)
    c = np.percentile(in_l, q=q)
    return np.where(in_l > c)



def v_centric_training(
        model : torch.nn.Module, 
        optimizer : torch.optim.Optimizer, 
        sample_fun, 
        iter_n=10000,
        device=torch.device('cpu')
        ):
    """Fit a neural network given sampling velocity function

    Args:
        model: Neural network model to fit vector field (e.g. MLP)
        optimizer: Optimizer for optimize parameter of model
        sample_fun: Sampling velocity function

    Returns:
        Trained neural network

    """
    model.to(device)
    history = []
    pbar = tqdm(range(iter_n))
    best_loss = np.inf
    losses = []
    for i in pbar:
        optimizer.zero_grad()
    
        t, xt, ut, _, _ = sample_fun()
        vt = model(t.to(device), xt.to(device))
        
        loss = torch.mean((vt - ut.to(device)) ** 2)
        #G = model.compute_G(t, xt)
        #l1_loss =  torch.norm(G, p=1)
        #loss += 0.001 * l1_loss
        #loss += torch.mean((vt-(G @ xt[:,:, None]).squeeze(-1))**2)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        history.append(loss.item())
        if i % 100 == 0:
            
            losses = np.mean(losses)
            best_loss = np.min([losses, best_loss])
            pbar.set_description('loss :{:.4f} best :{:.4f}'.format(losses, best_loss))
            losses = []
    return model, np.array(history)

        
