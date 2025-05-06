from scipy.sparse.csgraph import dijkstra
import ot as pot
import numpy as np
import scanpy as sc
import torch
from tqdm import tqdm
import pandas as pd
from .utils import NeighborsSampler
from sklearn.neighbors import KNeighborsRegressor


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
            n_neighbors : int = 50
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
        if neighbor_sampling:
            self.neighbor_sampler = []
            for i, t in enumerate(self.ts):
                self.neighbor_sampler.append(NeighborsSampler(n_neighbors, self.X_cost[i]))
        self.neighbor_sampling = neighbor_sampling

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
                self.adata.obsm[embedding_key][self.adata.obs[self.time_key] == t][:, :self.dim]
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
            transition = sc.pp.neighbors(self.adata[np.concatenate([self.index_list[i], self.index_list[i+1]])], 
                                         n_neighbors=n_neighbors, use_rep=self.graph_key, copy=True)
            graph = transition.obsp['distances']
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
        if self.data_dir != '':
            self.save_sp(self.data_dir)



    def shortest_path_cost(self, 
                           graph ):
        M = dijkstra(csgraph=graph, directed=True, return_predecessors=False)
        return M

    def interpolate_all_shortest_path(
            self,
            X : list,
            t_start : int, 
            shortest_path_indices : np.array,
            total_interpolations : int):
        # 插值生成特征向量
        interpolated_features = []
        total_dist = self.dist_map[t_start][shortest_path_indices[0], shortest_path_indices[-1]]
        remaining_interpolations = total_interpolations
    
        for i in range(len(shortest_path_indices) - 1):
            start_feature =  get_correct_x(X[t_start], X[t_start+1], shortest_path_indices[i])
            end_feature = get_correct_x(X[t_start], X[t_start+1], shortest_path_indices[i+1])
    
            distance = np.linalg.norm(end_feature - start_feature)
            # 计算在该段路径上插值的点数
            interpolation_count = round(total_interpolations * (distance / total_dist))
            if i == len(shortest_path_indices) - 2:
                interpolation_count = remaining_interpolations
            # 线性插值
            if interpolation_count > 0:
                interpolated_vectors = np.linspace(start_feature, end_feature, num=interpolation_count + 1)[1:]
                interpolated_features.append(interpolated_vectors)
            
            remaining_interpolations -= interpolation_count
        return np.concatenate(interpolated_features)[:total_interpolations]



    def save_sp(self, path):
        for i in range(len(self.ts) - 1):
            save_path = path + '_' + str(self.ts[i]) + 'to' + str(self.ts[i+1]) 
            np.save(save_path + '_map.npy', self.sp_map[i])
            np.save(save_path + '_dist.npy', self.dist_map[i])
    
    
    def load_sp(self, path):
        self.sp_map = []
        self.dist_map = []
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


    def interpolate_along_shortest_path(
            self, 
            t_start : int,
            shortest_path_indices : np.array,
            a : float,
            interpolate : bool = False,
            ):
        """ interpolate data point in the selected shortest path

        SP(k) is the k-th cell in the shortest path of cell i to cell j.
        Interpolated data point x_start equals to SP(int(a * len(SP) )),
        the next data point x_end equals to SP(int(a * len(SP)) + 1)
        
        Parameters
        ----------
            t_start : int
                source cell time
            shortest_path_indices : np.array, shape=(l,)
                record the indices of shorest path of given source cell i to target cell j, e.g. [i, ..., j]
            a : float
                interpolate ratio
            interpolate : bool
                interpolate between two node in the path or not

        Return
        ------
            x_start : np.array, shape=(dim, )
                interpolated data point in the shortest path
            x_end : np.array, shape=(dim, ) 
                next data point of interpolated data point
            
        """
        # 确定插值点在最短路径上的位置
        
        num_interpolated_points = len(shortest_path_indices) - 1
        interpolated_index = int(a * num_interpolated_points)

        start_node_index = shortest_path_indices[interpolated_index]
        x_start = get_correct_x(self.X[t_start], self.X[t_start+1], start_node_index)
        x_cost_start = get_correct_x(self.X_cost[t_start], self.X_cost[t_start+1], start_node_index)
        
        end_node_index = shortest_path_indices[interpolated_index + 1]
        x_end = get_correct_x(self.X[t_start], self.X[t_start+1], end_node_index)
        x_cost_end = get_correct_x(self.X_cost[t_start], self.X_cost[t_start+1], end_node_index)
        
        #dt = np.sum((x_cost_end - x_cost_start) ** 2) / self.dist_map[t_start][shortest_path_indices[0], shortest_path_indices[-1]]
        #print(np.sum((x_cost_end - x_cost_start) ** 2), self.dist_map[t_start][shortest_path_indices[0], shortest_path_indices[-1]], dt)
        # interpolate between two node
        if interpolate:
            a2 = (a * num_interpolated_points) % 1
            xt =  x_start + (x_end - x_start) * (a2)
            xt_cost =  x_cost_start + (x_cost_end - x_cost_start) * (a2)
            return x_start, x_end, xt, x_cost_start, x_cost_end, xt_cost

        return x_start, x_end, x_cost_start, x_cost_end
    

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
            distance_matrices='L2', 
            
            ):
        
        interpolate_num += 1
        if not self.neighbor_sampling or t_start == 0:
            source_idx = np.random.choice(self.n_list[t_start], size=batch_size)
            target_idx = np.random.choice(self.n_list[t_start + 1], size=batch_size)
        else:
            target_idx = np.random.choice(self.n_list[t_start + 1], size=batch_size)
            source_idx = self.neighbor_sampler[t_start].sample(self.X_cost[t_start + 1][target_idx])

        xa_t, ua_t = [], []
        x0, x1, i, j_map = self.sample_pair(source_idx, target_idx, t_start, distance_matrices)
        
        for idx in range(len(x0)):
            source = i[idx]
            target = j_map[idx]
            
            path, flag = self.get_path(t_start, target=target, source=source)
            if flag == 0 or len(path) < 2:
                if not self.linear:
                    continue
                va = np.linspace(x0[idx], 
                            x1[idx],
                              num=interpolate_num)               
            
            else:
                va = self.interpolate_all_shortest_path(self.X, t_start, path, total_interpolations=interpolate_num)
                
            
            xa_t.append(va[:-1])
            ua_t.append(va[1:] - va[:-1])
            
        return np.array(xa_t), np.array(ua_t), x0, x1
    

    def sample_batch_path(
            self,
            sigma : float , 
            dt : float,
            batch_size : int,
            distance_matrices : str = 'L2', 
            all_in : bool = False,
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
            #t_diff = 1
            '''
            try:
                xa_t, ua_t, x0, x1 = self._sample_one_time_point(t_start, 
                                                    batch_size=batch_size, 
                                                    interpolate_num=int(t_diff/dt),
                                                    distance_matrices=distance_matrices,
                                                    )
                print(xa_t, ua_t, xa_t.shape, ua_t.shape)
            except Exception as e:
                print(e)
                error_num -= 1
                if error_num == 0:
                    print('More than ten time mini-batch ot errors, try to reduce batch size.')
                    raise RuntimeError
                continue
            '''
            xa_t, ua_t, x0, x1 = self._sample_one_time_point(t_start, 
                                                    batch_size=batch_size, 
                                                    interpolate_num=int(t_diff/dt),
                                                    distance_matrices=distance_matrices,
                                                    )
            if len(xa_t) == 0:
                raise Exception('low connection of graph, please increase `n_neighbors` or set `linear` into `True` ')
            X0.append(x0)
            X1.append(x1)
            if not all_in:
                
                t = np.random.rand(xa_t.shape[0])[:,None] * t_diff
                idx = (np.random.rand(xa_t.shape[0]) * int(t_diff/dt)).astype(int)
                xa_t = xa_t[np.arange(xa_t.shape[0]),idx,:]
                ua_t = ua_t[np.arange(xa_t.shape[0]),idx,:]
            else:
                t = np.repeat(np.linspace(0, t_diff, int(t_diff/dt) + 1)[1:,None], xa_t.shape[0], axis=-1).T[:,:,None]
            
            

            #t = self.ts[t_start] + t
            t = t_start + t
            T.append(t)
            X.append(xa_t)
            U.append(ua_t)
            
        
        X, U, T = np.concatenate(X), np.concatenate(U)/dt, np.concatenate(T)
        X, U, T =  X.reshape(-1, X.shape[-1]), U.reshape(-1, U.shape[-1]), T.reshape(-1, T.shape[-1])
        if add_noise:
                X = X + sigma*np.random.randn(X.shape[0], X.shape[1])
        
        return T, X, U, X0, X1
    

    def filtered_sample_batch_path(
            self,
            sigma : float , 
            dt : float,
            batch_size : int,
            distance_matrices : str = 'L2', 
            all_in : bool = False,
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
        T, X, U, X0, X1 = self.sample_batch_path(sigma, dt, batch_size, distance_matrices, all_in, add_noise)
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

        
