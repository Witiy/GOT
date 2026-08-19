from tqdm import tqdm
from scipy.sparse.csgraph import dijkstra            
import torch
import ot as pot
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
import os
import random
import pickle




class Landmarks:
    def __init__(self, ):    
        pass
    def fit(self, X, n_neighbors=30, n_landmarks=5000,  verbose=True,):

        try:
            import faiss

        except ImportError:
            raise ImportError(
                    "Please install FAISS to use landmarks")
        
        
        print('Fit {} landmarks by FAISS KMean'.format(n_landmarks))
        kmeans = faiss.Kmeans(X.shape[1], n_landmarks, niter=1000, verbose=verbose, gpu=torch.cuda.is_available(),)
        kmeans.train(X)
        _, assignments = kmeans.assign(X)
        self.assignments = assignments
        self.centroid = kmeans.centroids
            
        # remove no assigned centroid
        record_centroid = np.unique(self.assignments)
        self.centroid = self.centroid[record_centroid]
        mapping = dict(zip(record_centroid, range(len(self.centroid))))
        self.assignments = np.array([mapping[c] for c in self.assignments])
        
        self.inverse_assignments = {i:np.where(self.assignments == i)[0] for i in range(n_landmarks)}
        self.dijkstra(n_neighbors,)
        self.cell2centoids = np.linalg.norm(X - self.centroid[self.assignments], axis=1)

    def dijkstra(self, n_neighbors=30,):
        # compute each shorest path of landmark in time i to cell in time {i, i+1}
        self.centroid_predecessors_list, self.centroid_dist_list = compute_dijkstra(
            self.centroid, min(n_neighbors, self.centroid.shape[0]), len(self.centroid), )
        
    def compute_graphical_distance(self, source_idx, end_idx):
        return self.cell2centoids[source_idx] \
            + self.centroid_dist_list[self.assignments[source_idx]][:,self.assignments[end_idx]] + \
            self.cell2centoids[end_idx]
        
    def project_path(self, source_idx, end_idx):
        if self.assignments[source_idx] == self.assignments[end_idx]:
            return [source_idx, end_idx], 1
        land_path, flag = _get_path(self.centroid_predecessors_list[self.assignments[source_idx]], 
                                    target=self.assignments[end_idx], source=self.assignments[source_idx])
        cell_path = [source_idx] + [np.random.choice(self.inverse_assignments[l]) for l in land_path[1:-1]] + [end_idx]
        return cell_path, flag


class GraphPath:
    def __init__(self, X, n_neighbors=30, n_landmarks=5000, n_dijk=None, landmarks=False, X_cost=None):
        self.landmarks = landmarks
        if X_cost is None:
            self.X_cost = X
        else:
            self.X_cost = X_cost
        if self.landmarks:
            self.lands = Landmarks()
            self.lands.fit(self.X_cost, n_neighbors, n_landmarks, )
        else:
            if n_dijk is None:
                n_dijk = len(self.X_cost)
            self.predecessors_list, self.dist_list = compute_dijkstra(self.X_cost, n_neighbors, n_dijk,)
        
        self.X = X
    def graphical_distance(self, source_idx, end_idx):
        
        if self.landmarks:
            return self.lands.compute_graphical_distance(source_idx, end_idx)
        else:
            return self.dist_list[source_idx,:][:,end_idx]

    def graphical_path(self, source_idx, end_idx):
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
        
        if self.landmarks:
            return self.lands.project_path(source_idx, end_idx)
        else:
            path_map = self.predecessors_list[source_idx]
            path, flag = _get_path(path_map, source=source_idx, target=end_idx)
            return path, flag

    def interpolate_one_point(self, path, ti):
        if len(path) == 2:
            va = (self.X[path[1]] - self.X[path[0]])
            xa = ti * self.X[path[1]] + (1-ti) * self.X[path[0]]
            return xa, va
        
        if self.landmarks:
            centroid_dist = self.lands.centroid_dist_list[self.lands.assignments[path[0]], self.lands.assignments[path[1:-1]]]
            ori_path_dist = np.array([self.lands.cell2centoids[path[0]]] + 
                list(centroid_dist) + [centroid_dist[-1] + self.lands.cell2centoids[path[-1]]])
        else:
            ori_path_dist = np.array([self.dist_list[path[0], path[i]] for i in range(0, len(path))])
            #ori_path_dist = self.dist_list[path[0], path[1:]]
            
        total_dist = ori_path_dist[len(ori_path_dist) - 1]
        path_dist = ori_path_dist / total_dist
        s_idx = np.argmin(np.abs(path_dist - ti))
        td = path_dist[s_idx]
        remain_t = ti - td
        if remain_t > 0:
            start_feature = self.X[path[s_idx]]
            end_feature = self.X[path[s_idx + 1]]
        else:
            start_feature = self.X[path[s_idx - 1]]
            end_feature = self.X[path[s_idx]]
        
        va = end_feature - start_feature
        norm_va = np.linalg.norm(va)
        
        va = (total_dist / norm_va) * va
        
        xa = start_feature + remain_t * va if remain_t > 0 else end_feature + remain_t * va
        
        if np.isnan(xa.sum()) or np.isnan(va.sum()):
            print('xa: {}, va: {}'.format(xa, va))
            print('path: {}'.format(path))
            print('remain_t: {}'.format(remain_t))
            print('start_feature: {}'.format(start_feature))
            print('end_feature: {}'.format(end_feature))
            print('va: {}'.format(va))
            print('norm_va: {}'.format(norm_va))
        return xa, va  


def _get_path(
            path_map,
            target : int, 
            source : int):
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
            return path, True
        else:
            return path, False
        


def compute_dijkstra(t0t1X, n_neighbors, n_source):
    neighbors = NearestNeighbors(n_neighbors=min(n_neighbors,t0t1X.shape[0]), metric="euclidean").fit(t0t1X)
    graph = neighbors.kneighbors_graph(t0t1X, mode="distance")
            
    predecessors_list, dist_list = [], []
            
    # compute each shorest path of cell in time i to cell in time {i, i+1}
    for j in tqdm(range(n_source)):
        dist, predecessors = dijkstra(csgraph=graph, directed=True, indices=j, return_predecessors=True)
        predecessors_list.append(predecessors)
        dist[np.isinf(dist)] = 9999
        dist_list.append(dist)
        
    predecessors_list = np.array(predecessors_list).astype(int)
    dist_list = np.array(dist_list).astype('float32')
    return predecessors_list, dist_list
    





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
            "Data Dir": self.data_dir,
            "KNN Constraint": self.knn_constraint,
            "Full Connect": self.full_connect,

        }
        return str(item)

    def __init__(
            self, 
            adata: sc.AnnData, 
            time_key : str,
            graph_key : str, 
            embedding_key : str,
            device : torch.device,
            landmarks : bool = False,
            path  : str  = '',
            linear : bool = False,
            n_neighbors : int = 50,
            p : int = 2,
            landmarks_fold : int = 5,
            knn_constraint : bool = True,
            full_connect : bool = False,
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
            landmarks: bool
                use landmarks to appoximate graphical path
            n_neighbors : int
                the number of neighbors
            knn_constraint : bool
                use graph paths for interpolation and enable kNN velocity
                filtering. OT cost is selected independently by
                ``distance_metrics`` when sampling
            full_connect : bool
                construct one kNN graph from all time points instead of one
                graph for each adjacent time-point pair
        
        """
        assert (p == 1) or (p == 2)
        self.ts = np.sort(np.unique(adata.obs[time_key]))
        self.dim = adata.obsm[embedding_key].shape[1]
        self.linear = linear
        
        self.index_list = [
            adata[adata.obs[time_key] == t].obs.index
            for t in np.sort(pd.unique(adata.obs[time_key]))
        ]
        self.n_list = [len(self.index_list[i]) for i in range(len(self.index_list))]
        self.offsets = np.concatenate([[0], np.cumsum(self.n_list)]).astype(int)
        self.landmarks = landmarks
        self.time_key = time_key
        self.device = device
        self.adata = adata
        self.set_cost(graph_key)
        self.set_embedding(embedding_key)
        self.data_dir = path
        self.knn_constraint = knn_constraint
        self.full_connect = full_connect
        self.n_neighbors = n_neighbors
        self.landmarks_fold = landmarks_fold
        self.GPs = None
        if self.knn_constraint:
            self._ensure_graph_paths()
        self.p = p
        
        
        
       
    def set_cost(self, graph_key):
        self.X_cost = [
                self.adata.obsm[graph_key][self.adata.obs[self.time_key] == t]
                for t in self.ts
            ]
        self.X_cost_all = np.concatenate(self.X_cost)
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
        self.X_all = np.concatenate(self.X)
        self.embedding_key = embedding_key
        

    def compute_shortest_path(
            self, 
            n_neighbors : int =50,
            p : int = 2,
            landmarks_fold : int = 5,
    ):
        """ compute shortest path in constructed kNN graph

        Parameters
        ----------
            n_neighbors : int
                number of neighbors in kNN graph

        Return
        ------
            None, restore shorest path map (shape=(t, n_t, n_t+m_t)) as self.sp_map, and shorest path cost matrix ((shape=(t, n_t, n_t+m_t))) as self.dist_map
        """
        if self.GPs is not None:
            return

        
        self.GPs = []
        if self.full_connect:
            X = self.X_all
            X_cost = self.X_cost_all if self.graph_key != self.embedding_key else None
            if self.landmarks and len(X) > 5000:
                landmarks = True
                n_landmarks = max(min(25000, len(X) // landmarks_fold), 2000)
                print('Number of landmarks : {} for full graph'.format(n_landmarks))
            else:
                landmarks = False
                n_landmarks = 0
            print('calcu shortest path on all time points')
            self.GPs.append(
                GraphPath(
                    X,
                    landmarks=landmarks,
                    n_neighbors=n_neighbors,
                    n_landmarks=n_landmarks,
                    n_dijk=int(self.offsets[-2]),
                    X_cost=X_cost,
                )
            )
        else:
            self._compute_pairwise_shortest_paths(
                n_neighbors,
                landmarks_fold=landmarks_fold,
            )

        # construct interpolator list
        if self.data_dir != '':
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            self.save_gp(self.data_dir)

    def _compute_pairwise_shortest_paths(self, n_neighbors, landmarks_fold=5):
        for i in range(len(self.ts)-1):
            print('calcu shortest path between {} to {}'.format(self.ts[i], self.ts[i+1]))
            if self.graph_key != self.embedding_key:
                X_cost = self.adata[np.concatenate([self.index_list[i], self.index_list[i+1]])].obsm[self.graph_key] 
            else:
                X_cost = None
            X = self.adata[np.concatenate([self.index_list[i], self.index_list[i+1]])].obsm[self.embedding_key]
            
            if self.landmarks and len(X) > 5000:
                landmarks = True
                n_landmarks = max(min(25000, len(X) // landmarks_fold), 2000)
                print('Number of landmarks : {} between {} to {}'.format(n_landmarks, self.ts[i], self.ts[i+1]))
            else:
                landmarks = False
                n_landmarks = 0

            gp = GraphPath(X,
                      landmarks=landmarks,
                      n_neighbors=n_neighbors, 
                      n_landmarks=n_landmarks,
                      n_dijk=len(self.index_list[i]),
                      X_cost=X_cost
                    )
            self.GPs.append(gp)

    def _ensure_graph_paths(self):
        """Load or compute graph paths only when graph-based behavior is used."""
        if self.GPs is not None:
            return
        if self.data_dir != '':
            self.load_gp(self.data_dir)
        if self.GPs is None:
            self.compute_shortest_path(
                self.n_neighbors,
                landmarks_fold=self.landmarks_fold,
            )

    
    def save_gp(self, path):
        if self.full_connect:
            save_path = os.path.join(path, '_global.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(self.GPs[0], f)
            return
        for i in range(len(self.ts) - 1):
            save_path = os.path.join(
                path,
                '_' + str(self.ts[i]) + 'to' + str(self.ts[i+1]) + '.pkl',
            )
            with open(save_path, 'wb') as f:
                pickle.dump(self.GPs[i], f)
            
    
    
    def load_gp(self, path):
        self.GPs = []
        print("loading saved shortest path profile")
        if self.full_connect:
            save_path = os.path.join(path, '_global.pkl')
            try:
                with open(save_path, 'rb') as f:
                    self.GPs.append(pickle.load(f))
            except Exception as e:
                self.GPs = None
                print(e)
                print('Error in loading full shortest path file')
            return
        for i in range(len(self.ts) - 1):
            save_path = os.path.join(path, '_' + str(self.ts[i]) + 'to' + str(self.ts[i+1]) + '.pkl' )
            try:
                with open(save_path, 'rb') as f:
                    self.GPs.append(pickle.load(f))
            except Exception as e:
                self.GPs = None
                print(e)
                print('Error in loading shortest path file')
                break    

    def _graph(self, t_start):
        return self.GPs[0] if self.full_connect else self.GPs[t_start]

    def _graph_indices(self, t_start, source_idx, target_idx):
        """Map adjacent-time local indices into the active graph coordinates."""
        source_idx = np.asarray(source_idx)
        target_idx = np.asarray(target_idx)
        if self.full_connect:
            return (
                source_idx + self.offsets[t_start],
                target_idx + self.offsets[t_start + 1],
            )
        return source_idx, target_idx + self.n_list[t_start]

    def path_coordinates(self, t_start, path):
        """Return embedding coordinates for graph-node indices in ``path``."""
        path = np.asarray(path, dtype=int)
        if self.full_connect:
            return self.X_all[path]
        return np.concatenate([self.X[t_start], self.X[t_start + 1]])[path]
        



    def sample_pair(
            self,
            source_idx : np.array, 
            target_idx : np.array, 
            t_start : int,
            distance_metrics : str,
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
            distance_metrics: str
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
        if distance_metrics == 'SP':
            # Shortest path distance as optimal transport cost matrix
            self._ensure_graph_paths()
            source_graph, target_graph = self._graph_indices(
                t_start,
                source_idx,
                target_idx,
            )
            M = self._graph(t_start).graphical_distance(source_graph, target_graph)
        else:
            # L2 distance as optimal transport cost matrix
            M = torch.cdist(x0_c, x1_c)
        
        if self.p == 2:
            M = M ** 2    
        M = torch.Tensor(M)
        pi = solve_ot(M)
        
        # re-sampling by OT plan
        num_i, num_j = sample_map(pi, x0_c.shape[0])
        
        i = source_idx[num_i]
        j = target_idx[num_j]
                

        if outlier_filter:
            # exclude the far pair
            dist = M[num_i, num_j].cpu().numpy()
        
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
            distance_metrics='SP',
            n_samples_in_path=1,
            return_metadata=False,
            ):
        if not isinstance(n_samples_in_path, (int, np.integer)) or n_samples_in_path < 1:
            raise ValueError('`n_samples_in_path` must be a positive integer')

        interpolate_num += 1
        
        source_idx = np.random.choice(self.n_list[t_start], size=batch_size)
        target_idx = np.random.choice(self.n_list[t_start + 1], size=batch_size)
        
        x0, x1, i, j_map = self.sample_pair(source_idx, target_idx, t_start, distance_metrics)
        
        xa_t, ua_t = [], []
        ts = []
        sample_pair_id = []
        sample_connected = []
        sample_linear_fallback = []
        sample_path_length = []
        sample_chord_length = []
        pair_paths = []
        pair_connected = []
        pair_linear_fallback = []
        pair_path_length = []
        pair_chord_length = []

        for idx in range(len(x0)):
            source = int(i[idx])
            target = int(j_map[idx])
            target_local = target - self.n_list[t_start]
            source_graph, target_graph = self._graph_indices(
                t_start,
                source,
                target_local,
            )
            source_graph = int(source_graph)
            target_graph = int(target_graph)
            chord_length = float(np.linalg.norm(x1[idx] - x0[idx])) if return_metadata else 0.0

            if not self.knn_constraint:
                path = np.array([source_graph, target_graph])
                connected = True
                use_linear = True
                used_linear_fallback = False
                path_length = chord_length
            else:
                path, flag = self._graph(t_start).graphical_path(
                    source_idx=source_graph,
                    end_idx=target_graph,
                )
                connected = bool(flag and len(path) >= 2)
                used_linear_fallback = not connected and self.linear
                use_linear = used_linear_fallback

                if connected and return_metadata:
                    path_length = float(
                        np.asarray(
                            self._graph(t_start).graphical_distance(
                                np.array([source_graph]),
                                np.array([target_graph]),
                            )
                        ).squeeze()
                    )
                else:
                    path_length = chord_length

            if return_metadata:
                pair_paths.append(np.asarray(path, dtype=int))
                pair_connected.append(connected)
                pair_linear_fallback.append(used_linear_fallback)
                pair_path_length.append(path_length)
                pair_chord_length.append(chord_length)

            if not connected and not use_linear:
                continue

            for _ in range(n_samples_in_path):
                t = random.random()
                if use_linear:
                    xa = (1-t) * x0[idx] + t * x1[idx]
                    ua = x1[idx] - x0[idx]
                else:
                    xa, ua = self._graph(t_start).interpolate_one_point(path, ti=t, )
                xa_t.append(xa)
                ua_t.append(ua)
                ts.append(t)
                if return_metadata:
                    sample_pair_id.append(idx)
                    sample_connected.append(connected)
                    sample_linear_fallback.append(used_linear_fallback)
                    sample_path_length.append(path_length)
                    sample_chord_length.append(chord_length)

        result = (np.array(xa_t), np.array(ua_t), np.array(ts)[:,None], x0, x1)
        if not return_metadata:
            return result

        metadata = {
            'alpha': np.asarray(ts, dtype=float),
            'pair_id': np.asarray(sample_pair_id, dtype=int),
            'connected': np.asarray(sample_connected, dtype=bool),
            'used_linear_fallback': np.asarray(sample_linear_fallback, dtype=bool),
            'path_length': np.asarray(sample_path_length, dtype=float),
            'chord_length': np.asarray(sample_chord_length, dtype=float),
            'pair_source_idx': np.asarray(i, dtype=int),
            'pair_target_idx': np.asarray(j_map, dtype=int) - self.n_list[t_start],
            'pair_x0': np.asarray(x0),
            'pair_x1': np.asarray(x1),
            'pair_connected': np.asarray(pair_connected, dtype=bool),
            'pair_used_linear_fallback': np.asarray(pair_linear_fallback, dtype=bool),
            'pair_path_length': np.asarray(pair_path_length, dtype=float),
            'pair_chord_length': np.asarray(pair_chord_length, dtype=float),
            'paths': pair_paths,
        }
        return result + (metadata,)

    

    def sample_batch_path(
            self,
            sigma : float , 
            batch_size : int,
            distance_metrics : str = 'L2', 
            add_noise : bool = True,
            n_samples_in_path : int = 1,
            return_metadata : bool = False,
            ):
        """ sample data point x_t and corresponding velocity u_t using OT and SP
        
        Parameters
        ----------
            sigma : float, belongs to [0,1]
                noise level of interpolated data point
            batch_size : int
            distance_metrics: str
                using 'SP' (i.e. shortest path distance) or 'L2'
            add_noise : bool
                xt add noise or not
            n_samples_in_path : int
                number of interpolation samples drawn from each OT-paired path
            return_metadata : bool
                return sampling metadata for debugging

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
        metadata_list = []
        pair_offset = 0

        for t_start in range(len(self.ts) - 1):

            sample_result = self._sample_one_time_point(
                t_start,
                batch_size=batch_size,
                distance_metrics=distance_metrics,
                n_samples_in_path=n_samples_in_path,
                return_metadata=return_metadata,
            )
            xa_t, ua_t, t, x0, x1 = sample_result[:5]
                
            if len(xa_t) == 0:
                raise Exception('low connection of graph, please increase `n_neighbors` or set `linear` into `True` ')
            X0.append(x0)
            X1.append(x1)
            
            #t = self.ts[t_start] + t
            t = t_start + t

            
            T.append(t)
            X.append(xa_t)
            U.append(ua_t)

            if return_metadata:
                metadata = sample_result[5]
                metadata['pair_id'] = metadata['pair_id'] + pair_offset
                metadata['transition_idx'] = np.full(len(xa_t), t_start, dtype=int)
                metadata['pair_transition_idx'] = np.full(len(x0), t_start, dtype=int)
                metadata_list.append(metadata)
                pair_offset += len(x0)
            
        
        X, U, T = np.concatenate(X), np.concatenate(U), np.concatenate(T)
        X, U, T =  X.reshape(-1, X.shape[-1]), U.reshape(-1, U.shape[-1]), T.reshape(-1, T.shape[-1])
        X_clean = X.copy()
        if add_noise:
            X = X + sigma*np.random.randn(X.shape[0], X.shape[1])
        T = torch.Tensor(T)
        X = torch.Tensor(X)
        U = torch.Tensor(U)
        result = (T, X, U, X0, X1)
        if not return_metadata:
            return result

        sample_fields = [
            'alpha',
            'pair_id',
            'connected',
            'used_linear_fallback',
            'path_length',
            'chord_length',
            'transition_idx',
        ]
        pair_fields = [
            'pair_source_idx',
            'pair_target_idx',
            'pair_x0',
            'pair_x1',
            'pair_connected',
            'pair_used_linear_fallback',
            'pair_path_length',
            'pair_chord_length',
            'pair_transition_idx',
        ]
        metadata = {
            key: np.concatenate([item[key] for item in metadata_list])
            for key in sample_fields + pair_fields
        }
        metadata['paths'] = [path for item in metadata_list for path in item['paths']]
        metadata['x_clean'] = X_clean
        metadata['pre_filter_t'] = T.detach().cpu().numpy().copy()
        metadata['pre_filter_x'] = X.detach().cpu().numpy().copy()
        metadata['pre_filter_u'] = U.detach().cpu().numpy().copy()
        metadata['filter_mask'] = np.ones(len(X), dtype=bool)
        return result + (metadata,)
    

    def filtered_sample_batch_path(
            self,
            sigma : float , 
            batch_size : int,
            distance_metrics : str = 'SP', 
            add_noise : bool = True,
            k=15,
            q=80,
            n_samples_in_path : int = 1,
            return_metadata : bool = False,
           
            ):
        """ sample data point x_t and corresponding velocity u_t using OT and SP, filter outlier using gaussian dist with knn center
        
        Parameters
        ----------
            sigma : float, belongs to [0,1]
                noise level of interpolated data point
            batch_size : int
            distance_metrics: str
                using 'SP' (i.e. shortest path distance) or 'L2'
            add_noise : bool
                xt add noise or not
            k : int
                knn kernel neighbors number
            q : int
                cutoff, filter in q % data points.
            n_samples_in_path : int
                number of interpolation samples drawn from each OT-paired path
            return_metadata : bool
                return sampling metadata for debugging

        Return
        ------
            T : torch.Tensor, shape=(batch_size * t_max, )
                correponding t to x
            X : torch.Tensor, shape=(batch_size * t_max, dim)
                sampled interpolated data points
            U : torch.Tensor, shape=(batch_size * t_max, dim)
                corresponding velocity to x

        """
        if not self.knn_constraint:
            return self.sample_batch_path(
                sigma,
                batch_size,
                distance_metrics,
                add_noise,
                n_samples_in_path=n_samples_in_path,
                return_metadata=return_metadata,
            )

        sample_result = self.sample_batch_path(
            sigma,
            batch_size,
            distance_metrics,
            add_noise,
            n_samples_in_path=n_samples_in_path,
            return_metadata=return_metadata,
        )
        T, X, U, X0, X1 = sample_result[:5]

        filtered_idx = filter_outlier(torch.Tensor(X), torch.tensor(U), k=k, q=q)
        filtered_idx = np.asarray(filtered_idx[0], dtype=int)
        T = T[filtered_idx]
        X = X[filtered_idx]
        U = U[filtered_idx]

        result = (T, X, U, X0, X1)
        if not return_metadata:
            return result

        metadata = sample_result[5]
        filter_mask = np.zeros(len(metadata['pre_filter_x']), dtype=bool)
        filter_mask[filtered_idx] = True
        metadata['filter_mask'] = filter_mask
        for key in [
            'alpha',
            'pair_id',
            'connected',
            'used_linear_fallback',
            'path_length',
            'chord_length',
            'transition_idx',
            'x_clean',
        ]:
            metadata[key] = metadata[key][filtered_idx]
        return result + (metadata,)



def filter_outlier(xt, ut, k=15, q=80):
    
    knn = KNeighborsRegressor(k)
    ut_norm = ut / np.linalg.norm(ut, axis=1)[:,np.newaxis]
    knn.fit(xt, ut_norm)
    ut_knn = knn.predict(xt)
    in_l = ((ut_norm - ut_knn)**2).sum(axis=1)
    c = np.percentile(in_l, q=q)
    
    return np.where(in_l < c)



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
        #print('#', torch.norm(xt[0]), torch.norm(ut[0]), torch.norm(vt[0]), loss)

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

        
