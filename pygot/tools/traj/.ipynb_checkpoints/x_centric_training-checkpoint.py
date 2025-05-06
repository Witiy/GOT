from scipy.sparse import save_npz, load_npz, csr_matrix
import scanpy as sc
import copy 
import os
from tqdm import tqdm
from torchdiffeq import odeint

from .v_centric_training import GraphicalOTVelocitySampler
from .loss_func import *
from .utils import NeighborsSampler

def get_path(
            sp_map,
            dist_map,
            X_start,
            X_end,
            target , 
            source):
               
        N_start = len(X_start)
        get_correct_x = lambda x :X_end[x - N_start] if x >= N_start else X_start[x]
        
        # get shorest path map
        path_map = sp_map[source]

        if path_map[target] < 0:          
            return np.array([get_correct_x(source), get_correct_x(target)]), None, 0

        path_dist = dist_map[source]
        dist = []
        Xi = []
        
        next_node = target
        # search map to get wanted path
        while (True):
            if path_map[next_node] < 0:
                break
            
            dist.append(path_dist[next_node])
            Xi.append(get_correct_x(next_node))
            next_node = path_map[next_node]

        Xi.append(get_correct_x(source))
        dist = np.array(dist[::-1])
        Xi = np.array(Xi[::-1])
        
        return Xi, dist, 1


def interpolate_func(Xi, dist, ratio):
    cutoff = ratio * dist[-1]

    for i in range(len(dist)):
        if dist[i] >= cutoff:
            start = i
            end = i + 1
            break
    a = ratio - dist[start - 1]/dist[-1] if start > 0 else ratio
    b = dist[start]/dist[-1] - ratio
    alpha = a / (a+b)
    return (1-alpha) * Xi[start] + alpha * Xi[end]

def get_nn(graph, node, k=5):
    nn = graph.getrow(node)
    
    distances = nn.data
    _, indices = nn.nonzero()
    d = -np.log(distances)
    prob = d / np.sum(d) 
    
    idx = np.random.choice(range(len(indices)), p=prob, size=k)
    
    return indices[idx]

class GraphicalOTInterpolater:
    def __init__(self, velocity_sampler:GraphicalOTVelocitySampler, 
                 distance_metrics='SP', randomized=False, n_neighbors=20) -> None:
        self.sp_sampler = velocity_sampler
        self.distance_metrics=distance_metrics
        self.randomized = randomized
        if self.randomized:
            self.construct_graph(self.sp_sampler.adata,
                                 self.sp_sampler.time_key,
                                 self.sp_sampler.graph_key,
                                 self.sp_sampler.data_dir,
                                 n_neighbors)
        

    def construct_graph(self, adata, time_key, graph_key, data_dir='', n_neighbors=20):
        
        ts = np.sort(np.unique(adata.obs[time_key]))
        self.X_graph = []
        for t in ts:
            if data_dir != '':
                file_path = os.path.join(data_dir, str(t)+'_graph.npz')
                
                if os.path.exists(file_path):
                    print('Loading kNN graph at time {}'.format(t))
                    self.X_graph.append(load_npz(file_path))
                    continue
            
            print('Construct kNN graph at time {}'.format(t))
            transition = sc.pp.neighbors(adata[adata.obs[time_key] == t], 
                                         n_neighbors=n_neighbors, use_rep=graph_key, copy=True)
            graph = transition.obsp['distances']
            graph.data = (graph.data - np.min(graph.data))/ (np.max(graph.data) - np.min(graph.data))
            graph = csr_matrix(graph.toarray())
            self.X_graph.append(graph)
            if data_dir != '':
                save_npz(file_path, graph)

    
    def interpolate(self, x0_idx, x1_idx, t_start, ratio=0.5, sigma=0.05,):
        try:
            x0, x1, i, j_map = self.sp_sampler.sample_pair(x0_idx, x1_idx, t_start, self.distance_metrics,
                                                           outlier_filter=False)
        except :
            
            raise Exception('if error with probability sum to 1, please increase batch size')
        
        x_paths = []
        ratios = np.random.normal(0, sigma, len(x0_idx)) + ratio

        if self.randomized:
            j = j_map - self.sp_sampler.n_list[t_start]
            for idx in range(len(x0)):
                indices = get_nn(self.X_graph[t_start+1], j[idx])
                j_nn_map = indices + self.sp_sampler.n_list[t_start]
                x_inter_idx = []

                for p in range(len(j_nn_map)):
                    Xi, dist, flag = get_path(sp_map=self.sp_sampler.sp_map[t_start], dist_map=self.sp_sampler.dist_map[t_start], 
                                              X_start=self.sp_sampler.X[t_start], X_end=self.sp_sampler.X[t_start+1],
                                    target=j_nn_map[p], source=i[idx])
                    if flag != 0:
                        xt = interpolate_func(Xi, dist, ratio=ratios[idx])
                    else:
                        xt = ratios[idx] * Xi[-1] + (1-ratios[idx]) * Xi[0]
                    x_inter_idx.append(xt)
                x_inter_idx = np.mean(np.array(x_inter_idx) , axis=0)
                x_paths.append(x_inter_idx)

        else:
            for idx in range(len(x0)):
            
                Xi, dist, flag = get_path(self.sp_sampler.sp_map[t_start], self.sp_sampler.dist_map[t_start], 
                                      self.sp_sampler.X[t_start], self.sp_sampler.X[t_start + 1], 
                                      target=j_map[idx], source=i[idx])
                if flag != 0:
                    x_inter = interpolate_func(Xi, dist, ratio=ratios[idx])
                else:
                    x_inter = ratios[idx] * Xi[-1] + (1-ratios[idx]) * Xi[0]
                x_paths.append(x_inter)   
        
        return np.array(x_paths), x0, x1


        

class XCentricTrainer:
    def __init__(self, adata, time_key, embedding_key, 
                 reverse_schema=True, reverse_n=2, 
                 batch_size=60,
                 lambda_density=5, lambda_ot=1, 
                 graph_inter=False, sp_sampler=None, distance_metrics='SP',randomized=False, 
                 neighbor_sampling=False,
                 device=torch.device('cpu')) -> None:
        self.ts = np.sort(np.unique(adata.obs[time_key]))
        self.reversed_ts = copy.deepcopy(self.ts[::-1])
        self.X = {
            t:adata[adata.obs[time_key] == t].obsm[embedding_key]
            for t in self.ts
        }
        self.reverse_schema = reverse_schema
        self.reverse_n = reverse_n
        self.batch_size = batch_size
        self.loss = XCentericLoss(lambda_density=lambda_density,
                                         lambda_ot=lambda_ot, device=device)
        self.ts_map = dict(zip(self.ts, range(len(self.ts))))
        self.graph_inter = graph_inter
        
        if graph_inter:
            assert sp_sampler != None
            self.gotiter = GraphicalOTInterpolater(sp_sampler, distance_metrics, randomized=randomized)
            
        else:
            self.graph_inter = False
        
        if neighbor_sampling:
            self.neighbor_sampler = {}
            for t in self.ts:
                self.neighbor_sampler[t] = NeighborsSampler(50, self.X[t])
        self.neighbor_sampling = neighbor_sampling
        self.device = device
        
    def sample(self, t, ):
        return np.random.choice(range(len(self.X[t])), size=self.batch_size)
    
    
    def graph_interpolate(self, X, t_idxs, group):
        
        reversed = group[0] > group[-1]
        X_inter = [X[0]]
        group_inter = [group[0]]

        for i in range(1, len(X)):
            x0_idx = t_idxs[i-1]
            x1_idx = t_idxs[i]
            t_start = group[i-1]
            t_diff = abs(group[i] - group[i-1])
            new_group = group[i-1] + t_diff * 0.5
            if reversed:
                x0_idx, x1_idx = x1_idx, x0_idx
                t_start = group[i]
                new_group = group[i-1] - t_diff * 0.5
            x_inter, _, _  = self.gotiter.interpolate(x0_idx, x1_idx, t_start=self.ts_map[t_start])
            X_inter.append(torch.Tensor(x_inter))
            group_inter.append(new_group)
            #inter
            X_inter.append(X[i])
            group_inter.append(group[i])

        return X_inter, group_inter
            
    def sample_X_group(self, i):
        t_start = i % (len(self.ts)-1)
        #t_start = 0
        reverse = True if self.reverse_schema and i % self.reverse_n == 0 else False

        X = [] 
        if reverse:
            group = self.reversed_ts[:len(self.ts)-t_start]
        else:
            group = self.ts[t_start:]
        if not self.neighbor_sampling or reverse:
            t_idxs = []
            for i in range(len(group)):
                t=group[i]
                t_idx = self.sample(t)
                t_idxs.append(t_idx)
                X.append(self.X[t][t_idx])
        else:
            # reverse sample and sample befor distribution according to the neighbor of last distribution 
            
            t_idxs = [self.sample(group[-1])]
            X.append(self.X[group[-1]][t_idxs[-1]])

            for i in range(len(group) -1)[::-1]:
                t=group[i]
                t_idx = self.neighbor_sampler[t].sample(X[-1])
                t_idxs.append(t_idx)
                X.append(self.X[t][t_idx])
            X = X[::-1]
            t_idxs = t_idxs[::-1]
        X = list(map(lambda x: torch.Tensor(x).float(), X))
        return X, t_idxs, group
   
    def calcu_loss(self, model, i):
        X, t_idxs, group = self.sample_X_group(i)
        
        if self.graph_inter:
            X, group = self.graph_interpolate(X, t_idxs, group)

        X_pred = odeint(model, X[0].to(self.device), t=torch.Tensor(group).to(self.device), method='rk4')
        loss = self.loss(X_pred, X, group)

        return loss
    

def x_centric_training(
        adata, time_key, embedding_key, model, 
        iter_n=2000,
        batch_size=60,
        lambda_density=5,
        lambda_ot=1,
        reverse_schema=True,
        reverse_n=2,
        graph_inter=False,
        sp_sampler=None,
        distance_metrics='L2',
        randomized=False,
        neighbor_sampling=False,
        device=torch.device('cpu')):
    history = []
    optim = torch.optim.AdamW(model.parameters(), weight_decay=0.001)
    trainer = XCentricTrainer(adata, time_key, embedding_key, 
                            batch_size=batch_size, reverse_schema=reverse_schema, reverse_n=reverse_n,
                            lambda_density=lambda_density, lambda_ot=lambda_ot, graph_inter=graph_inter, sp_sampler=sp_sampler,
                            randomized=randomized, distance_metrics=distance_metrics,
                            neighbor_sampling=neighbor_sampling, device=device)
    pbar = tqdm(range(iter_n))
    best_loss = np.inf
    losses = []
    for i in pbar:
        optim.zero_grad()
        loss = trainer.calcu_loss(model, i)
        
        history.append(loss.item())
        loss.backward()
        losses.append(history[-1])
        optim.step()
        if i % 100 == 0:
            losses = np.mean(losses)
            best_loss = np.min([losses, best_loss])
            pbar.set_description('loss :{:.4f}  best :{:.4f}'.format(losses, best_loss))
            losses = []
    return model, history

