import numpy as np
from tqdm import tqdm
import torch
import torchdiffeq
import pandas as pd
from functools import partial
from sklearn.neighbors import KDTree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def KNN_minibatch_torch_func(query, ref, ref_norm, K, av_mem=None, metric='euclidean'):
    if av_mem is None:
        av_mem = int(5e9)
    # 5000 Mb of GPU memory per batch
    Ntrain, D = ref.shape
    # Estimate the largest reasonable batch size:
    Ntest = query.shape[0]
    print
    # Remember that a vector of D float32 number takes up 4*D bytes:
    Ntest_loop = min(max(1, av_mem // (4 * D * Ntrain)), Ntest)
    Nloop = (Ntest - 1) // Ntest_loop + 1
    out1 = torch.empty(Ntest, K)
    out2 = torch.empty(Ntest, K).long()

    
    # Actual K-NN query:
    for k in range(Nloop):
        x_test_k = query[Ntest_loop * k : Ntest_loop * (k + 1), :]
        out1[Ntest_loop * k : Ntest_loop * (k + 1), :], out2[Ntest_loop * k : Ntest_loop * (k + 1), :] = KNN_torch_fun(
                    ref, ref_norm, x_test_k, K, metric
        )
    return out1, out2

def KNN_torch_fun(x_train, x_train_norm, x_test, K, metric):
    largest = False  # Default behaviour is to look for the smallest values

    if metric == "euclidean":
        x_test_norm = (x_test**2).sum(-1)
        diss = (
            x_test_norm.view(-1, 1)
            + x_train_norm.view(1, -1)
            - 2 * x_test @ x_train.t()  # Rely on cuBLAS for better performance!
        )

    elif metric == "manhattan":
        diss = (x_test[:, None, :] - x_train[None, :, :]).abs().sum(dim=2)

    elif metric == "angular":
        diss = x_test @ x_train.t()
        largest = True

    elif metric == "hyperbolic":
        x_test_norm = (x_test**2).sum(-1)
        diss = (
            x_test_norm.view(-1, 1)
            + x_train_norm.view(1, -1)
            - 2 * x_test @ x_train.t()
        )
        diss /= x_test[:, 0].view(-1, 1) * x_train[:, 0].view(1, -1)
    else:
        raise NotImplementedError(f"The '{metric}' distance is not supported.")
    
    return diss.topk(K, dim=1, largest=largest)   

def gpu_sampling(p, num_samples):
    return p.multinomial(num_samples=num_samples, replacement=False)


class StochasticNeighborsDiffusioner:
    
    
    def __init__(self, model, adata, embedding_key, check=True, check_step=50, dt=0.01, sigma=0.10, k=10):
        self.model = model
        self.check = check
        self.check_step = check_step
        self.dt = dt
        self.sigma = sigma
        self.k = k
        self.classifiaction_model = None
        self.embedding_key = embedding_key
        self.adata = adata
        self.device = next(self.model.parameters()).device
        self.ref = torch.tensor(adata.obsm[embedding_key]).float().to(self.device)
        if self.device == torch.device('cpu'):
            self.tree = KDTree(self.ref)
            self.query_func = partial(self.tree.query, k=self.k)
            print('cpu device | mapping cell with scipy KD-tree')
        else:
            
            self.ref_norm = (self.ref**2).sum(-1).to(self.device)
            self.query_func = partial(KNN_minibatch_torch_func, ref=self.ref, ref_norm=self.ref_norm, K=self.k)
            
            print('cuda device | mapping cell with tensor distance directly')
        
        

    @torch.no_grad()
    def travel(self, xt, step=10, t_start=0):
        xt = torch.Tensor(xt).to(self.device)
        
        traj = torchdiffeq.odeint(self.model, xt,method='rk4', t=torch.Tensor(np.linspace(t_start, t_start + self.dt * step, step)).to(self.device))
        
        xt = traj[-1].float()
        
        if self.check:
            dist, ind = self.query_func(xt)
            
            if self.k > 1:
                
                #dist = dist.max(dim=1)[0][:,None] - dist 
                #dist = dist / torch.sum(dist, dim=1)[:,None]
                indx = np.array([ind[i][torch.randperm(self.k)[0]] for i in range(dist.shape[0])])
                #indx = np.array([ind[i][np.random.choice(range(self.k), p=dist[i], size=1)[0]] for i in range(dist.shape[0])])
            else:
                indx = ind
            
            xt = self.ref[indx.flatten(), :]
            
        xt += self.sigma * torch.rand(xt.shape[0], xt.shape[1]).to(self.device)
        #print(xt)
        return xt, traj
    
    @torch.no_grad()
    def simulation(self, xt, 
                    t_start, t_end, 
                    ):
        xt = xt.to(next(self.model.parameters()).device)
        checkpoint_traj = [xt]
        next_xt = xt
        time_interval = self.dt * self.check_step
        total_step = int((t_end - t_start) / self.dt)
        limit = total_step // self.check_step
        remain_step = total_step % self.check_step
        all_traj = []
        for i in range(limit + 1):
            if i < limit:
                next_xt, traj = self.travel(next_xt, self.check_step, t_start)            
            else:
                next_xt, traj = self.travel(next_xt, remain_step, t_start)
            all_traj.append(traj)
            checkpoint_traj.append(next_xt)
            t_start += time_interval
                    
        checkpoint_traj, all_traj = torch.concat(checkpoint_traj, dim=0), torch.concat(all_traj, dim=0)
        
        #return checkpoint_traj, all_traj.reshape(-1, all_traj.shape[-2], all_traj.shape[-1])
        return checkpoint_traj, all_traj
        
    @torch.no_grad()
    def pred_cell_fate(self, x0_adata, time_key, end, sample_size=100):
        if self.classifiaction_model is None:
            raise Exception('Please set up cell fate by `setup_cell_fate` first')
        
        labels = self.classifiaction_model.classes_.tolist()
        print('Calculate Cell Fate of ', labels)
        simulated_fate = np.zeros(shape=(len(x0_adata), len(labels)))
        
        start_t_idxs = [x0_adata.obs[time_key] == start for start in np.sort(np.unique(x0_adata.obs[time_key]))]
        
        for i in tqdm(range(sample_size)):
            for j, start in enumerate(np.sort(np.unique(x0_adata.obs[time_key]))):
                _, all_traj = self.simulation(torch.Tensor(x0_adata.obsm[self.embedding_key][start_t_idxs[j]]), start, end)
                
                simulated_fate[start_t_idxs[j],:] += self.classifiaction_model.predict_proba(all_traj[-1].cpu().numpy())
                
        
        simulated_fate /= sample_size

        return pd.DataFrame(simulated_fate, columns=labels, index=x0_adata.obs.index)
    
    def setup_cell_fate(self, cell_type_key, cell_type_list=None, obs_index=None, specified_model=None):
        if not obs_index is  None:
            train_adata = self.adata[obs_index]
        elif not cell_type_list is None:
            train_adata = self.adata[(self.adata.obs[cell_type_key].isin(cell_type_list))]
        else:
            train_adata = self.adata

    
        X = train_adata.obsm[self.embedding_key]
        y_encoded = train_adata.obs[cell_type_key]
    
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        if specified_model is None:
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            model = specified_model

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=model.classes_))

        print("Accuracy:", accuracy_score(y_test, y_pred))
        self.classifiaction_model = model
        