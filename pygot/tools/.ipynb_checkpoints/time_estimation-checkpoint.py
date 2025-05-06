import torch
import torch.nn as nn
import torch.autograd
import hnswlib
from sklearn.metrics import pairwise_distances
import numpy as np
from tqdm import tqdm

# 计算散度
def div(x, output):
    divergence = 0.
    for i in range(output.shape[1]):  # 对输出的每个分量求导
        grad = torch.autograd.grad(outputs=output[:, i], inputs=x, grad_outputs=torch.ones_like(output[:, i]), create_graph=True, retain_graph=True)[0]
        divergence += grad[:, i]
        
    return divergence
# 定义一个简单的神经网络
class TimeModel(nn.Module):
    def __init__(self, dim):
        super(TimeModel, self).__init__()
        self.fc1 = nn.Linear(dim+1, 32)  # 3维输入
        self.fc2 = nn.Linear(32, 32)  # 3维输出
        self.fc3 = nn.Linear(32, 1)  # 3维输出
        #self.fc3 = nn.Linear(128, 32)
        #self.fc4 = nn.Linear(32, 1)

    def forward(self, x, t):
        x_t = torch.cat((x, t), dim=1)  # 将x和t拼接作为输入
        x_t = torch.relu(self.fc1(x_t))
        x_t = torch.relu(self.fc2(x_t))
        return nn.functional.sigmoid(self.fc3(x_t))

    
def calcu_expectation(f_net, x):
    pxt = []
    sample_points = torch.logit(torch.rand(10).to(x.device))
    for i in sample_points:
        t = i.repeat(x.shape[0],1)
        pxt.append(f_net(x, t).flatten())
    pxt = torch.stack(pxt)
    pt_x = (pxt / pxt.sum(dim=0))
    expectation = (pt_x * sample_points[:,None]).sum(dim=0)
    return expectation

def torch_pearsonr_fix_y(x, y, dim=1):
    x = x - torch.mean(x, dim=dim)[:,None]
    #y = y - torch.mean(y, dim=dim)[:,None]
    x = x / (torch.std(x, dim=dim) + 1e-9)[:,None]
    #y = y / (torch.std(y, dim=dim) + 1e-9)[:,None]
    return torch.mean(x * y, dim=dim)  # (D,)
    
def cosine(a, b):
    return np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1))


def get_pair_wise_neighbors(X, n_neighbors=30):
    N_cell = X.shape[0]
    dim = X.shape[1]
    if N_cell < 3000:
        ori_dist = pairwise_distances(X, X)
        nn_t_idx = np.argsort(ori_dist, axis=1)[:, 1:n_neighbors]
    else:
        p = hnswlib.Index(space="l2", dim=dim)
        p.init_index(max_elements=N_cell, ef_construction=200, M=30)
        p.add_items(X)
        p.set_ef(n_neighbors + 10)
        nn_t_idx = p.knn_query(X, k=n_neighbors)[0][:, 1:].astype(int)
    return nn_t_idx


class TimeEstimator:
    def __init__(self, adata, embedding_key, n_neighbors=30, device=None):
        print('Should be deca! instead of ContinuityModel')
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.f_net = TimeModel(adata.obsm[embedding_key].shape[1]).to(device)
        self.x = torch.tensor(adata.obsm[embedding_key], requires_grad=True).float().to(device)
        self.nn_t_idx = get_pair_wise_neighbors(adata.obsm[embedding_key], n_neighbors=n_neighbors)
        self.v_hat = adata.obsm[embedding_key][self.nn_t_idx.flatten()].reshape(self.nn_t_idx.shape[0], self.nn_t_idx.shape[1], -1) - adata.obsm[embedding_key][:,None, :]
        #self.v_hat = self.v_hat.to(device)
        self.device = device
        
        
    def train_time_model(self, adata, embedding_key, time_key, v_net,  n_iters=1000, corr_cutoff=0.3, continuity_cutoff=0.01):
        adata.obs[time_key] = adata.obs[time_key].astype(float)
        v_net.to(self.device)
        mini_batch = False
        if len(adata) > 5000:
            mini_batch=True
            batch_size = 5000
        with torch.no_grad():
            t = torch.tensor(adata.obs[time_key], requires_grad=True)[:,None].float().to(self.device)
            v = v_net(self.x if not v_net.time_varying else torch.concat([self.x, t], dim=-1)).detach().cpu().numpy()
            cos_sim = cosine(v[:,None,:], self.v_hat)
            cos_sim = torch.tensor(cos_sim)
            norm_cos_sim = cos_sim - torch.mean(cos_sim, dim=1)[:,None]
            norm_cos_sim = norm_cos_sim / (torch.std(norm_cos_sim, dim=1) + 1e-9)[:,None]
            norm_cos_sim = norm_cos_sim.to(self.device)
        
        optimizer = torch.optim.SGD(self.f_net.parameters(), lr=1e-3)
        pbar = tqdm(range(n_iters))

        for i in pbar:
            if not mini_batch:
                sample_x = self.x
                sample_t = t
            else:
                batch_idx = np.random.choice(range(len(adata)), size=batch_size, replace=False)
                sample_x = self.x[batch_idx]
                sample_t = t[batch_idx]
            
            t_noise  = sample_t + torch.randn_like(sample_t) * 0.05
            x_noise  = sample_x + torch.randn_like(sample_x) * 0.05
            t_noise.requires_grad = True
           
            f_output = self.f_net(x_noise, t_noise)  # f(x, t) 的输出
            v_output = v_net(x_noise if not v_net.time_varying else torch.concat([x_noise, t_noise], dim=-1))

            # 计算 f(x, t) * v(x) 对于 x 的散度
            product = f_output * v_output  # 逐元素乘积
            div_pv = div(x_noise, product)
            dpdt = div(t_noise, f_output)

            continuity_loss = (dpdt + div_pv)**2
            continuity_loss = torch.relu(continuity_loss - continuity_cutoff)
            continuity_loss = torch.mean((dpdt + div_pv)**2)
        

            expectation = calcu_expectation(self.f_net, self.x)
            
            delta_t = expectation[self.nn_t_idx.flatten()].reshape(self.nn_t_idx.shape[0], self.nn_t_idx.shape[1]) - expectation[:,None]
            corr = torch_pearsonr_fix_y(delta_t, norm_cos_sim)
            mask = corr < corr_cutoff
            if torch.sum(mask) > 0:
                corr = torch.mean(corr[mask])
            else:
                corr = 0.
    
            loss = continuity_loss - corr 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (1 - torch.sum(mask).item() / len(mask)) > 0.95:
                break
            pbar.set_description("Train Loss {:.4f}, Corr {:.4f} Satisfied {:2f}%".format(continuity_loss.item(), corr.item(),  (1 - torch.sum(mask).item() / len(mask))*100 ))
            
    @torch.no_grad()
    def update_time(self, adata, update_key='expectation'):
        pxt = []
        for i in torch.logit(torch.linspace(0.05,0.95,100)):
            t = i.repeat(self.x.shape[0], 1).to(self.device)

            pxt.append(self.f_net(self.x, t).detach().cpu().flatten().numpy())
        pxt = np.array(pxt)

        pt_x = (pxt / pxt.sum(axis=0))
        expectation = (pt_x * np.linspace(0.05,0.95,100)[:,None]).sum(axis=0)
        expectation = (expectation - np.min(expectation)) / (np.max(expectation) - np.min(expectation))
        adata.obs[update_key] = expectation