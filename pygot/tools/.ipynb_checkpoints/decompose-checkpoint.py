import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
import torch
from tqdm import tqdm
from pygot.tools.loss_func import torch_pearsonr_fix_y

class BasicVelocityFunction(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.CELU(),
            torch.nn.Linear(16, 32),
            torch.nn.CELU(),
            torch.nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class DecompositeVelocityFunction(torch.nn.Module):
    def __init__(self, input_dim, output_dim, n_lineages):
        super().__init__()
        self.vector_lineages = torch.nn.ModuleList([
            BasicVelocityFunction(input_dim, output_dim) for _ in range(n_lineages)
        ])
        self.vector_growth = BasicVelocityFunction(input_dim, output_dim)
        
        self.n_lineages = n_lineages

    def forward(self, v, x, idx, norm_t, v_mean):
        v_g = self.vector_growth(x)
        v_ls = torch.zeros_like(v)
        loss_orth, loss_recon = 0., 0.
        cutoff = torch.zeros(v.shape[0],).to(v.device)
        for i in range(self.n_lineages):
            selected = (idx == i)
            v_l = self.vector_lineages[i](x[selected])
            loss_orth += torch.mean(torch.sum(v_g[selected] * v_l, dim=1) ** 2)
            loss_recon += torch.mean((v[selected] - (v_g[selected] + v_l))**2)
            v_ls[selected] = v_l
                  
        scores = torch.stack([torch.norm(v_g, dim=1), torch.norm(v_ls, dim=1)], dim=1)
        scores /= torch.sum(scores, dim=1)[:, None]
        #positive to lineages, negative to growth
        pcc_l = torch_pearsonr_fix_y(scores[:,1][:,None], norm_t, dim=0) 
        pcc_g = torch_pearsonr_fix_y(scores[:,0][:,None], norm_t, dim=0)
        
        loss_pcc = torch.tensor(0.)
        loss_pcc = loss_pcc + (pcc_l if pcc_l < 0.7 else 0)
        loss_pcc = loss_pcc - (pcc_g if pcc_g > -0.7 else 0)
        #maximam pcc
        loss_pcc = -loss_pcc
        
        loss_balance = torch.mean(torch.stack([torch.sum(v_g * v_ref, dim=1) / torch.norm(v_ref) for v_ref in v_mean]).std(dim=0))
        #loss_balance = torch.mean( ((torch.sum(v_g * v_mean[0], dim=1) / torch.norm(v_mean[0])) \
        #                - (torch.sum(v_g * v_mean[1], dim=1) / torch.norm(v_mean[1])) )**2)
        
        return loss_recon, loss_orth, loss_pcc, loss_balance

    @torch.no_grad()
    def inference(self, x, idx):
        v_g = self.vector_growth(x)
        v_ls = torch.zeros_like(v_g)
        for i in range(self.n_lineages):
            selected = (idx == i)
            v_l = self.vector_lineages[i](x[selected])
            v_ls[selected] = v_l
        return v_g, v_ls

def velocity_decompose(adata, lineages, time_key, stage_key, embedding_key='X_pca', velocity_key='velocity_pca', lineage_key='future_cell_type', 
                         lam=0.05, num_epochs=5000, device=None):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Lambda : {lam}')
    input_dim, output_dim, m = adata.obsm[embedding_key].shape[1], adata.obsm[velocity_key].shape[1], len(lineages)
    early_stage_idx = (adata.obs[stage_key] == 'early').to_numpy()

    idx = np.zeros(len(adata))
    for i, l in enumerate(lineages):
        idx[adata.obs[lineage_key] == l] = i
    idx = idx.astype(int)
    
    vs = torch.tensor(adata.obsm[velocity_key]).float().to(device)
    x = torch.tensor(adata.obsm[embedding_key]).float().to(device)
    t = torch.tensor(adata.obs[time_key].to_numpy()[:,None]).float().to(device)
    norm_t = t - torch.mean(t, dim=0)[:,None]
    norm_t = norm_t / (torch.std(norm_t, dim=0) + 1e-9)[:,None]
    v_mean = [torch.tensor(adata[adata.obs[lineage_key] == l].obsm[velocity_key].mean(axis=0)).to(device) for l in lineages]

    
    decom_func = DecompositeVelocityFunction(input_dim, output_dim, m).to(device)
    decom_func.train()
    
    ##pre-train
    print('Using early stage velocity to pre-train growth vector field')
    optimizer = optim.SGD(decom_func.vector_growth.parameters(), lr=0.01, weight_decay=0.01) 
    pbar = tqdm(range(num_epochs//20 + 1) )
    for epoch in pbar:
        optimizer.zero_grad()
        
        v_g = decom_func.vector_growth(x)
        
        loss = torch.mean((vs[early_stage_idx,:]-v_g[early_stage_idx,:])**2)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            pbar.set_description(f'Epoch [{epoch+1}/{num_epochs//20}], Loss : {loss.item():.4f}  ')
            
    print('Using late stage velocity to pre-train lineage vector field')
    optimizer = optim.SGD(decom_func.vector_lineages.parameters(), lr=0.01, weight_decay=0.01) 
    
    pbar = tqdm(range(num_epochs//20 + 1))
    for epoch in pbar:
        optimizer.zero_grad()
        
        v_ls = torch.zeros_like(vs)
        for i in range(decom_func.n_lineages):
            selected = (idx == i)
            v_l = decom_func.vector_lineages[i](x[selected])
            #print(v_l.shape)
            v_ls[selected] = v_l
        
        loss = torch.mean((vs[~early_stage_idx]-v_ls[~early_stage_idx])**2) 
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0:
            pbar.set_description(f'Epoch [{epoch+1}/{num_epochs//20}], Loss : {loss.item():.4f}  ')

    #train
    print('Starting decomposing velocity into growth(shared) and lineage(specified) velocity..')
    optimizer = optim.SGD(decom_func.parameters(), lr=0.003, weight_decay=0.01)
    
    pbar = tqdm(range(num_epochs))
    flag=True
    for epoch in pbar:

        optimizer.zero_grad()
        
        l_recon, l_orth, l_pcc, l_balance = decom_func(vs, x, idx, norm_t, v_mean)
        
        loss = l_recon + 0.15*l_orth  + lam * l_pcc + lam * l_balance
        loss.backward()

        optimizer.step()
        if (epoch+1) % 10 == 0:
            pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Loss : {loss.item():.4f}  Recon : {l_recon.item():.4f}  Orth : {l_orth.item():.4f} PCC: {l_pcc.item() :.4f} BAL: {l_balance.item() :.4f}')
            
    
    decom_func.eval()
    with torch.no_grad():
        v_g, v_l = decom_func.inference(x, idx)
        scores = torch.stack([torch.norm(v_g, dim=1), torch.norm(v_l, dim=1)], dim=1)
        #scores = torch.stack([torch.sum(v_g * vs, dim=1) / (torch.norm(v_g, dim=1) * torch.norm(vs, dim=1)), torch.sum(v_l * vs, dim=1) / (torch.norm(v_l, dim=1) * torch.norm(vs, dim=1)) ], dim=1)
        scores /= torch.sum(scores, dim=1)[:, None]
    
    adata.obsm[velocity_key + '_growth'] = v_g.detach().cpu().numpy()
    adata.obsm[velocity_key + '_lineages'] = v_l.detach().cpu().numpy()
    adata.obs[['growth', 'lineages']] = scores.detach().cpu().numpy()
    
    
