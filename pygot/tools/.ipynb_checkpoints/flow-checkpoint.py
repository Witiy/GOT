import torch
from scipy.stats import pearsonr
from scipy.sparse import issparse
from tqdm import tqdm
def latent_velocity(adata, odefunc, embedding_key='X_pca', time_vary=True, time_key=None, return_np=True):
    
    xt = torch.Tensor(adata.obsm[embedding_key])
    #TODO 如果时间统一了的话，这里需要修改
    if time_vary:
        #time_map = {i: t for t, i in enumerate(np.sort(np.unique(adata.obs[time_key])))}
        t = adata.obs[time_key]
        #t = torch.Tensor(np.array([time_map[x] for x in t]))[:, None]
        t = torch.Tensor(t)[:,None]
        #vt = model(xt, t).detach()
        
        vt = odefunc(torch.concat([xt, t], dim=-1).to(next(odefunc.parameters()).device))
    else:
        vt = odefunc(xt.to(next(odefunc.parameters()).device))
    if return_np:
        vt = vt.detach().cpu().numpy()
    return vt

def latent2gene_velocity(adata, velocity_key, A=None, dr_mode='linear', inverse_transform=None, dt=0.001):
    with torch.no_grad():
        v_latent = adata.obsm[velocity_key]
        if dr_mode == 'linear':
            adata.layers['velocity'] = v_latent @ A[:v_latent.shape[1]]
        elif dr_mode == 'nonlinear' and inverse_transform is not None:
            x0 = inverse_transform(adata.obsm[embedding_key])
            x1 = inverse_transform(adata.obsm[embedding_key] + v_latent * dt)
            adata.layers['velocity'] = (x1 - x0) / dt

def velocity(adata, odefunc, embedding_key='X_pca',  velocity_key=None, A=None, time_vary=True, time_key=None,
             dr_mode='linear', inverse_transform=None, dt=.001):
    
    if embedding_key == 'X_pca' and dr_mode == 'linear' and A is None:
        A = adata.varm['PCs'].T
    if velocity_key is None:
        velocity_key = 'velocity_'+embedding_key.split('_')[-1]
    v_latent = latent_velocity(adata, odefunc, embedding_key, time_vary, time_key, return_np=True)
    adata.obsm[velocity_key] = v_latent
    latent2gene_velocity(adata, velocity_key, A, dr_mode, inverse_transform, dt)


def calcu_dynamic_corr(adata, time_key='expectation', layer_key=None):
    corrs = []
    if layer_key is None:
        if issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = adata.X
    else:
        X = adata.layers[layer_key]
    t = adata.obs[time_key].to_numpy()
    for i in tqdm(range(X.shape[1])):
        corrs.append(pearsonr(X[:,i], t)[0])
    adata.var['corr'] = corrs
'''
def velocity(adata, odefunc, embedding_key='X_pca',  A=None, time_vary=True, time_key=None,
             dr_mode='linear', inverse_transform=None, dt=.001):
    
    if embedding_key == 'X_pca' and dr_mode == 'linear' and A is None:
        A = adata.varm['PCs'].T
    with torch.no_grad():
        v_latent = latent_velocity(adata, odefunc, embedding_key, time_vary, time_key, return_np=False)
        adata.obsm['velocity_'+embedding_key.split('_')[-1]] = v_latent.cpu().numpy()
        if dr_mode == 'linear':
            adata.layers['velocity'] = (v_latent @ torch.tensor(A[:v_latent.shape[1],:]).float().to(next(odefunc.parameters()).device)).cpu().numpy()
        elif dr_mode == 'nonlinear' and inverse_transform is not None:
            x0 = inverse_transform(adata.obsm[embedding_key])
            x1 = inverse_transform(adata.obsm[embedding_key] + v_latent * dt)
            adata.layers['velocity'] = (x1 - x0) / dt
'''    
    

   