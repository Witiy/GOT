import torch
from tqdm import tqdm
from torchdiffeq import odeint
import numpy as np
from .model_training import ODEwrapper, ODEwrapperNoTime
from anndata.experimental.pytorch import AnnLoader
from functools import partial
from tqdm import tqdm

def _get_minibatch_jacobian(y, x, return_np=True):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.

    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(y[:, j], x, torch.ones_like(y[:, j]), retain_graph=True,
                                      create_graph=True)[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    if return_np:
        return jac.detach().cpu().numpy()
    else:
        return jac



def latent2gene_velocity_scVI(ad,  embedding_key, velocity_key, vae, batch_size=64):
    use_cuda = vae.device.type != 'cpu'
    
    dataloader = AnnLoader(ad, batch_size=batch_size, shuffle=False, use_cuda=use_cuda)
    velocity_list = []
    
    for batch in tqdm(dataloader):
        z=batch.obsm[embedding_key]
        z.requires_grad = True
        px = vae.module.generative(z=z, library=torch.ones(z.shape[0], 1), batch_index=torch.tensor(batch.obs['_scvi_batch']).long()[:,None])['px']
        
        jac_xz = _get_minibatch_jacobian(px.mean, z, return_np=False)
        with torch.no_grad():
            velocity_batch = torch.matmul(jac_xz, batch.obsm[velocity_key].unsqueeze(-1)).squeeze(-1).cpu().numpy()
            velocity_list.append(velocity_batch)
            
        torch.cuda.empty_cache()
        
    return np.concatenate(velocity_list, axis=0)

def latent_velocity(adata, odefunc, embedding_key='X_pca', time_key=None):
    """Latent velocity inference using trained model.

    Arguments:
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    odefunc: class:`ODEwrapper` or class:`ODEwrapperNoTime`
        trained NeuralODE model by function `fit_velocity_model`
    embedding_key: `str`
        Name of latent space, in adata.obsm
    time_key: `str` (default: None)
        Name of time label, in adata.obs, use if the model input contains time label
    
    Returns
    -------
    latent_velocity (.obsm): :class`np.ndarray`
        latent velocity array, (n_cells, latent_dim), store in adata.obsm[`velocity_key`]
    """
    if isinstance(odefunc, ODEwrapperNoTime) or isinstance(odefunc, ODEwrapper):
        odefunc = odefunc.func
    assert (not odefunc.time_varying) or (odefunc.time_varying and (not time_key is None)), 'please offer `time_key`'

    xt = torch.Tensor(adata.obsm[embedding_key])
    
    if odefunc.time_varying:
        
        t = adata.obs[time_key]
        
        t = torch.Tensor(t)[:,None]
        
        
        vt = odefunc(torch.concat([xt, t], dim=-1).to(next(odefunc.parameters()).device))
    else:
        vt = odefunc(xt.to(next(odefunc.parameters()).device))
    
    return vt.detach().cpu().numpy()

def latent2gene_velocity(adata, velocity_key, embedding_key, A=None, dr_mode='linear', inverse_transform=None, dt=0.001):
    """Transform latent velocity into gene velocity

    Due to linearity and orthogonality, the gene velocity can be recover by directly multiply inverse 
    dimension reduction matrix, 

    .. math::

        v(x) ≈  Av(z), \quad z = A^Tx

    For non-linear transformation, 

    .. math::
    
        v(x) ≈ \\frac{g^{-1}(z + v(z) * dt) - g^{-1}(z)}{dt}, \quad z=g(x)

    Arguments:
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    embedding_key: `str`
        Name of latent space, in adata.obsm
    velocity_key: `str` (default: None)
        Name of latent velocity, in adata.obsm
    A: `np.ndarray` (default: None (using adata.varm['PCs']))
        Inverse matrix of linear dimension reduction
    dr_mode: 'linear' or 'nonlinear' (default: 'linear')
        Dimension reduction mode
    inverse_transform: `function` (default: None)
        Inverse function for non-linear dimension reduction (e.g. :math:`g^{-1}`)
    dt: `float` (default: 0.001)
        Parameter of non-linear velocity transformation from latent space to gene space
       
    Returns
    -------
    velocity (.layers): :class`np.ndarray`
        gene velocity array, (n_cells, n_genes)
    """
    if dr_mode not in {"linear", "nonlinear"}:
        raise ValueError(f"Dimension reduction mode must be 'linear' or 'nonlinear', was '{dr_mode}'.")
    with torch.no_grad():
        v_latent = adata.obsm[velocity_key]
        if dr_mode == 'linear':
            v_gene = v_latent @ A[:v_latent.shape[1]]
        elif dr_mode == 'nonlinear' and inverse_transform is not None:
            x0 = inverse_transform(adata.obsm[embedding_key])
            x1 = inverse_transform(adata.obsm[embedding_key] + v_latent * dt)
            v_gene = (x1 - x0) / dt
        else:
            raise NotImplementedError()
        return v_gene

def velocity(adata, odefunc, embedding_key='X_pca',  velocity_key=None, A=None, time_key=None,
             dr_mode='linear', inverse_transform=None, dt=.001):
    """Velocity inference using trained model.

    This function will infer velocity in latent space and gene space both. It can be sperate into
    `latent_velocity` and `latent2gene_velocity`

    Example:
    ----------
    For linear dimension reduction space::

        #using pca space as example
        #if pca is done in scanpy framework, the inverse matrix A will be store in adata.varm['PCs'], that do not need to specifed matrix A
        pygot.tl.traj.velocity(adate, model, embedding_key='X_pca', velocity_key='velocity_pca')
        #Otherwise, need to specify dimension reduction matrix A
        pygot.tl.traj.velocity(adate, model, embedding_key='X_pca', velocity_key='velocity_pca', A=pca.components_.T)

    For non-linear dimension reduction space::
    
        #using vae latent space as example
        #first, train the vae model to transform space
        gs_vae = pygot.pp.GS_VAE()
        gs_vae.register_model(adata, latent_dim=10)
        adata.obsm['X_latent'] = gs_vae.fit_transform(adata)

        #After train velocity model using `fit_velocity_model` or `fit_velocity_model_without_time`
        pygot.tl.traj.velocity(adate, model, 
            embedding_key='X_latent', velocity_key='velocity_latent', dr_mode='nonlinear', 
            inverse_transform=gs_vae.inverse_transform
        )

    Arguments:
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    odefunc: class:`ODEwrapper` or class:`ODEwrapperNoTime`
        trained NeuralODE model by function `fit_velocity_model`
    embedding_key: `str`
        Name of latent space, in adata.obsm
    velocity_key: `str` (default: None)
        Name of latent velocity to save, in adata.obsm
    A: `np.ndarray` (default: None (using adata.varm['PCs']))
        Inverse matrix of linear dimension reduction
    time_key: `str` (default: None)
        Name of time label, in adata.obs, use if the model input contains time label
    dr_mode: 'linear' or 'nonlinear' (default: 'linear')
        Dimension reduction mode
    inverse_transform: `function` (default: None)
        Inverse function for non-linear dimension reduction
    dt: `float` (default: 0.001)
        Parameter of non-linear velocity transformation from latent space to gene space
       
    Returns
    -------
    velocity (.layers): :class`np.ndarray`
        gene velocity array, (n_cells, n_genes)
    latent_velocity (.obsm): :class`np.ndarray`
        latent velocity array, (n_cells, latent_dim), store in adata.obsm[`velocity_key`]
    """
    if isinstance(odefunc, ODEwrapperNoTime) or isinstance(odefunc, ODEwrapper):
        odefunc = odefunc.func
    assert (not odefunc.time_varying) or (odefunc.time_varying and (not time_key is None)), 'please offer `time_key`'

    if embedding_key == 'X_pca' and dr_mode == 'linear' and A is None:
        A = adata.varm['PCs'].T
    if velocity_key is None:
        velocity_key = 'velocity_'+embedding_key.split('_')[-1]
    v_latent = latent_velocity(adata, odefunc, embedding_key, time_key)
    adata.obsm[velocity_key] = v_latent
    v_gene = latent2gene_velocity(adata, velocity_key,embedding_key, A, dr_mode, inverse_transform, dt)
    adata.obsm['velocity'] = v_gene
    return v_gene

def simulate_trajectory(adata, odefunc, embedding_key, start, end, n_points=100):
    """Simulate trajecotry using trained NeuralODE model.

    Arguments:
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    odefunc: class:`~ODEwrapper` or class:`~ODEwrapperNoTime`
        trained NeuralODE model by function `fit_velocity_model`
    embedding_key: `str'
        Name of latent space to fit, in adata.obsm
    start: `int`
        Start time of simulation
    end: `int`
        End time of simulation
    n_points: `int`
        Number of points in one trajectory (e.g. 100 points from day0 -> day7)
       
    Returns
    -------
    trajectory: :class`~np.ndarray`
        Trajectory array, (n_points, n_cells, latent_dim)
    """
    latent_embedding = adata.obsm[embedding_key]
    with torch.no_grad():
        return odeint(
            odefunc, 
            torch.tensor(latent_embedding).float().to(next(odefunc.parameters()).device),
            torch.tensor(np.linspace(start, end, n_points)).to(next(odefunc.parameters()).device),
        ).detach().cpu().numpy()




@torch.no_grad()
def _inverse_transform_scVI(z, vae, ad, batch_size=1024):
    use_cuda = vae.device.type != 'cpu'
    ad.obsm['X_lowd'] = z
    dataloader = AnnLoader(ad, batch_size=batch_size, shuffle=False, use_cuda=use_cuda)
    scaled_X = []
    for batch in dataloader:
        z=batch.obsm['X_lowd']
        px = vae.module.generative(z=z, library=torch.ones(z.shape[0], 1), batch_index=torch.tensor(batch.obs['_scvi_batch']).long()[:,None])['px']
        scaled_X.append(px.mean.cpu().numpy())
    return np.concatenate(scaled_X, axis=0)

def get_inverse_transform_func_scVI(adata, vae, batch_size=1024):
    inverse_transform = partial(_inverse_transform_scVI, vae=vae, ad=adata, batch_size=batch_size)
    return inverse_transform
    


    

   