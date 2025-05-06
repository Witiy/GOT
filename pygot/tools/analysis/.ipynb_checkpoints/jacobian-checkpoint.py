import numpy as np
import torch

        
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

def get_jacobian(adata, time_key, embedding_key, ode_func,  gene_names=None, cell_idx=None, time_vary=True,
                device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cell_idx is None:
        cell_idx = adata.obs.index.tolist()
    if gene_names is None:
        gene_names = adata.var.index.tolist()
    if time_vary:
        input_data = np.concatenate(
        [adata[cell_idx].obsm[embedding_key], adata[cell_idx].obs[time_key].to_numpy()[:, None]], axis=-1
    )
    else:
        input_data = adata[cell_idx].obsm[embedding_key]
    
    
    x =  torch.Tensor(input_data).to(device)
    x.requires_grad = True
    y = ode_func(x)
    jacobian = _get_minibatch_jacobian(y, x)[:,:,:adata.obsm[embedding_key].shape[1]]
    return jacobian


