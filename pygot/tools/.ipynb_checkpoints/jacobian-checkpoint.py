import numpy as np
import scanpy as sc
import torch
import pandas as pd
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF

class GRN:
    def __init__(self, latent_jacobian, gene_names) -> None:
        self.latent_jacobian = latent_jacobian.astype('float32')
        self.gene_names = pd.DataFrame(gene_names, columns=['gene_name']).set_index('gene_name')
        self.gene_names['idx'] = range(len(self.gene_names))
    
    def getdxdy(self, A, dx_name, dy_name):
        dx_idx = self.gene_names.loc[dx_name].idx
        dy_idx = self.gene_names.loc[dy_name].idx
        return A.T[dx_idx, :] @ self.latent_jacobian @ A[:, dy_idx]

    def latent2gene_jacobian(self, A, stest=True, t=10, low_memory=False):
        mean_latent_jacobian = np.mean(self.latent_jacobian, axis=0)

        gene_jacobian = A.T @ mean_latent_jacobian @ A
        self.gene_jacobian_df = pd.DataFrame(gene_jacobian, index=self.gene_names.index, columns=self.gene_names.index)

        if stest == False:
            self.pvalues_df = None
            return 
        
        
        permutated_matrix = np.stack([shuffle_matrix(mean_latent_jacobian) for i in range(t)]).astype('float32')
    
        if low_memory == False:
            try:
                permutated_jacobian =  A.T.astype('float32') @ permutated_matrix @  A.astype('float32')
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                print("set 'low_memory' as True or decrease t")
                return
        else:
            permutated_jacobian = []
    
        ps = []
        for i in tqdm(range(gene_jacobian.shape[0])):
            item = []
            for j in range(gene_jacobian.shape[1]):
                if len(permutated_jacobian) != 0:
                    pool = permutated_jacobian[:, i, j]
                else:
                    pool =  A.T[i,:] @ permutated_matrix @  A[:,j]
                if gene_jacobian[i,j] < 0:
                    pool = pool[pool < 0]
                    ecdf = ECDF(pool)
                    p = ecdf(gene_jacobian[i,j])
                else:
                    pool = pool[pool >= 0]
                    ecdf = ECDF(pool)
                    p = 1-ecdf(gene_jacobian[i,j])
        
            #p = min(p, 1-p) * 2

                item.append(p)
            ps.append(item)
        ps = np.array(ps)
        self.pvalues_df = pd.DataFrame(ps, index=self.gene_names.index, columns=self.gene_names.index)
        return 

        
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
    return GRN(jacobian, gene_names)


def shuffle_matrix(matrix_data):

    # 生成随机排列的行索引和列索引
    random_row_indices = torch.randperm(matrix_data.shape[0])
    random_col_indices = torch.randperm(matrix_data.shape[1])

    # 对行进行随机打乱
    shuffled_matrix = matrix_data[random_row_indices, :]

    # 对列进行随机打乱
    shuffled_matrix = shuffled_matrix[ :, random_col_indices]
    return shuffled_matrix


from statsmodels.distributions.empirical_distribution import ECDF
import copy
def sparse_gene_network(gene_jacobian_df):
    sparse_df = copy.deepcopy(gene_jacobian_df)
    # 每个基因被直接调控应该是sparse的
    for idx in sparse_df.index:

        x = gene_jacobian_df.loc[idx]
        mu = np.mean(x)
        sigma = np.std(x)
        sparse_idx = (x < (mu + 3 * sigma)) & (x > (mu - 3 * sigma))
        sparse_df.loc[idx, sparse_idx] = 0
    # 调控基因的target也让sparse

        x = gene_jacobian_df[idx]
        mu = np.mean(x)
        sigma = np.std(x)
        sparse_idx = (x < (mu + 3 * sigma)) & (x > (mu - 3 * sigma))
        sparse_df.loc[sparse_idx, idx] = 0
    return sparse_df

