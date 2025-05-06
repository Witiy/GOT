import torch
import torch.nn as nn
import torch.autograd
from sklearn.metrics import pairwise_distances
from pygot.tools.traj.utils import find_neighbors
import numpy as np
from tqdm import tqdm
import torch.nn as nn

from scipy.optimize import minimize_scalar
from torch.distributions.normal import Normal
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

def dcor_test(adata, pseudotime_key, num_resamples=1):
    import dcor
    X = adata.X.toarray()
    y = adata.obs[pseudotime_key].to_numpy()
    res = []
    for i in tqdm(range(adata.shape[1])):
        res.append(list(dcor.independence.distance_covariance_test(
            X[:,i],
            y,
            num_resamples=num_resamples,
        ))
    )
    adata.var[['pvalue', 'statistic']] = np.array(res)

def strings_to_tensor(string_list):
    
    unique_strings = list(set(string_list))
    string_to_index = {s: idx for idx, s in enumerate(unique_strings)}
    
    
    integer_list = [string_to_index[s] for s in string_list]
    
    
    tensor = torch.tensor(integer_list, dtype=torch.long)
    return tensor, string_to_index
    
def std_bound(x):
    upper_bound = np.mean(x)+3*np.std(x)
    lower_bound = np.mean(x)-3*np.std(x)
    x[x > upper_bound] = upper_bound
    x[x < lower_bound] = lower_bound
    return x


def normal_sample(mu, logvar, epsilon=1e-6):
    std = F.softplus(logvar) + epsilon
    dist = Normal(mu, std)
    
    z = dist.rsample()
    t = z  # Map to (-1, 1) using tanh
    return t

def normal_log_likelihood(x, mu, logvar, epsilon=1e-6):
    
    std = F.softplus(logvar) + epsilon
    
    normal_dist = Normal(mu, std)
    log_prob_z = normal_dist.log_prob(x)
    return log_prob_z


# Kumaraswamy 
def log_likelihood(x, a, b):
    if x <= 0 or x >= 1:
        return -np.inf  
    return np.log(a) + np.log(b) + (a - 1) * np.log(x) + (b - 1) * np.log(1 - x**a)


# RealNVP implemented by Jakub M. Tomczak
class RealNVP(nn.Module):
    def __init__(self, nets, nett, num_flows, prior, D=2, dequantization=True):
        super(RealNVP, self).__init__()
        
        self.dequantization = dequantization
        
        self.prior = prior
        self.t = torch.nn.ModuleList([nett() for _ in range(num_flows)])
        self.s = torch.nn.ModuleList([nets() for _ in range(num_flows)])
        self.num_flows = num_flows
        self.D = D
        
    def set_prior(self, prior):
        self.prior = prior
    
    def pad_to_even(self, x):
        if x.shape[1] % 2 != 0:
            # Padding one dimension with 0 to make it even
            padding = (0, 1)  # Pad on the right side along dim=1
            x = torch.nn.functional.pad(x, padding, mode='constant', value=0)
        return x    

    def coupling(self, x, index, forward=True):
        # x: input, either images (for the first transformation) or outputs from the previous transformation
        # index: it determines the index of the transformation
        # forward: whether it is a pass from x to y (forward=True), or from y to x (forward=False)
       
        (xa, xb) = torch.chunk(x, 2, 1)
        
        s = self.s[index](xa)
        t = self.t[index](xa)
        
        
        if forward:
            #yb = f^{-1}(x)
            yb = (xb - t) * torch.exp(-s)
        else:
            #xb = f(y)
            yb = torch.exp(s) * xb + t
        
        return torch.cat((xa, yb), 1), s

    def permute(self, x):
        return x.flip(1)

    def f(self, x):
        x = self.pad_to_even(x)
        log_det_J, z = x.new_zeros(x.shape[0]), x
        
        for i in range(self.num_flows):

            z, s = self.coupling(z, i, forward=True)
        
            z = self.permute(z)
            log_det_J = log_det_J - s.sum(dim=1)
        
        return z, log_det_J
    
    def log_prob(self, x):
        z, log_det_J = self.f(x)
        return self.prior.log_prob(z) + log_det_J


    def f_inv(self, z):
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x, _ = self.coupling(x, i, forward=False)

        return x

    def forward(self, x, reduction='avg'):
        z, log_det_J = self.f(x)
        
        if reduction == 'sum':
            return -(self.prior.log_prob(z) + log_det_J).sum()
        else:
            return -(self.prior.log_prob(z) + log_det_J).mean()

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, self.D))
        z = z[:, 0, :]
        x = self.f_inv(z)
        return x.view(-1, self.D)


# calcu pearson correlation between x and y, but y is already norm
def torch_pearsonr_fix_y(x, y, dim=1):
    x = x - torch.mean(x, dim=dim)[:,None]
    #y = y - torch.mean(y, dim=dim)[:,None]
    x = x / (torch.std(x, dim=dim) + 1e-9)[:,None]
    #y = y / (torch.std(y, dim=dim) + 1e-9)[:,None]
    return torch.mean(x * y, dim=dim)  # (D,)



# Neural Network for p(x,t)

class DensityModel(nn.Module):
    def __init__(self, dim, num_flows =8, M=256):
        super(DensityModel, self).__init__()
        block_dim = dim // 2 
        block_dim = block_dim + 1 if dim % 2 != 0 else block_dim 
        
        # scale (s) network
        nets = lambda: nn.Sequential(nn.Linear(block_dim, M), nn.LeakyReLU(),
                             nn.Linear(M, M), nn.LeakyReLU(),
                             nn.Linear(M, block_dim), nn.Tanh())

        # translation (t) network
        nett = lambda: nn.Sequential(nn.Linear(block_dim, M), nn.LeakyReLU(),
                             nn.Linear(M, M), nn.LeakyReLU(),
                             nn.Linear(M, block_dim))

        self.dim = dim
        # Prior (a.k.a. the base distribution): Gaussian
        prior = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        # Init RealNVP
        self.px = RealNVP(nets, nett, num_flows, prior, D=dim, dequantization=False)
        self.ptx = nn.Sequential(
            nn.Linear(dim + 1 if dim % 2 != 0 else dim , 64),
            nn.CELU(),
            nn.Linear(64,64),
            nn.CELU(),
            nn.Linear(64, 2),
        )
        


    def to(self, device):
        prior = torch.distributions.MultivariateNormal(torch.zeros(self.dim).to(device), torch.eye(self.dim).to(device))
        self.px.set_prior(prior)
        return super().to(device)
    
    def pearson_loss(self, x_noise, x_neigh, y,  reduction='avg', corr_cutoff=0.3):
        n_neighbors = x_neigh.shape[1]
        expectation_center = self.sample_t_given_x(x_noise).flatten()

        expectation_nn = self.sample_t_given_x(x_neigh.reshape(-1, x_neigh.shape[-1]))
        delta_t = expectation_nn.reshape(x_noise.shape[0], n_neighbors) - expectation_center[:,None]
                    
        corr = torch_pearsonr_fix_y(delta_t, y)
        
        mask = corr > corr_cutoff
        corr[mask] = 0.

        if sum(mask) == len(corr):
            return torch.tensor([0.]), 0.
        if reduction == 'avg':
            return -corr.sum() / (len(corr) - sum(mask)), sum(mask) / len(corr)
        else:
            return -corr.sum(), sum(mask)
    
    def sample_t_given_x(self, x):
        z, _ = self.px.f(x)
        pt_x = self.ptx(z)
        pt_x_a, pt_x_b = pt_x[:,0][:,None], pt_x[:,1][:,None]
        return normal_sample(pt_x_a, pt_x_b)
        

        
        
    
    def log_prob_t_x(self, x, t, reduction='sum'):
        z, _ = self.px.f(x)
        
        pt_x = self.ptx(z)
        pt_x_a, pt_x_b = pt_x[:,0][:,None], pt_x[:,1][:,None]
        
        log_pt_x = normal_log_likelihood(t, pt_x_a, pt_x_b).flatten()
        if reduction == 'avg':
            return log_pt_x.mean()
        elif reduction == 'sum':
            return log_pt_x.sum()
        else:
            return log_pt_x


    def var_t_given_x(self, x):
        z, _ = self.px.f(x)
        pt_x = self.ptx(z)
        pt_x_a, pt_x_b = pt_x[:,0][:,None], pt_x[:,1][:,None]
        return F.softplus(pt_x_b)
        
        
    def estimate_t(self, x, mode='mle'):
        z, _ = self.px.f(x)
        pt_x = self.ptx(z)
        pt_x_a, pt_x_b = pt_x[:,0][:,None], pt_x[:,1][:,None]
        
        if mode == 'mle':
            pt_x_a, pt_x_b = pt_x_a.numpy().flatten(), pt_x_b.numpy().flatten()
            return np.array([minimize_scalar(lambda x: -log_likelihood(x, pt_x_a[i], pt_x_b[i]), bounds=(0, 1), method='bounded').x for i in tqdm(range(len(pt_x_a)))])
        return pt_x_a
        
    def log_prob_x(self, x, reduction='sum'):
        log_px = self.px.log_prob(x)
        if reduction == 'avg':
            return log_px.mean()
        elif reduction == 'sum':
            return log_px.sum()
        else:
            return log_px
        
    def joint_log_prob_xt(self, x, t, reduction='sum'):
        z, log_det_J = self.px.f(x)
        log_px = self.px.prior.log_prob(z) + log_det_J
        pt_x = self.ptx(z)
        pt_x_a, pt_x_b = pt_x[:,0][:,None], pt_x[:,1][:,None]
        
        log_pt_x = normal_log_likelihood(t, pt_x_a, pt_x_b).flatten()
        
        
        if reduction == 'avg':
            return (log_px + log_pt_x).mean()
        elif reduction == 'sum':
            return (log_px + log_pt_x).sum()
        else:
            return log_px + log_pt_x


def cosine(a, b):
    return np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1))

 
def get_pair_wise_neighbors(X, n_neighbors=30):
    """Compute nearest neighbors 
    
    Parameters
    ----------
        X: all cell embedding (n, m)
        n_neighbors: number of neighbors

    Returns
    -------
        nn_t_idx: neighbors index (n, n_neighbors)

    """
    N_cell = X.shape[0]
    dim = X.shape[1]
    
    if N_cell < 3000:
        ori_dist = pairwise_distances(X, X)
        nn_t_idx = np.argsort(ori_dist, axis=1)[:, 1:n_neighbors]
    else:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
        _, nn_t_idx = nbrs.kneighbors(X)

    
    return nn_t_idx


class ProbabilityModel:
    """Probability model for pseudotime estimation

    
   Example:
    ----------
    
    ::

        #Assume the velocity are already fitted in pca space
        embedding_key = 'X_pca' 
        velocity_key = 'velocity_pca'

        # Fit the probability model
        pm = pygot.tl.analysis.ProbabilityModel()
        history = pm.fit(adata,  embedding_key=embedding_key, velocity_key=velocity_key)

        # Estimated pseudotime of cells 
        adata.obs['pseudotime'] = pm.estimate_pseudotime(adata) # pseudotime
        adata.obs['var'] = pm.estimate_variance(adata) # variance of time

    """
    def __init__(self,  device=None):
        """Init model

        Arguments:
        ---------
        device: :class:`~torch.device`
            torch device
        
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        print('Device:', device)
        
    def to(self, device):
        self.density_model.to(device)
        self.device = device
        
    def fit(
            self, 
            adata, 
            embedding_key, 
            velocity_key, 
            n_neighbors=30,  
            n_iters=500, 
            mini_batch=True, 
            batch_size = 512, 
        ):
        """fit model

        
        Arguments:
        ---------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        embedding_key: `str` (default: None)
            Name of latent space, in adata.obsm. 
        velocity: `str` (default: None)
            Name of latent velocity, in adata.obsm. Use to do variantional inference of conditonal time distribution if it offers.
        time_key: `str` (default: None)
            Name of time label, in adata.obs. Use as addition information for conditonal time distribution fitting if it offers.
        n_neighbors: `int` (default: 30)
            Number of neighbors of cell
        n_iters: `float` (default: 500)
            Number of training iterations           
        mini_batch: `bool` (default: True)
            Use mini-batch training or not
        batch_size: `int` (default: 512)
            Number of batch size

        """

        self.density_model = DensityModel(adata.obsm[embedding_key].shape[1]).to(self.device)
        self.x = torch.tensor(adata.obsm[embedding_key], requires_grad=True).float().to(self.device)
        if 'distances' in adata.obsp.keys():
            self.nn_t_idx = find_neighbors(adata.obsp['distances'], directed=True)
            
        else:
            self.nn_t_idx = get_pair_wise_neighbors(adata.obsm[embedding_key], n_neighbors=n_neighbors)
        self.v_hat = adata.obsm[embedding_key][self.nn_t_idx.flatten()].reshape(self.nn_t_idx.shape[0], self.nn_t_idx.shape[1], -1) - adata.obsm[embedding_key][:,None, :]
        self.velocity_key = velocity_key
        self.embedding_key = embedding_key
        
        if len(adata) > 5000 and self.device == 'cpu' and mini_batch == False:
            print('Large dataset and cpu device. Suggest to use mini-batch')
        
        with torch.no_grad():

            v = adata.obsm[self.velocity_key]
            norm_cos_sim = (v[:,None,:] * self.v_hat)

            norm_cos_sim = norm_cos_sim.sum(axis=-1) / (np.linalg.norm(v, axis=1) ** 2)[:,None]
            norm_cos_sim = torch.tensor(norm_cos_sim).to(self.device)
            
        optimizer = torch.optim.Adamax(self.density_model.parameters(), lr=1e-3)

        pbar = tqdm(range(n_iters))
        for i in pbar:
            if not mini_batch:
                batch_idx = list(range(len(self.x)))
                sample_x = self.x
            else:
                batch_idx = np.random.choice(range(len(adata)), size=batch_size, replace=False)
                sample_x = self.x[batch_idx]
               
            x_noise  = sample_x + torch.randn_like(sample_x) * 0.05
            
            
            sub_nn_t_idx = self.nn_t_idx[batch_idx]
            sample_idx = np.unique(sub_nn_t_idx)
            mapper = (np.ones(len(self.x)) * -1).astype(int)
            mapper[sample_idx] = range(len(sample_idx))
            mapper[sub_nn_t_idx.flatten()]
            
            
            expectation_center = self.density_model.sample_t_given_x(x_noise).flatten()
            expectation_nn = self.density_model.sample_t_given_x(self.x[sample_idx]).flatten()
            
            delta_t = expectation_nn[mapper[sub_nn_t_idx.flatten()]].reshape(sub_nn_t_idx.shape[0], sub_nn_t_idx.shape[1]) - expectation_center[:,None]
            
            taylor_loss = torch.mean((delta_t - norm_cos_sim[batch_idx])**2)
                
            
            loss = taylor_loss
            
            
            pbar.set_description("Taylor Loss {:.4f}".format(loss.item()))
            
            optimizer.zero_grad()
            
            if not (loss.grad_fn is None):
                loss.backward()
                optimizer.step()
            

        del self.x
        del self.nn_t_idx
        del self.v_hat
        
            
            
        
    @torch.no_grad()
    def estimate_pseudotime(self, adata, mode='mean'):
        """estimate the pseudotime

        Arguments:
        ---------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        Returns
        -------
        :math:`t^*|x`: :class:`~np.ndarray`
            pseudotime of cells

        """
        
        expectation = self.density_model.estimate_t(torch.tensor(adata.obsm[self.embedding_key].copy()).float().to(self.device), mode=mode)
        if isinstance(expectation, torch.Tensor):
            expectation = expectation.detach().cpu().numpy()
        
        return expectation

    @torch.no_grad()
    def estimate_variance(self, adata):
        """estimate the variance of pseudotime

        Arguments:
        ---------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        
        Returns
        -------
        var: :class:`~np.ndarray`
            variance of cell time

        """
        
        var = self.density_model.var_t_given_x(torch.tensor(adata.obsm[self.embedding_key].copy()).float().to(self.device)).detach().cpu().numpy()
        return var
