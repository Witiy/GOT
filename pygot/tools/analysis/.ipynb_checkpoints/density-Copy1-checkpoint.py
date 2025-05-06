import torch
import torch.nn as nn
import torch.autograd
import hnswlib
from sklearn.metrics import pairwise_distances
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from functools import partial
from scipy.optimize import minimize_scalar
from torch.distributions.normal import Normal
import torch.nn.functional as F
from torch import distributions
import math
import dcor


def dcor_test(adata, pseudotime_key, num_resamples=1):
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
    # 获取唯一的字符串值，并为每个唯一值分配一个整数
    unique_strings = list(set(string_list))
    string_to_index = {s: idx for idx, s in enumerate(unique_strings)}
    
    # 将字符串列表转换为对应的整数列表
    integer_list = [string_to_index[s] for s in string_list]
    
    # 转换为 PyTorch 张量
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


class SSLGaussMixture(torch.distributions.Distribution):

    def __init__(self, k, d, inv_cov_stds=None, device=None):
        
        
        self.means = torch.randn(k, d, device=device)
        self.n_components, self.d = self.means.shape
        if inv_cov_stds is None:
            self.inv_cov_stds = math.log(math.exp(1.0) - 1.0) * torch.ones((len(self.means)), device=device)
        else:
            self.inv_cov_stds = inv_cov_stds

        self.weights = torch.ones((len(self.means)), device=device)
        self.device = device
    def to(self, device):
        self.means = self.means.to(device)
        self.inv_cov_stds = self.inv_cov_stds.to(device)
        self.weights = self.weights.to(device)
        self.device = device
        return self
        
    @property
    def gaussians(self):
        gaussians = [distributions.MultivariateNormal(mean, F.softplus(inv_std)**2 * torch.eye(self.d).to(self.device))
                          for mean, inv_std in zip(self.means, self.inv_cov_stds)]
        return gaussians


    def parameters(self):
       return [self.means, self.inv_cov_std, self.weights]
        
    def sample(self, sample_shape, gaussian_id=None):
        if gaussian_id is not None:
            g = self.gaussians[gaussian_id]
            samples = g.sample(sample_shape)
        else:
            n_samples = sample_shape[0]
            idx = np.random.choice(self.n_components, size=(n_samples, 1))
            all_samples = [g.sample(sample_shape) for g in self.gaussians]
            samples = all_samples[0]
            for i in range(self.n_components):
                mask = np.where(idx == i)[0]
                samples[mask] = all_samples[i][mask]
        return samples
        
    def log_prob(self, x, y=None, label_weight=1.):
        
        all_log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        mixture_log_probs = torch.logsumexp(all_log_probs + torch.log(F.softmax(self.weights)), dim=1)
        if y is not None:
            log_probs = torch.zeros_like(mixture_log_probs)
            mask = (y == -1)
            log_probs[mask] += mixture_log_probs[mask]
            for i in range(self.n_components):
                #Pavel: add class weights here? 
                mask = (y == i)
                log_probs[mask] += all_log_probs[:, i][mask] * label_weight
            return log_probs
        else:
            return mixture_log_probs

    def class_logits(self, x):
        log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        log_probs_weighted = log_probs + torch.log(F.softmax(self.weights))
        return log_probs_weighted

    def classify(self, x):
        log_probs = self.class_logits(x)
        return torch.argmax(log_probs, dim=1)

    def class_probs(self, x):
        log_probs = self.class_logits(x)
        return F.softmax(log_probs, dim=1)

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

    def forward(self, x, y=None, reduction='avg'):
        z, log_det_J = self.f(x)
        
        if reduction == 'sum':
            return -(self.prior.log_prob(z, y) + log_det_J).sum()
        else:
            return -(self.prior.log_prob(z, y) + log_det_J).mean()

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
    def __init__(self, dim, k=1, num_flows =8, M=256):
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
        #prior = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        prior = SSLGaussMixture(k, dim)
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
    #    #prior = torch.distributions.MultivariateNormal(torch.zeros(self.dim).to(device), torch.eye(self.dim).to(device))
        self.px.set_prior(self.px.prior.to(device))
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
        #z, _ = self.px.f(x)
        pt_x = self.ptx(x)
        pt_x_a, pt_x_b = pt_x[:,0][:,None], pt_x[:,1][:,None]
        
        sample_t = []
        
        for _ in range(10):
            sample_t.append(normal_sample(pt_x_a, pt_x_b))
            
        return torch.stack(sample_t).mean(dim=0)
        
        
    
    def log_prob_t_x(self, x, t, reduction='sum'):
        #z, _ = self.px.f(x)
        pt_x = self.ptx(x)
        pt_x_a, pt_x_b = pt_x[:,0][:,None], pt_x[:,1][:,None]
        
        log_pt_x = normal_log_likelihood(t, pt_x_a, pt_x_b).flatten()
        if reduction == 'avg':
            return log_pt_x.mean()
        elif reduction == 'sum':
            return log_pt_x.sum()
        else:
            return log_pt_x


    def var_t_given_x(self, x):
        #z, _ = self.px.f(x)
        pt_x = self.ptx(x)
        pt_x_a, pt_x_b = pt_x[:,0][:,None], pt_x[:,1][:,None]
        return F.softplus(pt_x_b)
        
        
    def estimate_t(self, x):
        #z, _ = self.px.f(x)
        pt_x = self.ptx(x)
        pt_x_a, pt_x_b = pt_x[:,0][:,None], pt_x[:,1][:,None]
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
        pt_x = self.ptx(x)
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
        p = hnswlib.Index(space="l2", dim=dim)
        p.init_index(max_elements=N_cell, ef_construction=200, M=30)
        p.add_items(X)
        p.set_ef(n_neighbors + 10)
        nn_t_idx = p.knn_query(X, k=n_neighbors)[0][:, 1:].astype(int)
    return nn_t_idx




class ProbabilityModel:
    """Probability model for density and pseudotime estimation

    It parametrically modelling the joint distribution of :math:`\log{P(x,t)} = \log{P(x)} + \log{P(t|x)}`. 
    To modelling :math:`\log{P(x)}`, it use `RealNVP` (one of normalizing flow model) to estimate the density.
    To modelling :math:`\log{P(t|x)}`, it use pearsonr correlation between conditional expectation :math:`\mathbb{E}_{p(t|x)}(t)` 
    and velocity :math:`v(x)` among neighbors :math:`N(x)` to fit the probabilistic neural network of conditional distribution :math:`t|x`.
    
    Model fitting can be divided into two part, one for :math:`\log P(x)` and another for :math:`\log P(t|x)`.

    To fit :math:`\log P(x)`, we use normalizing flow model RealNVP to modelling 
    It transform the distribution of :math:`P(x)` into :math:`P(z), z \\thicksim \mathcal{N}(z|0, I)`  
    by invertible neural network :math:`\\theta(x)`. The density can be analytical solved as

    .. math::

        P(x) = \mathcal{N}(z=\\theta(x)|0, I)\det{\\frac{\partial \\theta(x)}{\partial x}}

    And the loss function is negative likelihood

    .. math::

        L_{marginal}(x_i|\\theta) = -\log{P(x)}


    The detail of model is in the original paper :cite:p:`dinh2016density`.

        
    To fit :math:`\log P(t|x)`, here, we frist assume the conditional time distribution is 

    .. math::

        t|x \\thicksim \mathcal{N} (\mu | x, \sigma | x)

    And we use probabilistic neural network :math:`\\theta_{\mu}(x), \\theta_{\sigma}(x)` to
    estimate cell-dependent parameters :math:`\mu|x, \sigma|x`.
        
    Then, the similarities between velocity and neighbors transition is computed as 
    :math:`\Pi_{x_i}^{v} = \{\cos(v(x_i), x_j - x_i)|j \in N(x)\}`, and the time similarities is 
    computed as :math:`\Pi_{x_i}^{t} = \{\mathbb{E}_{p(t|x_j)}(t) - \mathbb{E}_{p(t|x_i)}(t)|j \in N(x)\}`.
    we use reparameterization trick to estimated the expectation :math:`\mathbb{E}_{p(t|x_j)}(t) = \mathbb{E}_{p(\epsilon)}(\mu + \sigma \epsilon), \epsilon \\thicksim \mathcal{N} (0, I)`.
    The conditional time expectation should be relevant to velocity, so pearsonr correlation loss is used 
    to train :math:`\\theta_{\mu}(x), \\theta_{\sigma}(x)`.
        
    .. math::

        L_{conditional}(x_i|\\theta_{\mu}, \\theta_{\sigma}) = \\frac{cov(\Pi_{x_i}^{v}, \Pi_{x_i}^{t})}{\sigma(\Pi_{x_i}^{v})\sigma(\Pi_{x_i}^{t})}
    
   Example:
    ----------
    
    ::

        #Assume the velocity are already fitted in pca space
        embedding_key = 'X_pca' 
        velocity_key = 'velocity_pca'

        # Fit the probability model
        pm = pygot.tl.analysis.ProbabilityModel()
        history = pm.fit(adata,  embedding_key=embedding_key, velocity_key=velocity_key, n_epoch=5, corr_cutoff=0.3)

        # Estimated the density and pseudotime of cells 
        adata.obs['log_px'] = pm.log_prob_x(adata, bound=True) # log density
        adata.obs['pseudotime'] = pm.estimate_pseudotime(adata) # pseudotime
        adata.obs['var'] = pm.estimate_variance(adata) # variance of time

    """
    def __init__(self,  device=None):
        """Init model

        Arguments:
        ---------`
        device: :class:`~torch.device`
            torch device
        
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        print('Device:', device)
        self.density_model = None
    def to(self, device):
        self.density_model.to(device)
        self.device = device
        
    def fit(
            self, 
            adata, 
            embedding_key, 
            velocity_key,
            cell_type_key=None, 
            n_neighbors=30,  
            corr_cutoff=0.3, 
            n_iters=1000, 
            marginal=False, 
            conditional=True, 
            batch_size = 512, 
            lam = 1.
        ):
        """fit model

        Model fitting can be divided into two part, one for :math:`\log P(x)` and another for `\log P(t|x)`.

        To fit :math:`\log P(x)`, we use normalizing flow model RealNVP to modelling 
        It transform the distribution of :math:`P(x)` into :math:`P(z), z \\thicksim \mathcal{N}(z|0, I)`  
        by invertible neural network :math:`\\theta(x)`. The density can be analytical solved as

        .. math::

            P(x) = \mathcal{N}(z=\\theta(x)|0, I)\det{\\frac{\partial \\theta(x)}{\partial x}}

        And the loss function is negative likelihood

        .. math::

            L_{marginal}(x_i|\\theta) = -\log{P(x)}


        The detail of model is in the original paper :cite:p:`dinh2016density`.

        
        To fit :math:`\log P(t|x)`, here, we frist assume the conditional time distribution is 

        .. math::

            t|x \\thicksim \mathcal{N} (\mu | x, \sigma | x)

        And we use probabilistic neural network :math:`\\theta_{\mu}(x), \\theta_{\sigma}(x)` to
        estimate cell-dependent parameters :math:`\mu|x, \sigma|x`.
        
        Then, the similarities between velocity and neighbors transition is computed as 
        :math:`\Pi_{x_i}^{v} = \{\cos(v(x_i), x_j - x_i)|j \in N(x)\}`, and the time similarities is 
        computed as :math:`\Pi_{x_i}^{t} = \{\mathbb{E}_{p(t|x_j)}(t) - \mathbb{E}_{p(t|x_i)}(t)|j \in N(x)\}`.
        we use reparameterization trick to estimated the expectation :math:`\mathbb{E}_{p(t|x_j)}(t) = \mathbb{E}_{p(\epsilon)}(\mu + \sigma \epsilon), \epsilon \\thicksim \mathcal{N} (0, I)`.
        The conditional time expectation should be relevant to velocity, so pearsonr correlation loss is used 
        to train :math:`\\theta_{\mu}(x), \\theta_{\sigma}(x)`.
        
        .. math::

            L_{conditional}(x_i|\\theta_{\mu}, \\theta_{\sigma}) = \\frac{cov(\Pi_{x_i}^{v}, \Pi_{x_i}^{t})}{\sigma(\Pi_{x_i}^{v})\sigma(\Pi_{x_i}^{t})}
            

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
        cor_cutoff: `float` (default: 0.3)
            Cutoff of correlation, if correlation beyond cutoff, :math:`L_{conditional}(x_i)` will not be optimized 
        n_iters: `int` (default: 1000)
            Number of training iterations
        marginal: `bool` (default: True)
            Train mariginal distribution, i.e. log P(x)
        conditional: `bool` (default: True)
            Train conditional distribution, i.e. log P(t|x)
        batch_size: `int` (default: 256)
            Number of batch size
        lam: `float` (default: 1.)
            Loss = lam * L_marginal + L_conditonal
        
        """

        assert conditional or marginal 


        
        x = torch.tensor(adata.obsm[embedding_key], requires_grad=True).float().to(self.device)
        nn_t_idx = get_pair_wise_neighbors(adata.obsm[embedding_key], n_neighbors=n_neighbors)
        v_hat = adata.obsm[embedding_key][nn_t_idx.flatten()].reshape(nn_t_idx.shape[0], nn_t_idx.shape[1], -1) - adata.obsm[embedding_key][:,None, :]
        self.velocity_key = velocity_key
        self.embedding_key = embedding_key
        self.cell_type_key = cell_type_key
        density_history = []
        
        if not cell_type_key is None:
            y, self.mapping = strings_to_tensor(adata.obs[cell_type_key].tolist())
            y = y.to(self.device)
            k = len(self.mapping)
        else:
            y = None
            k = 1
        
        with torch.no_grad():

            v = adata.obsm[self.velocity_key]
            cos_sim = cosine(v[:,None,:], v_hat)
            cos_sim = torch.tensor(cos_sim)
            norm_cos_sim = cos_sim - torch.mean(cos_sim, dim=1)[:,None]
            norm_cos_sim = norm_cos_sim / (torch.std(norm_cos_sim, dim=1) + 1e-9)[:,None]
            norm_cos_sim = norm_cos_sim.to(self.device)

        if self.density_model is None:
            self.density_model = DensityModel(adata.obsm[embedding_key].shape[1], k=k).to(self.device)

        optimizer = torch.optim.Adamax(self.density_model.parameters(), lr=1e-3)
        pbar = tqdm(range(n_iters))

        self.density_model.train()
        for i in pbar:
            batch_idx = np.random.choice(range(len(adata)), size=batch_size, replace=False)
            sample_x = x[batch_idx]
            sample_norm_cos = norm_cos_sim[batch_idx]
            sample_y = y[batch_idx] if cell_type_key else None
        
            x_noise  = sample_x + torch.randn_like(sample_x) * 0.05
            if conditional:
                sub_nn_t_idx = nn_t_idx[batch_idx]
                sample_idx = np.unique(sub_nn_t_idx)
                mapper = (np.ones(len(x)) * -1).astype(int)
                mapper[sample_idx] = range(len(sample_idx))
                mapper[sub_nn_t_idx.flatten()]

                expectation_center = self.density_model.sample_t_given_x(x_noise).flatten()
                expectation_nn = self.density_model.sample_t_given_x(x[sample_idx]).flatten()
                #print(expectation)
                delta_t = expectation_nn[mapper[sub_nn_t_idx.flatten()]].reshape(sub_nn_t_idx.shape[0], sub_nn_t_idx.shape[1]) - expectation_center[:,None]
                
                corr = torch_pearsonr_fix_y(delta_t, sample_norm_cos)
                
                mask = corr < corr_cutoff
                
                if torch.sum(mask) > 0:
                    corr = torch.mean(corr[mask])
                else:
                    corr = torch.tensor(0.)

            if marginal:
                density_loss = self.density_model.px(x_noise, sample_y)
                density_history.append(density_loss.item())

            loss = 0
            loss += lam*density_loss if marginal else 0
            loss -= corr if conditional else 0
            
            #pbar.set_description("Density Loss {:.4f}".format(density_loss.item()))
            a = density_loss.item() if marginal else 0.
            b = corr.item() if conditional else 0.
            c = (1 - torch.sum(mask).item() / len(mask))*100 if conditional else 0.
            pbar.set_description("Density Loss {:.4f}, Corr {:.4f} Satisfied {:2f}%".format(a, 
                                                                                            b,  
                                                                                            c))
            
            optimizer.zero_grad()
            
            if not (loss.grad_fn is None):
                loss.backward()
                optimizer.step()
            '''
            if conditional:
                if (1 - torch.sum(mask).item() / len(mask)) > 0.95 and i > 50:
                    if marginal == False:
                        break
                    conditional = False
            '''
        self.density_model.eval()
        


                
        
    @torch.no_grad()
    def estimate_pseudotime(self, adata):
        """estimate the pseudotime

            The time of cells :math:`t, t|x \\thicksim \mathcal{N} (\mu, \sigma)`. To obtained the pseudotime of cell :math:`t^*|x`,
            expectation time :math:`t^*|x = \mathbb{E}[t|x]=\mu 
            are inplemented.
        
        See Also:
        ---------
        ProbabilityModel.estimate_variance :  :meth:`ProbabilityModel.estimate_variance`
            This function estimate variance of time of cells.


        Arguments:
        ---------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        Returns
        -------
        :math:`t^*|x`: :class:`~np.ndarray`
            pseudotime of cells

        """
        
        expectation = self.density_model.estimate_t(torch.tensor(adata.obsm[self.embedding_key].copy()).float().to(self.device))
        if isinstance(expectation, torch.Tensor):
            expectation = expectation.detach().cpu().numpy()
        
        return expectation

    @torch.no_grad()
    def estimate_variance(self, adata):
        """estimate the variance of pseudotime

            The time of cells :math:`t, t|x \\thicksim Kumaraswamy(a, b)`, the variance of Kumaraswamy is given by
            :math:`Var(t|x) = \\frac{b\Gamma(1+\\frac{2}{a}) \Gamma(b)}{\Gamma(1+\\frac{2}{a}+b)}-(\\frac{b}{a+b})^2` 

        See Also:
        ---------
        ProbabilityModel.estimate_pseudotime :  :meth:`ProbabilityModel.estimate_pseudotime`
            This function estimate expectation of time of cells.



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

    @torch.no_grad()
    def log_prob_x(self, adata, bound=False, lower_bound=None):
        """compute probability of :math:`log P(x)`

        Arguments:
        ---------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        lower_bound: `float` (default: -1000)
            lower bound of :math:`log P(x)`

        Returns
        -------
        :math:`log P(x)`: :class:`~np.ndarray`
            log probability of x

        """
        log_px = self.density_model.log_prob_x(torch.tensor(adata.obsm[self.embedding_key].copy()).float().to(self.device), reduction=None).detach().cpu().numpy()
        
        if bound:
            return std_bound(log_px)
        if not lower_bound is None:
            log_px[log_px < lower_bound] = lower_bound
        return log_px
    
    @torch.no_grad()
    def log_prob_x_t(self, adata, t, lower_bound=None):
        """compute joint probability of :math:`log P(x, t)`

            joint probability is given by :math:`\log{P(x|t)}=\log{P(t|x)} + \log{P(x)}`

        Arguments:
        ---------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        t: `float`
            time
        lower_bound: `float` (default: -1000)
            lower bound of :math:`log P(x, t)`

        Returns
        -------
        :math:`log P(x, t)`: :class:`~np.ndarray`
            log joint probability of (x, t)

        """
        if isinstance(t, torch.Tensor):
            if len(t.shape) == 1:
                t = t[:,None].to(self.device)
        else:
            t =  (torch.ones(size=(len(adata), 1)) * t).to(self.device)
        
        log_pxt = self.density_model.joint_log_prob_xt(torch.tensor(adata.obsm[self.embedding_key].copy()).to(self.device), t, reduction=None).detach().cpu().numpy()
        if not lower_bound is None:
            log_pxt[log_pxt < lower_bound] = lower_bound
        
        return log_pxt

    @torch.no_grad()
    def log_prob_t_given_x(self, adata, t, lower_bound=None):
        """compute conditional probability of :math:`log P(t|x)`

        See Also:
        ---------
        ProbabilityModel.fit :  :meth:`ProbabilityModel.fit`
            This function describe how to fit the density and time of cells.

        Arguments:
        ---------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        t: `float`
            time
        lower_bound: `float` (default: -1000)
            lower bound of :math:`log P(t|x)`

        Returns
        -------
        :math:`log P(t|x)`: :class:`~np.ndarray`
            log probability of t given x

        """
        if isinstance(t, torch.Tensor):
            if len(t.shape) == 1:
                t = t[:,None].to(self.device).float()
            
        else:
            t =  (torch.ones(size=(len(adata), 1)) * t).to(self.device).float()
        
        log_pt_x = self.density_model.log_prob_t_x(torch.tensor(adata.obsm[self.embedding_key].copy()).float().to(self.device), t, reduction=None).detach().cpu().numpy()
        if not lower_bound is None:
            log_pt_x[log_pt_x < lower_bound] = lower_bound
        
        return log_pt_x