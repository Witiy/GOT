import torch
from functools import partial
from .v_centric_training import v_centric_training, GraphicalOTVelocitySampler
from .x_centric_training import x_centric_training


class ODEwrapper(torch.nn.Module):
    def __init__(self, func):
        super(ODEwrapper, self).__init__()
        self.func = func
        


    def forward(self, t, x): #NOTE the forward pass when we use torchdiffeq must be forward(self,t,x)
        if self.func.time_varying:
            if len(t.size()) == 0:
                time = t.repeat(x.size()[0],1)
            else:
                time = t
        
            dxdt = self.func(torch.concat([x, time], dim=-1).float())
        else:
            dxdt = self.func(x)
        return dxdt
    
    
    def compute_G(self, t, x):
        if self.func.time_varying:
            if len(t.size()) == 0:
                time = t.repeat(x.size()[0],1)
            else:
                time = t
            G = self.func.compute_G(torch.concat([x, time], dim=-1).float())
        else:
            G = self.func.compute_G(x).float()
        return G

class ODEwrapperNoTime(torch.nn.Module):
    def __init__(self, func):
        super(ODEwrapperNoTime, self).__init__()
        self.func = func


    def forward(self, t, x): #NOTE the forward pass when we use torchdiffeq must be forward(self,t,x)
        dxdt = self.func(x)
        return dxdt

class MLP(torch.nn.Module):
    def __init__(self, dim, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), 16, bias=False),
            torch.nn.CELU(),
            torch.nn.Linear(16, 32, bias=False),
            torch.nn.CELU(),
            torch.nn.Linear(32, 16, bias=False),
            torch.nn.CELU(),
            torch.nn.Linear(16, dim, bias=False),
            #JAC(out_dim)
        )

    def forward(self, x):
        return self.net(x) 
    



def fit_velocity_model(
        adata, time_key, embedding_key, 
        landmarks=False,
        device=None,
        graph_key=None,
        dim=None,
        n_neighbors=50,
        v_centric_iter_n=1000, v_centric_batch_size=256, 
        add_noise=True, sigma=0.1,
        neighbor_sampling = False,
        lr=5e-3, 
        path='',
        linear=False,
        distance_metrics='SP',
        pretrained_model=None,
        x_centric=True,
        x_centric_iter_n=2000,
        x_centric_batch_size=256,
        reverse_schema=True,
        graph_inter=False,
        randomized=False,
        time_varying=True,
        **kwargs
        ):
    """Estimates velocities and fit trajectories in latent space.

    Based optimal transport and time label information, the function 
    use two-stage training strategy to fit the neural network model.
    First, it use optimal transport to sample the most prossible pairs 
    between two time point, then sample the velocity by interpolation 
    in the shortest path, which call `v-centric` training. Second, it use 
    NeuralODE to simulate trajectory from start to end (e.g. day0 -> day7), 
    and minize the distribution difference between simulation distribution 
    and empirical distribution.

    

    Arguments:
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    time_key: `str`
        Name of time label in adata.obs
    embedding_key: `str'
        Name of latent space to fit, in adata.obsm
    landmarks: `bool`
        Use landmarks to approximate graphical paths
    device: :class:`~torch.device`
        torch device
    dim: `int` (default: None)
        Dimension number of latent space (using all components by default)
    n_neighbors: `int` (default: 50)
        Neighbors number in kNN
    v_centric_iter_n: `int` (default: 1000)
        Iteration number of v-centric training
    v_centric_batch_size: `int` (default: 256)
        Batch size of v-centric training. Note: increase will dramatically increase training time due to the complexity of OT
    add_noise: `bool` (default: True)
        Assumption of gaussian distribution of velocity
    sigma: `float` (default: 0.1)
        The variance of gaussian distribution of velocity
    neighbor_sampling: `str` (default: False)
        Mini-batching sampling in the kNN graph between two time point
    lr: `float` (default: 5e-3)
        Learning rate
    distance_metrics: 'SP' or 'L2' (default: 'SP')
        Distance metrics in optimal transport (shortest path distance or euclidean distance)
    pretrained_model: :class`~torch.nn.Module` (default: None)
        Training for pretraiened model
    x_centric: `bool` (default: True)
        Do x-centric training
    x_centric_iter_n: `int` (default: 2000)
        Iteration number of x-centric training
    x_centric_batch_size: `int` (default: 256)
        Batch size of x-centric training
    reverse_schema: `bool` (default: True)
        Simulation trajectory from end to start also (e.g. day7 -> day0), in x-centric training
    graph_inter: `bool` (default: False)
        Interpolation between two time points using kNN graph.
    randomized: `bool` (default: False)
        Randomize the kNN graph to interpolation between two time points
    time_varying: `bool` (default: True)
        The neural network model use time label as input or not
    
       
    Returns
    -------
    model: :class`~ODEwrapper`
        velocity model
    history: `list`
        training history
    """
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if dim == None:
        dim = adata.obsm[embedding_key].shape[1]
    adata.obsm[embedding_key] = adata.obsm[embedding_key][:,:dim]
    sp_sampler = GraphicalOTVelocitySampler(
                    adata, time_key, 
                    graph_key=embedding_key if graph_key==None else graph_key, 
                    embedding_key=embedding_key, 
                    landmarks=landmarks,
                    device=device, 
                    path=path, 
                    linear=linear,
                    neighbor_sampling=neighbor_sampling,
                    n_neighbors=n_neighbors,
                    **kwargs
                    
    )
    
    if pretrained_model == None:
        model = ODEwrapper(MLP(dim=dim, time_varying=time_varying)).to(device)
        #MLP(dim=dim,time_varying=True, w=w_dim).to(device)
    else:
        model = pretrained_model

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=0.001)
    sample_fn_path = partial(sp_sampler.filtered_sample_batch_path, sigma=sigma, batch_size=v_centric_batch_size, distance_metrics=distance_metrics, add_noise=add_noise)
    model, history = v_centric_training(model, optimizer, sample_fn_path, iter_n=v_centric_iter_n, device=device)

    if x_centric:

        model, history2 = x_centric_training(adata, time_key, embedding_key, model, 
                                            reverse_schema=reverse_schema,
                                            batch_size=x_centric_batch_size, 
                                            iter_n=x_centric_iter_n, 
                                            graph_inter=graph_inter, 
                                            sp_sampler=sp_sampler, 
                                            distance_metrics=distance_metrics, 
                                            randomized=randomized,
                                            neighbor_sampling=neighbor_sampling,
                                            device=device)
        return model, (history, history2)

    return model, history
