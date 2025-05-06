import scanpy as sc
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pygot.tools.traj.model_training import ODEwrapper, MLP
from .otcfm import _train_ode

import pandas as pd
import time

#########  OT-CFM  ###################
def get_batch(FM, X, batch_size, n_times, device, return_noise=False):
    """Construct a batch with point sfrom each timepoint pair"""
    ts = []
    xts = []
    uts = []
    noises = []
    for t_start in range(n_times - 1):
        x0 = (
            torch.from_numpy(X[t_start][np.random.randint(X[t_start].shape[0], size=batch_size)])
            .float()
            .to(device)
        )
        x1 = (
            torch.from_numpy(
                X[t_start + 1][np.random.randint(X[t_start + 1].shape[0], size=batch_size)]
            )
            .float()
            .to(device)
        )
        if return_noise:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(
                x0, x1, return_noise=return_noise
            )
            noises.append(eps)
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)
    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)
    if return_noise:
        noises = torch.cat(noises)
        return t[:,None], xt, ut, noises
    return t[:,None], xt, ut





def OTCFM_interface(
        adata, time_key, embedding_key, device, 
        iter_n=1000, 
        sigma=0.1,
        batch_size=256, lr=1e-4, 
        model=None
):
    try:
        from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
        
    except ImportError:
        raise ImportError(
                "Please install the OTCFM algorithm: `https://github.com/atong01/conditional-flow-matching`.")
    

    n_times = len(pd.unique(adata.obs[time_key]))
    dim = adata.obsm[embedding_key].shape[1]
    if model == None:
        ot_cfm_model = ODEwrapper(MLP(dim=dim, time_varying=True)).to(device)
        #MLP(dim=dim,time_varying=True, w=w_dim).to(device)
    else:
        ot_cfm_model = model

    ot_cfm_optimizer = torch.optim.Adam(ot_cfm_model.parameters(), lr)
    FM =  ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    X = [
        adata.obsm[embedding_key][adata.obs[time_key] == t]
        for t in np.sort(pd.unique(adata.obs[time_key]))
    ]
    n_times = len(pd.unique(adata.obs[time_key]))
    ode_sample_fn = partial(get_batch, FM=FM, X=X, batch_size=batch_size, n_times=n_times, device=device)
    ot_cfm_model, history = _train_ode(ot_cfm_model, ot_cfm_optimizer, ode_sample_fn, iter_n=iter_n)
    return ot_cfm_model, history




#########  TIGON  ###################


from .tigon import *
class TIGONwrapper(nn.Module):
    def __init__(self, tigon):
        super().__init__()
        self.model = tigon
    def forward(self,x):
        state = x
        ii = 0
        for layer in self.model.hyper_net1.net:
            if ii == 0:
                x = layer(state)
            else:
                x = layer(x)
            ii =ii+1
        x = self.model.hyper_net1.out(x)
        return x

def TIGON_interface(adata, time_key, embedding_key, device,
                    niters=5000, lr=3e-3, num_samples=100, hidden_dim=16,
                    n_hiddens=3, activation='Tanh', max_patience=10):
    
    
    args = Args()
    args.niters,args.lr, args.num_samples, args.hidden_dim, args.n_hiddens, args.activation = niters, lr, num_samples, hidden_dim, n_hiddens, activation

    # load dataset
    integral_time = np.sort(np.unique(adata.obs[time_key]))
    data_train = [
        torch.Tensor(adata.obsm[embedding_key][adata.obs[time_key] == t]).type(torch.float32).to(device)
        for t in integral_time
    ]
    

    time_pts = range(len(data_train))
    leave_1_out = []
    train_time = [x for i,x in enumerate(time_pts) if i!=leave_1_out]


    # model
    func = UOT(in_out_dim=data_train[0].shape[1], hidden_dim=args.hidden_dim,n_hiddens=args.n_hiddens,activation=args.activation).to(device)
    func.apply(initialize_weights)


    # configure training options
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': 0.01})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})

    optimizer = optim.Adam(func.parameters(), lr=args.lr, weight_decay= 0.01)
    lr_adjust = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.niters-400,args.niters-200], gamma=0.5, last_epoch=-1)
    mse = nn.MSELoss()

    LOSS = []
    L2_1 = []
    L2_2 = []
    Trans = []
    Sigma = []
    sigma_now = 1
    early_stopping = False
    t = max_patience
    best_loss = np.inf
    best_model = func.state_dict()
    print('Iteration:', args.niters)
    print('ODE method:',  options['method'])
    for itr in range(1, args.niters + 1):
        if early_stopping:
            break
        optimizer.zero_grad()
            
        loss, loss1, sigma_now, L2_value1, L2_value2 = train_model(mse,func,args,data_train,train_time,integral_time,sigma_now,options,device,itr)

            
        loss.backward()
        optimizer.step()
        lr_adjust.step()

        LOSS.append(loss.item())
        Trans.append(loss1[-1].mean(0).item())
        Sigma.append(sigma_now)
        L2_1.append(L2_value1.tolist())
        L2_2.append(L2_value2.tolist())
            
        print('Iter: {}, loss: {:.4f}'.format(itr, loss.item()))

        flag = LOSS[-1] < best_loss
        if flag:
            t = max_patience
            best_model = func.state_dict()
            best_loss = LOSS[-1]
        else:
            t -= 1
        
        if t == 0:
            early_stopping = True

    func.load_state_dict(best_model)
    return func

#########  MIOFlow  ###################





def MIOFlow_interface(
        adata, embedding_key, time_key, exp_dir, hold_out = 1, batch_size=60, 
        gae_embedded_dim = 32,
        n_local_epochs = 20, n_epochs = 80, n_batches=20, hold_one_out=False,use_density_loss = True,
        use_emb = False,
        use_gae = False):
    try:
        from MIOFlow.utils import generate_steps, config_criterion
        from MIOFlow.models import make_model, Autoencoder
        from MIOFlow.train import train_ae, training_regimen
        from MIOFlow.geo import setup_distance

        
    except ImportError:
        raise ImportError(
                "Please install the MIOFlow algorithm: `https://github.com/KrishnaswamyLab/MIOFlow`."
            )
    df = pd.DataFrame(adata.obsm[embedding_key], columns=['d'+str(i) for i in range(adata.obsm[embedding_key].shape[1])])
    phate_dims = len(df.columns)
    df['samples'] = adata.obs[time_key].tolist()
    use_cuda = torch.cuda.is_available()
    
    # The dimensions in the input space, it is columns - 1 because we assume one column is equal to "samples".
    model_features = len(df.columns) - 1
    
    groups = sorted(df.samples.unique())


    need_to_train_gae = (use_emb or use_gae) and use_emb != use_gae

    # If the reconstruction loss needs to be computed.
    recon = use_gae and not use_emb 

    # These are training GAE hyperparameters needed for training
    # Distance_type in ['gaussian', 'alpha_decay'], and Gaussian scale
    distance_type = 'gaussian'
    rbf_length_scale=0.5
    knn=5
    t_max=5
    dist = setup_distance(distance_type, rbf_length_scale=rbf_length_scale, knn=knn, t_max=t_max)

    #Can be changed depending on the dataset
    n_epochs_emb = 1500
    samples_size_emb = (30, )

    
    encoder_layers = [model_features, 8, gae_embedded_dim]

    gae = Autoencoder(
        encoder_layers = encoder_layers,
        decoder_layers = encoder_layers[::-1],
        activation='ReLU', use_cuda = use_cuda
    )
    optimizer = torch.optim.AdamW(gae.parameters())

    # Added in extra cell just for iterative programming / running of code
    #   but could be added to code block above

    if need_to_train_gae:
        start_time_geo = time.time()

        losses = train_ae(
            gae, df, groups, optimizer, 
            n_epochs=n_epochs_emb, sample_size=samples_size_emb,
            noise_min_scale=0.09, noise_max_scale=0.15, 
            dist=dist, recon=recon, use_cuda=use_cuda,
            hold_one_out=hold_one_out, hold_out=hold_out
        )
        run_time_geo = time.time() - start_time_geo

        print(run_time_geo)
        autoencoder = gae
    else:
        autoencoder = None


    # Weight of density (not percentage of total loss)
    lambda_density = 5

    # For petal=LeakyReLU / dyngen=CELU
    activation = 'CELU'

    # Can change but we never really do, mostly depends on the dataset.
    layers = [16,32,16]

    # Scale of the noise in the trajectories. Either len(groups)*[float] or None. Should be None if using an adaptative ODE solver.
    sde_scales = len(groups)*[0.2] 

    if recon:    
        model_features = gae_embedded_dim

    model = make_model(
        model_features, layers, 
        activation=activation, scales=None, use_cuda=use_cuda,
        n_aug=0
    )
    #model.func.alpha = None

    # Basically "batch size"
    sample_size=(batch_size, )

    # Training specification
    
    n_post_local_epochs = 0

    # Using the reverse trajectories to train
    reverse_schema = True
    # each reverse_n epoch
    reverse_n = 2


    criterion_name = 'ot'
    criterion = config_criterion(criterion_name)

    optimizer = torch.optim.AdamW(model.parameters())

    # Bookkeeping variables
    batch_losses = []
    globe_losses = []
    if hold_one_out and hold_out in groups:
        groups_ho = [g for g in groups if g != hold_out]
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups_ho)}
    else:
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups)}

    # For creating output.
    n_points = 100
    n_trajectories = 100
    n_bins = 100

    start_time = time.time()
    local_losses, batch_losses, globe_losses = training_regimen(
        # local, global, local train structure
        n_local_epochs=n_local_epochs, 
        n_epochs=n_epochs, 
        n_post_local_epochs=n_post_local_epochs,
        n_batches=n_batches,
        # where results are stored
        exp_dir=exp_dir, 
        
        # BEGIN: train params
        model=model, df=df, groups=groups, optimizer=optimizer, 
        criterion=criterion, use_cuda=use_cuda,
    
        hold_one_out=hold_one_out, hold_out=hold_out,
    
        use_density_loss=use_density_loss, 
        lambda_density=lambda_density,
    
        autoencoder=autoencoder, use_emb=use_emb, use_gae=use_gae, 
        
        sample_size=sample_size, 
        reverse_schema=reverse_schema, reverse_n=reverse_n,
        # END: train params

        plot_every=None,
        n_points=n_points, n_trajectories=n_trajectories, n_bins=n_bins, 
        #local_losses=local_losses, batch_losses=batch_losses, globe_losses=globe_losses
    )

    return model, gae

##########TrajectoryNet##############################

def TrajectoryNet_save_model(adata_path, res_path, embedding_key, n_iter=1000, dim=10):
    command = "python -m TrajectoryNet.load_save_model \
        --dataset {} \
            --embedding_name \"{}\"  --max_dim {}  --top_k_reg 0.1 \
                --training_noise 0.0 --niter {} --vecint 0 --solver 'rk4' \
                --dims \"16-32-16\" \
                    --save {} ".format(adata_path, embedding_key, dim, n_iter, res_path, )
    os.system(command)


def TrajectoryNet_interface(adata_path, res_path, embedding_key, n_iter=1000, dim=10):
    try:
        import TrajectoryNet    
    except ImportError:
        raise ImportError(
                "Please install the TrajectoryNet algorithm: `https://github.com/KrishnaswamyLab/TrajectoryNet`."
            )
    command = "python -m TrajectoryNet.main \
        --dataset {} \
            --embedding_name \"{}\"  --max_dim {}  --top_k_reg 0.1 \
                --training_noise 0.0 --niter {} --vecint 0 --solver 'rk4' \
                --dims \"16-32-16\" \
                    --save {} ".format(adata_path, embedding_key, dim, n_iter, res_path, )
    os.system(command)
    command = "python -m TrajectoryNet.eval \
        --dataset {} \
            --embedding_name \"{}\"  --max_dim {}  --top_k_reg 0.1 \
                --training_noise 0.0 --niter {} --vecint 0 --solver 'rk4' \
                --dims \"16-32-16\" \
                    --save {} ".format(adata_path, embedding_key, dim, n_iter, res_path, )
    os.system(command)
    TrajectoryNet_save_model(adata_path, res_path, embedding_key, n_iter, dim)
    model = torch.load('{}/model.pkl'.format(res_path))
    #traj = np.load('{}/backward_trajectories.npy'.format(res_path))
    return model


#########  WOT  ###################

# apply logistic function to transform to birth rate and death rate

def logistic(x, L, k, x0=0):
    f = L / (1 + np.exp(-k * (x - x0)))
    return f
def gen_logistic(p, beta_max, beta_min, pmax, pmin, center, width):
    return beta_min + logistic(p, L=beta_max - beta_min, k=4 / width, x0=center)

def beta(p, beta_max=1.7, beta_min=0.3, pmax=1.0, pmin=-0.5, center=0.25):
    return gen_logistic(p, beta_max, beta_min, pmax, pmin, center, width=0.5)

def delta(a, delta_max=1.7, delta_min=0.3, amax=0.5, amin=-0.4, center=0.1):
    return gen_logistic(a, delta_max, delta_min, amax, amin, center,
                          width=0.2)

def estimate_growth_rates(adata, species='human'):
    try:
        from moscot.problems.time import TemporalProblem  
    except ImportError:
        raise ImportError(
                "Please install the MOSCOT algorithm: `https://github.com/theislab/moscot`."
            )
    tp = TemporalProblem(adata)
    tp = tp.score_genes_for_marginals(
        gene_set_proliferation=species, gene_set_apoptosis=species
    )
    proliferation = adata.obs['proliferation']
    apoptosis = adata.obs['apoptosis']
    birth = beta(proliferation)
    death = delta(apoptosis)

    # growth rate is given by 
    gr = np.exp(birth-death)
    growth_rates_df = pd.DataFrame(index=adata.obs.index, data={'cell_growth_rate':gr})

    adata.obs['cell_growth_rate'] = growth_rates_df.cell_growth_rate.tolist()
    return adata

def wot_interpolate(adata, time_key, embedding_key, t0, t1, all_path=None, species='human'):
    try:
        import wot    
    except ImportError:
        raise ImportError(
                "Please install the WOT algorithm: `https://broadinstitute.github.io/wot/tutorial/`."
            )
    if all_path != None:
        adata_all = sc.read_h5ad(all_path)
        adata_all = adata_all[adata.obs.index]
        adata_all = estimate_growth_rates(adata_all, species=species)
        adata.obs[['proliferation', 'apoptosis', 'cell_growth_rate']] = adata_all.obs[['proliferation', 'apoptosis', 'cell_growth_rate']]
    else:
        adata = estimate_growth_rates(adata, species=species)
    adata.obs['day'] = adata.obs[time_key].tolist()

    # wot model
    ot_model = wot.ot.OTModel(adata,epsilon = 0.05, lambda1 = 1,lambda2 = 50) 
    tmap_annotated = ot_model.compute_transport_map(t0,t1)

    p0_ds = ot_model.matrix[ot_model.matrix.obs[ot_model.day_field] == float(t0), :]
    p1_ds = ot_model.matrix[ot_model.matrix.obs[ot_model.day_field] == float(t1), :]

    local_pca = adata.obsm[embedding_key].shape[1]
    p0_pca, p1_pca = adata[p0_ds.obs.index].obsm[embedding_key], adata[p1_ds.obs.index].obsm[embedding_key]
    p0_ds = sc.AnnData(p0_pca, obs=p0_ds.obs,
                                    var=pd.DataFrame(index=pd.RangeIndex(start=0, stop=local_pca, step=1)))
    p1_ds = sc.AnnData(p1_pca, obs=p1_ds.obs,
                                    var=pd.DataFrame(index=pd.RangeIndex(start=0, stop=local_pca, step=1)))
    
    interp_frac = 0.5
    interp_size = len(p0_ds)
    i05 = wot.ot.interpolate_with_ot(p0_ds.X, p1_ds.X, tmap_annotated.X, interp_frac,
                                                 interp_size)
    return i05


