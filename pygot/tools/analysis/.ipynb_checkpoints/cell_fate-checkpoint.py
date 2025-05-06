import numpy as np
from tqdm import tqdm
import torch
import torchdiffeq
import pandas as pd
from functools import partial
from sklearn.neighbors import KDTree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import copy


class FaissQuery:
    def __init__(self, X, device):
        try:
            import faiss
        
        except ImportError:
            raise ImportError(
                "Please install the FAISS package: `https://faiss.ai/ `.")
        if device != torch.device('cpu'):
            import faiss.contrib.torch_utils
            res = faiss.StandardGpuResources()  # use a single GPU
            # build a flat (CPU) index
            self.index = faiss.IndexFlatL2(X.shape[1])
            # make it into a gpu index
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
            self.index.add(X)  # 添加受检索数据
        else:
            self.index = faiss.IndexFlatL2(X.shape[1])  # 使用L2距离
            self.index.add(X)  # 添加受检索数据

    def query(self, x, k):
        distances, indices = self.index.search(x.cpu(), k)
        return distances, indices

def KNN_minibatch_torch_func(query, ref, ref_norm, K, av_mem=None, metric='euclidean'):
    if av_mem is None:
        av_mem = int(5e9)
    # 5000 Mb of GPU memory per batch
    Ntrain, D = ref.shape
    # Estimate the largest reasonable batch size:
    Ntest = query.shape[0]
    # Remember that a vector of D float32 number takes up 4*D bytes:
    Ntest_loop = min(max(1, av_mem // (4 * D * Ntrain)), Ntest)
    Nloop = (Ntest - 1) // Ntest_loop + 1
    out1 = torch.empty(Ntest, K)
    out2 = torch.empty(Ntest, K).long()

    
    # Actual K-NN query:
    for k in range(Nloop):
        x_test_k = query[Ntest_loop * k : Ntest_loop * (k + 1), :]
        out1[Ntest_loop * k : Ntest_loop * (k + 1), :], out2[Ntest_loop * k : Ntest_loop * (k + 1), :] = KNN_torch_fun(
                    ref, ref_norm, x_test_k, K, metric
        )
    return out1, out2

def KNN_torch_fun(x_train, x_train_norm, x_test, K, metric):
    largest = False  # Default behaviour is to look for the smallest values

    if metric == "euclidean":
        x_test_norm = (x_test**2).sum(-1)
        diss = (
            x_test_norm.view(-1, 1)
            + x_train_norm.view(1, -1)
            - 2 * x_test @ x_train.t()  # Rely on cuBLAS for better performance!
        )

    elif metric == "manhattan":
        diss = (x_test[:, None, :] - x_train[None, :, :]).abs().sum(dim=2)

    elif metric == "angular":
        diss = x_test @ x_train.t()
        largest = True

    elif metric == "hyperbolic":
        x_test_norm = (x_test**2).sum(-1)
        diss = (
            x_test_norm.view(-1, 1)
            + x_train_norm.view(1, -1)
            - 2 * x_test @ x_train.t()
        )
        diss /= x_test[:, 0].view(-1, 1) * x_train[:, 0].view(1, -1)
    else:
        raise NotImplementedError(f"The '{metric}' distance is not supported.")
    
    return diss.topk(K, dim=1, largest=largest)   

def gpu_sampling(p, num_samples):
    return p.multinomial(num_samples=num_samples, replacement=False)


class CellFate:
    """Quantification cell fate probability with stochastic neighbors diffusion

    After velocity model trained, trajectories of every cell that destoryed in sequenced time point can be infered by NeuralODE.
    Here we infer the terminate states of each cell and using simple logistic regression model to distinguish 
    the terminate states of different cell types. Besides, stochasistic neighbors diffusion is performed to make ODE stochastic to 
    probabilistic quantify future cell type.
    
    Example:
    ----------
    
    ::

        cf = CellFate(adata, embedding_key, model)
        cf.setup_cell_fate(cell_type_key, cell_type_list=['Neu', 'Mono'])
        x0_adata = adata[adata.obs[time_key] == 0]# early cell profiles 
        cell_fate_quant_df = cf.pred_cell_fate(x0_adata, time_key, end = np.max(adata.obs[time_key]) + 0.1)
    
    """
    
    def __init__(self, adata, embedding_key, model, appro=False, check=True, check_step=50, dt=0.01, sigma=0.10, n_neighbors=10, ):
        """Init model

        Arguments:
        ---------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        embedding_key: `str`
            Name of latent space, in adata.obsm
        model: `torch.nn.Module`
            trained velocity model
        appro: `bool` (default: False)
            Use `faiss` to search neighbors approximately, CPU recommended
        check: `bool` (default: True)
            Use stochastic neighbors diffusion (SND)
        check_step: `int` (default: 50)
            Number of step for each intergral
        dt: `float` (default: 0.01)
            dt for each step
        sigma: `float` (default: 0.1)
            Variance of noise
        n_neighbors: `int` (default: 10)
            Number of neighbors for SND
        
        """
        self.model = model
        self.check = check
        self.check_step = check_step
        self.dt = dt
        self.sigma = sigma
        self.k = n_neighbors
        self.classifiaction_model = None
        self.embedding_key = embedding_key
        self.adata = adata
        self.device = next(self.model.parameters()).device
        self.ref = torch.tensor(adata.obsm[embedding_key]).float().to(self.device)
        print('transite to neighbor for every time:{}'.format(dt * check_step))
        if appro:
            #self.tree = KDTree(self.ref)
            self.queryer = FaissQuery(self.ref, self.device)
            self.query_func = partial(self.queryer.query, k=self.k)
            
            print('Approximate Mode: {} | mapping cell with faiss'.format(appro))
        else:
            
            self.ref_norm = (self.ref**2).sum(-1).to(self.device)
            self.query_func = partial(KNN_minibatch_torch_func, ref=self.ref, ref_norm=self.ref_norm, K=self.k)
            
            print('Approximate Mode: {} | mapping cell with tensor distance directly'.format(appro))
        
        
    
    @torch.no_grad()
    def snd(self, xt, step=50, t_start=0):
        """
        Stochastic neighbors diffusion algorithm

        
        In tranditional ODE simulation, :math:`x_{end} = ODESolver(x_{start}, v, start, end)`, which is determinstic.
        To make ODE stochastic, we split simulation process into multiple stage.
        
        .. math::

            x_{start+(k+1)\delta} \sim N(ODESolver(x_{start+k\delta}, v, start+k\delta, start+(k+1)\delta)) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma)

        This function accept input :math:`x_{start+k\delta}` and output :math:`x_{start+(k+1)\delta}`
            
        Arguments:
        ----------
        xt: :class:`np.ndarray`
            Start cell states in latent spacce
        step: `int` (default: 50)
            :math:`\delta = ` step * dt
        t_start: `float` (default: 0)
            t_start :math:`= start + k \delta`

        Returns
        -------
        x_t+1: :class:`torch.tensor` (n_cells, latent_dim)
            one stage final output
        traj: :class:`torch.tensor` (step, n_cells, latent_dim)
            one stage trajectories
        
        """
        xt = torch.Tensor(xt).to(self.device)
        traj = torchdiffeq.odeint(self.model, xt,method='rk4', t=torch.Tensor(np.linspace(t_start, t_start + self.dt * step, abs(step))).to(self.device))
        xt = traj[-1].float()
        
        if self.check:
            dist, ind = self.query_func(xt)
            #ind = self.query_func(xt)
            dist = dist.mean(axis=1)
            if self.k > 1:
                indx = [ind[i][torch.randperm(self.k)[0]] for i in range(ind.shape[0])]
            else:
                indx = ind
            
            xt = self.ref[indx, :]
        else:
            dist = torch.zeros(xt.shape[0],)
        xt += self.sigma * torch.rand(xt.shape[0], xt.shape[1]).to(self.device)
        
        return xt, traj, dist
    
    @torch.no_grad()
    def simulation(self, xt, 
                    t_start, t_end, mode
                    ):
        """
        Use SND to simulation

        Arguments:
        ----------
        xt: :class:`np.ndarray`
            Start cell states in latent spacce
        t_start: `float` 
            Start time
        t_end: `float`
            End time

        Returns
        -------
        check_traj: :class:`torch.tensor` (n_stages, n_cells, latent_dim)
            check point cell states
        all_traj: :class:`torch.tensor` ((t_end - t_start) // dt, n_cells, latent_dim)
            whole trajectories

        """
        xt = xt.to(next(self.model.parameters()).device)
        total_step = abs(int((t_end - t_start) / self.dt))
        if mode == 'stochastic':
            checkpoint_traj = [xt]
            next_xt = xt
            time_interval = self.dt * self.check_step
            limit = total_step // self.check_step
            remain_step = total_step % self.check_step
            all_traj = []
            flag = -1 if t_end < t_start else 1
            traj_dist = []
            for i in range(limit + 1):
                if i < limit:
                    next_xt, traj, dist = self.snd(next_xt, self.check_step * flag, t_start)            
                else:
                    if remain_step > 0:
                        next_xt, traj, dist = self.snd(next_xt, remain_step * flag, t_start)
                    else:
                        break
                all_traj.append(traj)
                checkpoint_traj.append(next_xt)
                traj_dist.append(torch.tensor(dist))
                t_start += (time_interval * flag)
                        
            checkpoint_traj, all_traj = torch.concat(checkpoint_traj, dim=0), torch.concat(all_traj, dim=0)
            
            traj_dist = torch.mean(torch.stack(traj_dist, dim=0), dim=0)
            return checkpoint_traj, all_traj, traj_dist
        else:
            traj = torchdiffeq.odeint(self.model, xt,method='rk4', t=torch.Tensor(np.linspace(t_start, t_end, total_step)).to(self.device))
            return None, traj, None
        
        
        
    @torch.no_grad()
    def pred_cell_fate(self, x0_adata, time_key, end, sample_size=100, batch_size=4096, mode='stochastic'):
        """
        Using SND simulation to quantify cell fate

        Arguments:
        ----------
        x0_adata: :class:`~anndata.AnnData`
            Annotated data matrix of start cells
        time_key: `str`
            Name of time label, in adata.obs
        end: `float` 
            Terminate time 
        sample_size: `int` (default: 100)
            Number of SND simulation
        
        Returns
        -------
        cell_fate_df: :class:`pd.DataFrame`, (n_cells, n_future_cell_types)
            Cell fate quantification result

        """

        if self.classifiaction_model is None:
            raise Exception('Please set up cell fate by `setup_cell_fate` first')
        
        labels = self.classifiaction_model.classes_.tolist()
        print('Calculate Cell Fate of ', labels)
        simulated_fate = np.zeros(shape=(len(x0_adata), len(labels)))
        simulated_dist = np.zeros(shape=(len(x0_adata,)))
        start_t_idxs = [np.where(x0_adata.obs[time_key] == start)[0] for start in np.sort(np.unique(x0_adata.obs[time_key]))]
        if mode == 'deterministic':
            sample_size = 1
        
        for i in range(sample_size):
            for j, start in enumerate(np.sort(np.unique(x0_adata.obs[time_key]))):
                batch_idxs = [start_t_idxs[j][i:i + batch_size] for i in range(0, len(start_t_idxs[j]), batch_size)]
                
                for batch in tqdm(batch_idxs):
                    _, all_traj, traj_dist = self.simulation(torch.Tensor(x0_adata.obsm[self.embedding_key][batch]), start, end, mode=mode)
                    simulated_fate[batch,:] += self.classifiaction_model.predict_proba(all_traj[-1].cpu().numpy())        
                    if mode != 'deterministic':
                        simulated_dist[batch] += traj_dist.cpu().numpy()

        
        simulated_fate /= sample_size
        simulated_dist /= sample_size

        simulated_fate = pd.DataFrame(simulated_fate, columns=labels, index=x0_adata.obs.index)
        simulated_dist = pd.DataFrame(simulated_dist, columns=['traj_dist'], index=x0_adata.obs.index)

        
        return simulated_fate, simulated_dist
    
    def setup_cell_fate(self, cell_type_key, cell_type_list=None, obs_index=None, specified_model=None, trained=False, report=True):
        """
        Set up cell fate quantification

        Arguments:
        ----------
        cell_type_key: `str`
            Name of cell type, in adata.obs
        cell_type_list: `list` (default: None (all cell type))
            Future cell type to be quantified
        obs_index: `list` (default: None) 
            Specify cell index to train the classification model for cell type
        specified_model: :class:`sklearn.base.ClassifierMixin` (default: None)
            If None, use sklearn.linear_model.LogisticRegression
            
        
        Returns
        -------
        classifiaction_model(self.): :class:`sklearn.base.ClassifierMixin`
            Trained classification model

        """
        
        if not obs_index is  None:
            train_adata = self.adata[obs_index]
        elif not cell_type_list is None:
            train_adata = self.adata[(self.adata.obs[cell_type_key].isin(cell_type_list))]
            cell_type_list = np.array(cell_type_list)
        else:
            train_adata = self.adata


        if specified_model is None:
            model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
        else:
            model = specified_model

        X = train_adata.obsm[self.embedding_key]
        y_encoded = train_adata.obs[cell_type_key]
        
        if not trained:
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)    
            model.fit(X_train, y_train)
            
        else:
            model.register_celltype_list(cell_type_list)
            model.classes_ = cell_type_list
            X_test = X
            y_test = y_encoded
        if report:
            y_pred = model.predict(X_test)
            print("Classification Report:")
            print(classification_report(y_test, y_pred, target_names=model.classes_))

            print("Accuracy:", accuracy_score(y_test, y_pred))
        self.classifiaction_model = model



def compute_fate_coupling(
    stage_adata,
    model,
    current_stage,
    next_stage,
    n_neighbors=10,
    sigma=0.1,
    embedding_key = 'X_pca',
    time_key = 'stage_numeric',
    cell_type_key='cell_state',  
    dt=0.001,
    check_step=50,
    scale = 1.0,
    min_cell = 20,
    backward=True,
    appro=False,
    sample_size=100,
    **kwargs
):
    
    print(current_stage, next_stage)
    
    cell_state_counts = stage_adata.obs[cell_type_key].value_counts()
    
    stage_adata = stage_adata[stage_adata.obs[cell_type_key].isin(cell_state_counts[cell_state_counts > min_cell].index)]
    
    
    next_stage_cell_state = stage_adata.obs.loc[stage_adata.obs[time_key] == next_stage][cell_type_key].unique().tolist()
    current_stage_cell_state = stage_adata.obs.loc[stage_adata.obs[time_key] == current_stage][cell_type_key].unique().tolist()
    
    cf = CellFate(stage_adata, embedding_key=embedding_key, model=model
                                    , dt=dt, check=True, appro=appro,check_step=check_step,
                  n_neighbors=n_neighbors, sigma=sigma
                 )
    
    next_meta = stage_adata.obs.loc[stage_adata.obs[time_key] == next_stage]
    next_meta[cell_type_key] = next_meta[cell_type_key].astype(str)
    current_meta = stage_adata.obs.loc[stage_adata.obs[time_key] == current_stage]
    current_meta[cell_type_key] = current_meta[cell_type_key].astype(str)
    t_diff = next_stage - current_stage
    if backward:
        cell_type_list = current_stage_cell_state
        meta = next_meta
        x0_adata = stage_adata[stage_adata.obs[time_key] == next_stage]
        x1_stage = next_stage - (t_diff * scale)
    else:
        cell_type_list = next_stage_cell_state
        meta = current_meta
        x0_adata = stage_adata[stage_adata.obs[time_key] == current_stage]
        x1_stage = current_stage + (t_diff * scale)
        
    cf.setup_cell_fate(cell_type_key=cell_type_key,cell_type_list=cell_type_list, **kwargs)
    cell_coupling, cell_traj_dist = cf.pred_cell_fate(x0_adata, time_key, x1_stage, sample_size=sample_size)

    cc = cell_coupling.copy().to_numpy()
    cc[cc < (1./ cc.shape[1])] = np.nan
    cc[cc > (1./ cc.shape[1])] = 1.
    traj_dist = pd.DataFrame(cell_traj_dist['traj_dist'].to_numpy()[:,None] * cc, index=meta.index, columns = cell_coupling.columns)
    traj_dist = np.stack(meta.groupby(cell_type_key).apply(lambda x: np.nanmean(traj_dist.loc[x.index], axis=0)))
    traj_dist = traj_dist.T if backward else traj_dist
    
    state_coupling = meta.groupby(cell_type_key).apply(lambda x: cell_coupling.loc[x.index].mean(axis=0))
    state_coupling = state_coupling.T if backward else state_coupling

    #permutation test
    permutated_dist = []
    for k in range(100):
        p = meta[cell_type_key].tolist()
        np.random.shuffle(p)
        
        meta['permutated'] = p
        permutated_state_coupling = meta.groupby('permutated').apply(lambda x: cell_coupling.loc[x.index].mean(axis=0))
        permutated_state_coupling = permutated_state_coupling.T if backward else permutated_state_coupling
        permutated_dist.append(permutated_state_coupling.loc[state_coupling.index][state_coupling.columns].to_numpy().flatten())
        
    permutated_dist = np.concatenate(permutated_dist)
    
    state_coupling_dist = pd.DataFrame(traj_dist, columns=state_coupling.columns, index=state_coupling.index)
    
    del cf
    
    return state_coupling, state_coupling_dist, cell_coupling, permutated_dist


class scANVIClassifier:
    def __init__(self, lvae, adata):
        self.lvae = lvae
        self.label2celltype = lvae.registry_['field_registries']['labels']['state_registry']['categorical_mapping']
    def register_celltype_list(self, cell_type_list):
        self.register_list = np.array(cell_type_list)
        self.target_celltype = np.array([np.where(self.label2celltype == c)[0][0] for c in cell_type_list])
        
    @torch.no_grad()
    def predict_proba(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.lvae.device)
        logit_prob = self.lvae.module.classifier(x)
        logit_prob = logit_prob[:, self.target_celltype]

        prob = logit_prob.softmax(dim=-1)
        return prob.cpu().numpy()
        
    @torch.no_grad()
    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.lvae.device)
        
        logit_prob = self.lvae.module.classifier(x)
        logit_prob = logit_prob[:, self.target_celltype]
        
        
        y_pred = self.register_list[logit_prob.argmax(dim=-1).cpu().numpy()]
        return y_pred


# Classifier
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, output_dim)
        self.device = device
            
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
    def register_celltype_list(self, cell_type_list):
        self.register_list = np.array(cell_type_list)
        self.target_celltype = np.array([np.where(self.label2celltype == c)[0][0] for c in cell_type_list])
        
    @torch.no_grad()
    def predict_proba(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.device)
        logit_prob = self.forward(x)
        logit_prob = logit_prob[:, self.target_celltype]
        prob = logit_prob.softmax(dim=-1)
        return prob.cpu().numpy()
        
    @torch.no_grad()
    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.device)
        
        logit_prob = self.forward(x)
        logit_prob = logit_prob[:, self.target_celltype]
        
        y_pred = self.register_list[logit_prob.argmax(dim=-1).cpu().numpy()]
        return y_pred

    def save(self, path):
        torch.save(self, path + '/map_model.pt')

def learn_embed2class_map(adata, embedding_key, class_key, batch_size=256, num_epochs = 100, patience = 5,
                       device=None):
    input_dim = adata.obsm[embedding_key].shape[1]
    output_dim = adata.obs[class_key].unique().shape[0]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X = torch.tensor(adata.obsm[embedding_key]).float().to(device)
    
    adata.obs[class_key] = pd.Categorical(adata.obs[class_key])
    label2celltype = adata.obs[class_key].cat.categories
    y = torch.tensor(adata.obs[class_key].cat.codes).long().to(device)  

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = Classifier(input_dim, output_dim, device).to(device)
    model.label2celltype = label2celltype
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    

    best_val_loss = float('inf')
    patience_counter = 0
    
    # training
    
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        model.train()
        train_loss = 0.
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        # 验证模型
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_outputs = model(val_X)
                val_loss += criterion(val_outputs, val_y).item()
        
        val_loss /= len(val_loader)
        pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        #print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # eval
    model.eval()
    model.register_celltype_list(label2celltype)
    with torch.no_grad():
        y_pred = []
        y_test = []
        for val_X, val_y in val_loader:
            y_pred.append(model.predict(val_X))
            
            y_test.append(model.label2celltype[val_y.cpu().numpy()])
        y_pred = np.concatenate(y_pred)
        y_test = np.concatenate(y_test)
        print(classification_report(y_test, y_pred))
    model.load_state_dict(best_state)
                
    return model(X).detach().cpu().numpy(), model  