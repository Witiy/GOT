import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

import networkx as nx
from itertools import combinations
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from anndata.experimental.pytorch import AnnLoader



# 定义VAE的编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 128)
        self.fc2_mean = nn.Linear(128, latent_dim)
        self.fc2_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar

# 定义VAE的解码器
class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 400),
            nn.ELU(),
            nn.Linear(400, input_dim)
        )
        
    def forward(self, z):
        return self.net(z)
        
    
    def elbo(self, z):
        return self.forward(z)
        
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, beta=1e-2) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim)
        self.beta = beta

    def elbo(self, x):
        
        mean, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)

        x_hat = self.decoder(z)

        recon_loss = torch.sum((x_hat -x)**2)
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) * self.beta
        return recon_loss, kl_div

    def forward(self, x):
        mean, logvar = self.encoder(x)
        x_hat = self.decoder(mean)
        return x_hat
    
    def inverse_transform(self, z):
        x_hat = self.decoder(z)
        return x_hat


class Translator(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                        nn.CELU(),
                                        nn.Linear(latent_dim, latent_dim),
                                        nn.CELU(),
                                        nn.Linear(latent_dim, out_dim))
    def forward(self, x):
        return self.net(x)
      
class funcVAE(VAE):
    
    def __init__(self, input_dim, latent_dim, interpretation_dims, lam=1e-2, hinge_value=1e-3, beta=1e-6) -> None:
        assert latent_dim % len(interpretation_dims) == 0
        super().__init__(input_dim, latent_dim, beta)
        if len(interpretation_dims) > 0:
            self.n_interpretator = len(interpretation_dims)
        else:
            self.n_interpretator = 0
        self.interpretators = nn.ModuleList([Decoder(interpretation_dims[i], latent_dim // self.n_interpretator) for i in range(len(interpretation_dims))])
        self.interpretation_dims = interpretation_dims
        self.hinge_value = hinge_value
        self.lam = lam
        self.translator = Translator(latent_dim, latent_dim)
        #self.translator = nn.ModuleList([Translator(latent_dim, latent_dim // self.n_interpretator) for i in range(len(interpretation_dims))])
    def forward(self, x):
        mean, logvar = self.encoder(x)

        z_tuple = self.translator(mean)
        x_hat = self.decoder(z_tuple)
        return x_hat
    
    def inverse_transform(self, z):
        z_tuple = self.translator(z)
        x_hat = self.decoder(z_tuple)
        return x_hat
    
    def elbo(self, x, x_func):
        mean, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)
        z_tuple = self.translator(z)
        x_hat = self.decoder(z_tuple)

        
        kl_div = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp()) * self.beta

        recon_loss = torch.mean((x_hat -x)**2)
        
        if len(self.interpretation_dims) == 0:
            return kl_div, recon_loss, 0

        z = z.view(z.size(0), -1, self.n_interpretator)
        start = 0
        func_loss = 0
        for i in range(len(self.interpretation_dims)):
            
            end = start + self.interpretation_dims[i]
            x_func_hat = self.interpretators[i](z[:,:,i])
            
            func_loss += torch.mean(torch.relu((x_func_hat - x_func[:,start:end])**2 - self.hinge_value))
            start = end
        
        return kl_div, recon_loss, self.lam*func_loss / len(self.interpretation_dims)
    
    def pred_func(self, x):
        z, _ = self.encoder(x)

        z = z.view(z.size(0), -1, self.n_interpretator)
        x_func_hats = []
        for i in range(len(self.interpretation_dims)):
            x_func_hat = self.interpretators[i](z[:,:,i])
            
            x_func_hats.append(x_func_hat)
        x_func_hat = torch.concat(x_func_hats, dim=-1)
        
        return x_func_hat
    
    def pred_func_latent(self, z):

        z = z.view(z.size(0), -1, self.n_interpretator)
        x_func_hats = []
        for i in range(len(self.interpretation_dims)):
            x_func_hat = self.interpretators[i](z[:,:,i])
            
            x_func_hats.append(x_func_hat)
        x_func_hat = torch.concat(x_func_hats, dim=-1)
        
        return x_func_hat
    
    def get_local_global_latent(self, x):
        z, _ = self.encoder(x)
        z_tuple = self.translator(z)
        return z, z_tuple


def gmt_to_decoupler(pth, mouse=False) -> pd.DataFrame:
    """
    Parse a gmt file to a decoupler pathway dataframe.
    """
    from itertools import chain, repeat

    pathways = {}

    with Path(pth).open("r") as f:
        for line in f:
            name, _, *genes = line.strip().split("\t")
            pathways[name] = genes

    geneset = pd.DataFrame.from_records(
        chain.from_iterable(zip(repeat(k), v) for k, v in pathways.items()),
        columns=["geneset", "genesymbol"],
    )
    if mouse:
        geneset.genesymbol = geneset.genesymbol.apply(lambda x : x[0]+x[1:].lower())
    return geneset

def clique_filter(df, correlation_threshold = 0.7):

    # 计算特征之间的相关性
    correlation_matrix = df.corr()

    # 创建相关性图
    G = nx.Graph()
    for i, j in combinations(correlation_matrix.index, 2):
        if correlation_matrix.loc[i, j] >= correlation_threshold:
            G.add_edge(i, j)

    # 查找所有最大团
    cliques = list(nx.find_cliques(G))
    
    # 合并特征并添加到DataFrame中
    for clique in cliques:
        new_feature_name = '+'.join(clique)
        new_feature_values = None
        for feat in clique:
            if new_feature_values is None:
                new_feature_values = df[feat].copy()
            else:
                # 根据相关性的方向和强度来决定如何合并特征
                if correlation_matrix.loc[feat, clique[0]] >= 0:
                    new_feature_values += df[feat]
                else:
                    new_feature_values -= df[feat]
        new_feature_values /= len(clique)
        df[new_feature_name] = new_feature_values

    # 删除原始DataFrame中与合并特征相关的所有特征列
    df.drop(columns=sum(cliques, []), inplace=True)
    return df



class GlobalDiscriminator(nn.Module):
    def __init__(self, n_interpretator, dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(nn.Linear(dim, dim), 
                                 nn.ReLU(),
                                 nn.Linear(dim, dim))
        self.l0 = nn.Linear(dim+dim, 128)
        self.l1 = nn.Linear(128, 128)
        self.l2 = nn.Linear(128, 1)
    
    def forward(self, y, M):
        #M:[bs, d//c, c]
        #y:[bs, d]
        h = self.net(M.view(M.shape[0], -1))
        h = torch.concat([y, h], dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)
    
class LocalDiscriminator(nn.Module):
    def __init__(self, n_interpretator, dim,):
        super().__init__()
        self.l1 = nn.ModuleList([nn.Linear(dim + dim // n_interpretator, 128) for i in range(n_interpretator)])
        self.l2 = nn.ModuleList([nn.Linear(128, 128) for i in range(n_interpretator)])
        self.l3 = nn.ModuleList([nn.Linear(128, 1) for i in range(n_interpretator)])


    def forward(self, y_M):
        #y_M : [bs, d + d+d//c, c]
        score = []
        for i in range(y_M.shape[-1]):
            h = F.relu(self.l1[i](y_M[:,:,i]))
            h = F.relu(self.l2[i](h))
            score.append(self.l3[i](h))
        
        return torch.concat(score, dim=-1)

class DeepInfoMaxLoss(nn.Module):
    def __init__(self, n_interpretator, dim, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        self.global_d = GlobalDiscriminator(n_interpretator, dim)
        self.local_d = LocalDiscriminator(n_interpretator, dim)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_interpretator = n_interpretator
    def forward(self, y, M):
        #y:[bs, d]
        #M:[bs, d]
        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        
        y_exp = y[:,:,None].expand(-1, -1, self.n_interpretator)
        M = M.view(M.size(0), -1, self.n_interpretator)
        M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha
        '''
        prior = torch.rand_like(y)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
        
        return LOCAL + GLOBAL + PRIOR
        '''
        return LOCAL + GLOBAL


def training_embedder(model, adata, loss_func, lr=1e-4, batch_size=128, n_epoch = 100, device=torch.device('cpu')):
    model.to(device)
    dataloader = AnnLoader(adata, batch_size=batch_size, shuffle=True, 
                           use_cuda=False if device == torch.device('cpu') else True,
                          )
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    pbar = tqdm(range(n_epoch))
    history = []
    vq_history, zinb_history = [], []
    for epoch in pbar:
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss = loss_func(model, batch)
            loss.backward()
            optimizer.step()
            #vq_history.append(vq_loss.item())
            #zinb_history.append(zinb_loss.item())
            history.append(loss.item())
            pbar.set_description('loss :{:.4f}'.format(np.mean(history[-100:])))
            #pbar.set_description('vq loss :{:.4f} dist loss :{:.4f}'.format(np.mean(vq_history[-100:]), np.mean(zinb_history[-100:])))
    return model


def DIM_training_embedder(model, adata, loss_func, dim, n_interpretator, lr=1e-4, batch_size=128, n_epoch = 100, device=torch.device('cpu')):
    model.to(device)
    dataloader = AnnLoader(adata, batch_size=batch_size, shuffle=True,
                           use_cuda=False if device == torch.device('cpu') else True, )
    dim_fn = DeepInfoMaxLoss(n_interpretator=n_interpretator, dim=dim).to(device)

    loss_optim = optim.Adam(dim_fn.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    pbar = tqdm(range(n_epoch))
    history = []
    vq_history, zinb_history = [], []
    for epoch in pbar:
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss_optim.zero_grad()
            loss = loss_func(model, batch)
            loss.backward()
            
            #y, _ = model.encoder(batch.X.float())
            #M = model.translator(y)
            
            M, y = model.get_local_global_latent(batch.X)
            
            dim_loss = 1e-3 * dim_fn(y, M)
            
            dim_loss.backward()
            
            optimizer.step()
            loss_optim.step()
            #vq_history.append(vq_loss.item())
            #zinb_history.append(zinb_loss.item())
            history.append(dim_loss.item() + loss.item())
            pbar.set_description('loss :{:.4f}'.format(np.mean(history[-100:])))
            #pbar.set_description('vq loss :{:.4f} dist loss :{:.4f}'.format(np.mean(vq_history[-100:]), np.mean(zinb_history[-100:])))
    return model


def vae_loss_func(model:VAE, batch):
    vq_loss, recon_loss = model.elbo(batch.X.float())
    loss = vq_loss + recon_loss
    return loss

def functional_loss_func(model:VAE, batch):
        
    vq_loss, recon_loss, func_loss = model.elbo(batch.X.float(), batch.obsm['aucell'])

    loss = vq_loss + recon_loss + func_loss
    
    return loss



class GS_VAE:
    def __init__(self, device=None):
        print('To use VAE, run as following:')
        print('     1. gs_vae.register_model')
        print('     2. gs_vae.fit / gs_vae.fit_transform')

        print('To use GS-VAE, run as following:')
        print('     1. gs_vae.precompute_gs')
        print('     2. gs_vae.process_gs')
        print('     3. gs_vae.register_model')
        print('     4. gs_vae.fit / gs_vae.fit_transform')
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
    def fit(self, adata, lr=1e-4, batch_size=128, n_epoch = 100):
        
        self.model.train()
        
        self.model = self.trainer(model=self.model, adata=adata, loss_func=self.loss_func, lr=lr, batch_size=batch_size, n_epoch=n_epoch)

        self.model.eval()

    def fit_transform(self, adata, lr=1e-4, batch_size=128, n_epoch = 100):
        self.fit(adata, lr, batch_size, n_epoch)
        return self.transform(adata)
    
    @torch.no_grad()
    def transform(self, adata):
        if isinstance(adata.X, np.ndarray):
            X = torch.Tensor(adata.X).to(self.device)
        else:
            X = torch.Tensor(adata.X.toarray()).to(self.device)
        z, _ = self.model.encoder(X)
        return z.detach().cpu().numpy()
    
    @torch.no_grad()
    def inverse_transform(self, x_latent):
        if not isinstance(x_latent, torch.Tensor):
            x_latent = torch.Tensor(x_latent)
        return self.model.inverse_transform(x_latent.to(self.device)).detach().cpu().numpy()
    
    @torch.no_grad()
    def pred_func_latent(self, x_latent):
        if self.flag:
            return self.model.pred_func_latent(x_latent.to(self.device))
        else:
            print('The model is VAE, please use precompute_gs and process_gs to register gene set book in anndata')
    
    @torch.no_grad()
    def pred_func(self, x):
        if self.flag:
            return self.model.pred_func(x.to(self.device))
        else:
            print('The model is VAE, please use precompute_gs and process_gs to register gene set book in anndata')
    


    def precompute_gs(self, adata, gene_set_book_paths, mouse=False, ):
        try:
            import decoupler
        except ImportError:
                raise ImportError(
                    "Please install the decoupler via pip install decoupler`.")
        auc_space = []

        for i in range(len(gene_set_book_paths)):
            geneset = gmt_to_decoupler(gene_set_book_paths[i], mouse=mouse)
            decoupler.run_aucell(
                adata,
                geneset,
                source="geneset",
                target="genesymbol",
                use_raw=False,
        
            )
            auc_space.append(adata.obsm['aucell_estimate'])
        return auc_space
    
    def process_gs(self, adata, auc_space, c_filter=False, c_threshold=0.7):
        pathway_names = []
        for i in range(len(auc_space)):
            auc_space[i] = auc_space[i][auc_space[i].columns[(auc_space[i]>0.1).sum(axis=0 ) > (0.05 * len(adata))]]
            pathway_names.append(auc_space[i].columns.to_numpy())
            if c_filter == True:
                auc_space[i] = clique_filter(auc_space[i], correlation_threshold = c_threshold)
        self.pathway_names = np.concatenate(pathway_names)
        aucell_df = pd.concat(auc_space, axis=1)
        print('Gene Set Number:')
        print(aucell_df.shape[1])

        adata.obsm['aucell_df'] = aucell_df
        adata.obsm['aucell'] = adata.obsm['aucell_df'].to_numpy()
        adata.uns['idx'] = [df.shape[1] for df in auc_space]
        return adata

    def register_model(self, adata, latent_dim, lam=1, hinge_value=1e-2, beta=1e-6):
        if 'idx' in adata.uns.keys():
            self.model = funcVAE(adata.X.shape[1], latent_dim=latent_dim, lam=lam, interpretation_dims = adata.uns['idx'], hinge_value=hinge_value, beta=beta)
            self.loss_func = functional_loss_func
            self.trainer = partial(DIM_training_embedder, dim=latent_dim, n_interpretator=len(adata.uns['idx']), device=self.device)
            self.flag = True
        else:
            self.model = VAE(adata.X.shape[1], latent_dim=latent_dim)
            self.loss_func = vae_loss_func
            self.trainer = partial(training_embedder, device=self.device)
            self.flag = False