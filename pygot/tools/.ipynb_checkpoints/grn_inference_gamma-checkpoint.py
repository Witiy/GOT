import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from pygot.evalute import *
from scipy.sparse import issparse

class GeneDegradation(nn.Module):
    def __init__(self, output_size, init_beta=1.0, min_beta=0.0, beta_grad=True):
        super(GeneDegradation, self).__init__()
        if beta_grad:
            self.beta = nn.Parameter(init_beta*torch.ones(output_size))
            self.beta.register_hook(self.hinge_hook)
            self.min_beta = torch.tensor(min_beta)
        else:
            self.beta = init_beta*torch.ones(output_size)
        self.relu = nn.ReLU()

    def hinge_hook(self, grad):
        with torch.no_grad():
            self.beta.data = torch.clamp(self.beta, min=self.min_beta)
        return grad
    def forward(self, x):
        return self.relu(self.beta) * x
# 定义多元多任务回归模型
class GeneRegulatroyModel(nn.Module):
    def __init__(self, tf_num, gene_num, tf_idx, init_jacobian=None, non_negative=True):
        super(GeneRegulatroyModel, self).__init__()
        #G_i,j ~ Gene_j -> Gene_i
        if init_jacobian is None:
            init_jacobian = torch.rand(gene_num, tf_num)
        self.linear = nn.Parameter(init_jacobian)
        
        # Remove self-regulated edge
        if tf_num == gene_num:
            self.linear.register_hook(self.remove_diagonal_hook)
        else:
            self.indices_to_remove = tf_idx
            self.linear.register_hook(self.custom_remove_hook)
        
        self.non_negative = non_negative

    def forward(self, x):
        return (self.linear @ x[:,:,None]).squeeze(-1)
        
    def custom_remove_hook(self, grad):
        with torch.no_grad():
            self.linear[self.indices_to_remove, range(self.linear.shape[1])] = 0
        return grad
        
    def remove_diagonal_hook(self, grad):
        with torch.no_grad():
            self.linear -= torch.diag(torch.diag(self.linear))
        return grad

    def apply_non_negative(self):
        with torch.no_grad():
            self.linear.data = torch.clamp(self.linear.data, min=0)
class GRN:
    def __init__(self, G_hat:GeneRegulatroyModel, beta_hat: GeneDegradation, tf_names, gene_names):
        self.model = G_hat
        self.G = G_hat.linear.detach().cpu().numpy()
        self.beta = beta_hat.beta.data.detach().cpu().numpy()
        self.ranked_edges = get_ranked_edges(self.G, tf_names=tf_names, gene_names=gene_names)
        self.tf_names = tf_names
        self.gene_names = gene_names
        self.models = {'G':G_hat, 'beta':beta_hat}

def preproce_dataset(adata, TF_names, batch_size, layer_key, early_stopping, val_split, device):
    y = torch.Tensor(adata.layers['scaled_velocity'])
    adata.var['idx'] = range(len(adata.var))
    if layer_key is None:
        X = torch.Tensor(adata.X)
    else:
        X = torch.Tensor(adata.layers[layer_key])

    tf_idx = adata.var.loc[TF_names]['idx'].to_numpy()
    # Split into training and validation sets if early stopping is enabled
    if early_stopping:
        dataset_size = X.shape[0]
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        X_train, y_train = X[train_indices].to(device), y[train_indices].to(device)
        X_val, y_val = X[val_indices].to(device), y[val_indices].to(device)
    else:
        X_train, y_train = X.to(device), y.to(device)
        X_val, y_val = None, None
    batch_size = min(batch_size, len(X_train))
    
    return X, y, X_train, y_train, X_val, y_val, tf_idx

def optimize_global_GRN(adata, TF_names, layer_key=None,
                        beta_grad=True, num_epochs=100000, lr=0.01, l1_penalty = 0.005, init_beta=1.0, min_beta=1.0, 
                        init_jacobian=None, device=torch.device('cpu'), 
                        early_stopping=False, batch_size=32, val_split=0.2, non_negative=True):
    
    
    print('l1_penalty:', l1_penalty, 'min_beta:', min_beta)
    X, y, X_train, y_train, X_val, y_val, tf_idx = preproce_dataset(adata, TF_names, batch_size, layer_key, early_stopping, val_split, device)
    batch_size = min(batch_size, len(X_train))
    gene_num = y.shape[1]
    tf_num = tf_idx.shape[0]
    
    
    G_hat = GeneRegulatroyModel(tf_num, gene_num, tf_idx, init_jacobian, non_negative=non_negative).to(device)
    if isinstance(init_beta, float):
        beta_hat = GeneDegradation(gene_num, init_beta, min_beta, beta_grad).to(device)
    elif isinstance(init_beta, GeneDegradation):
        beta_hat = init_beta
        
    beta_hat.min_beta = beta_hat.min_beta.to(device)
    beta_hat.beta = beta_hat.beta.to(device)
   
    optimizer_G = optim.SGD(G_hat.parameters(), lr=lr)
    if beta_grad:
        optimizer_beta = optim.SGD(beta_hat.parameters(), lr=lr)
    loss_list = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        G_hat.train()
        beta_hat.train()
        train_loss = 0
        permutation = torch.randperm(X_train.size()[0])
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            optimizer_G.zero_grad()
            if beta_grad:
                optimizer_beta.zero_grad()
            outputs = G_hat(batch_x[:,tf_idx]) - beta_hat(batch_x)
            mse_loss = torch.mean(((outputs - batch_y) ** 2))
            
            #l1_loss = l1_penalty *  (torch.norm(G_hat.linear, p=1, dim=0).sum() + torch.norm(G_hat.linear, p=1, dim=1).sum())
            l1_loss = l1_penalty * torch.norm(G_hat.linear, p=1)
            loss = mse_loss + l1_loss
            loss.backward()
            optimizer_G.step()
            if beta_grad:
                optimizer_beta.step()
            
            if G_hat.non_negative:
                G_hat.apply_non_negative()
            
            train_loss += loss.item()

        train_loss /= (X_train.size()[0] // batch_size)
        
        if early_stopping:
            G_hat.eval()
            beta_hat.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(0, X_val.size()[0], batch_size):
                    batch_x, batch_y = X_val[i:i + batch_size], y_val[i:i + batch_size]
                    outputs = G_hat(batch_x[:,tf_idx]) - beta_hat(batch_x)
                    mse_loss = torch.mean(((outputs - batch_y) ** 2))
                    loss = mse_loss + l1_loss
                    val_loss += loss.item()
            
            val_loss /= (X_val.size()[0] // batch_size) if batch_size < X_val.size()[0] else 1
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.5f}')
                break

            pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        else:
            pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')
        
        loss_list.append(train_loss)

    fit_godness = torch.mean((G_hat(X[:,tf_idx].to(device)) + beta_hat(X.to(device)) - y.to(device))**2, dim=-1).detach().cpu().numpy()
    return G_hat, beta_hat, fit_godness

def get_ranked_edges(jacobian, tf_names, gene_names, num_top=10000):
    df = pd.DataFrame(jacobian, index=gene_names, columns=tf_names).T
    stacked = df.stack()
    values = stacked.to_numpy().flatten()
    idx = np.argsort(values)
    top_idx = np.unique(np.concatenate([idx[:num_top], idx[-num_top:]]))
    gene1 = tf_names[top_idx // len(gene_names)]
    gene2 = gene_names[top_idx % len(gene_names)]
    result = pd.DataFrame([gene1, gene2, values[top_idx]], index=['Gene1', 'Gene2', 'EdgeWeight']).T
    result['absEdgeWeight'] = abs(result.EdgeWeight)
    result = result.sort_values('absEdgeWeight', ascending=False)
    return result

def infer_GRN(
    adata,
    TF_constrain=True,
    TF_names=None,
    species='human',
    lineage_key=None,
    layer_key=None, 
    beta_grad=True, 
    num_epochs=10000, 
    lr=0.01, 
    l1_penalty = 0.005, 
    init_beta=1.0, 
    min_beta=1.0, 
    init_jacobian=None, 
    device=None, 
    early_stopping=True, 
    batch_size=2048, 
    val_split=0.2,
    non_negative=True,

):
    
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if not 'velocity' in adata.layers.keys():
        raise KeyError('Please compute velocity first and store velocity in adata.layers')
    
    if not 'gene_name' in adata.uns.keys():
        adata.uns['gene_name'] = adata.var.index

    if TF_constrain:
        if TF_names is None:
            if species == 'human':
                from pygot.utils import TF_human
                TF_names = TF_human
            elif species == 'mm':
                from pygot.utils import TF_mm
                TF_names = TF_mm
            else:
                raise NotImplementedError('Default database do NOT contains TF list of speices{}. Please specify the `TF_names` parameter'.format(species))
    else:
        TF_names = adata.uns['gene_name']
    TF_names = pd.Index(TF_names).intersection(pd.Index(adata.uns['gene_name']))
    adata.uns['tf_name'] = TF_names
    print("TF number: {}, {}".format(len(TF_names), TF_names))
                                            
    if layer_key is None:
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        scale = np.mean(adata.X[adata.X > 0]) / np.mean(abs(adata.layers['velocity']))
    else:
        if issparse(adata.layers[layer_key]):
            adata.layers[layer_key] = adata.layers[layer_key].toarray()
        scale = np.mean(adata.layers[layer_key][adata.layers[layer_key] > 0]) / np.mean(abs(adata.layers['velocity']))
    print('scale velocity with factor : {}'.format(scale))
    adata.layers['scaled_velocity'] = scale * adata.layers['velocity']
    
    if lineage_key is not None:
        
        lineages = np.unique(adata.obs[lineage_key])
        lineages = lineages[lineages != 'uncertain']
        adatas = [adata[adata.obs.loc[adata.obs[lineage_key] == lineages[i]].index] for i in range(len(lineages))]
        if not isinstance(init_beta, GeneDegradation):
            print(f"Using whold dataset to estimate degradation..")
            _, beta_hat, _ = optimize_global_GRN(
                adata,
                TF_names=TF_names,
                layer_key=layer_key, 
                beta_grad=True,
                num_epochs=num_epochs, 
                lr=lr, 
                l1_penalty=l1_penalty, 
                init_beta=init_beta, 
                min_beta=min_beta, 
                init_jacobian=init_jacobian, 
                device=device, 
                early_stopping=early_stopping, 
                batch_size=batch_size, 
                val_split=val_split, 
                non_negative=non_negative
            )
        else:
            beta_hat = init_beta
        grns = {}  
        adata.obs['global_grn_fit_godness'] = np.nan
        for i in range(len(lineages)):
            print(f"Training GRN for lineage: {lineages[i]}")
            G_hat, beta_hat, fit_godness = optimize_global_GRN(
                adatas[i],
                TF_names=TF_names,
                layer_key=layer_key, 
                beta_grad=False,  # Do not update beta for each individual GRN
                num_epochs=num_epochs, 
                lr=lr, 
                l1_penalty=l1_penalty, 
                init_beta=beta_hat, 
                min_beta=min_beta, 
                init_jacobian=init_jacobian, 
                device=device, 
                early_stopping=early_stopping, 
                batch_size=batch_size, 
                val_split=val_split, 
                non_negative=non_negative
            ) 
            grn = GRN(G_hat, beta_hat, adata.uns['tf_name'], adata.uns['gene_name'])
            grns[lineages[i]] = grn
            adata.obs.loc[adatas[i].obs.index, 'global_grn_fit_godness'] = fit_godness
            
        return grns
    else:
        G_hat, beta_hat, fit_godness = optimize_global_GRN(
                adata,
                TF_names=TF_names,
                layer_key=layer_key, 
                beta_grad=True,
                num_epochs=num_epochs, 
                lr=lr, 
                l1_penalty=l1_penalty, 
                init_beta=init_beta, 
                min_beta=min_beta, 
                init_jacobian=init_jacobian, 
                device=device, 
                early_stopping=early_stopping, 
                batch_size=batch_size, 
                val_split=val_split, 
                non_negative=non_negative
            )
        adata.obs['global_grn_fit_godness'] = fit_godness
        grn = GRN(G_hat, beta_hat, adata.uns['tf_name'], adata.uns['gene_name'])
        return grn