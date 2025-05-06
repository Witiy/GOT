import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from pygot.evalute import *
from scipy.sparse import issparse

# 定义多元多任务回归模型
class MultiTaskRegression(nn.Module):
    def __init__(self, input_size, output_size, init_jacobian, beta_grad=True, init_beta=1., min_beta=0.0):
        super(MultiTaskRegression, self).__init__()
        self.linear = nn.Parameter(init_jacobian)
        self.linear.register_hook(self.remove_diagonal_hook)
        if beta_grad:
            self.beta = nn.Parameter(init_beta*torch.ones(output_size))
            self.beta.register_hook(self.hinge_hook)
            self.min_beta = torch.tensor(min_beta)
        else:
            self.beta = init_beta*torch.ones(output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return (self.linear @ x[:,:,None]).squeeze(-1) - self.relu(self.beta) * x
    
    def hinge_hook(self, grad):
        with torch.no_grad():
            self.beta.data = torch.clamp(self.beta, min=self.min_beta)
            #self.beta[self.beta < self.min_beta] = self.min_beta
        return grad
        
        

    def remove_diagonal_hook(self, grad):
        with torch.no_grad():
            self.linear -= torch.diag(torch.diag(self.linear))
            #self.linear.weight[self.linear.weight < 0] = 0
            
        return grad



class GRN:
    def __init__(self, G_hat:MultiTaskRegression, gene_names):
        self.model = G_hat
        self.G = G_hat.linear.detach().cpu().numpy()
        self.beta = G_hat.beta.data.detach().cpu().numpy()
        self.ranked_edges = get_ranked_edges(self.G, gene_names=gene_names)


def optimize_global_GRN(prototype_adata, time_key, layer_key=None,
                        beta_grad=True, num_epochs=100000, lr=0.01, l1_penalty = 0.005, init_beta=1.0, min_beta=1.0, 
                        coverage_cutoff=5e-3, true_df=None, init_jacobian=None, A=None, device=torch.device('cpu')):
    print('l1_penalty:', l1_penalty, 'min_beta:', min_beta)
    print('Coverage when weight change below {}'.format(coverage_cutoff))
    
    
    y = torch.Tensor(prototype_adata.layers['scaled_velocity'])
    
    if init_jacobian is None:

        init_jacobian = torch.rand(prototype_adata.X.shape[1],prototype_adata.X.shape[1])
    
    if layer_key is None:
        X_train = torch.Tensor(prototype_adata.X).to(device)
    else:
        X_train = torch.Tensor(prototype_adata.layers[layer_key]).to(device)
    y_train = y.to(device)


    # 初始化模型
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]  # 多任务回归的输出维度
    
    G_hat = MultiTaskRegression(input_size, output_size, init_jacobian, beta_grad, init_beta, min_beta).to(device)
    G_hat.min_beta = G_hat.min_beta.to(device)
    if true_df is not None:
        pred_df = get_ranked_edges(G_hat.linear.detach().cpu().numpy(), gene_names=prototype_adata.var.index)
        pr = compute_pr(true_df, pred_df)
        print('Init', pr)
    optimizer = optim.SGD(G_hat.parameters(), lr=lr)
    loss_list = []
    # 训练模型
    prev_weights = G_hat.linear.clone() 
    
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        optimizer.zero_grad()
        # 前向传播
        outputs = G_hat(X_train)
        
        #loss = torch.mean(weights * ((outputs - y_train) ** 2))
        mse_loss = torch.mean( ((outputs - y_train) ** 2))
        # L1正则化损失
        l1_loss = l1_penalty * torch.norm(G_hat.linear, p=1)
        loss = mse_loss + l1_loss
        # 反向传播和优化
        
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            # 检查权重是否收敛
            current_weights = G_hat.linear.clone()
            
            weight_change = torch.norm(current_weights - prev_weights)
            if weight_change < coverage_cutoff:  # 设置一个阈值，例如1e-4
                print(f'Converged at epoch {epoch+1}. Weight change: {weight_change.item():.5f}')
                break
    
            prev_weights = current_weights  # 更新前一次的权重
            if true_df is not None:
                matrix = G_hat.linear.detach().cpu().numpy()
                if A is not None:
                    matrix = A @ matrix @ A.T
                pred_df = get_ranked_edges(matrix, gene_names=prototype_adata.uns['gene_name'])
                pr = compute_pr(true_df, pred_df)
                epr, _ = compute_epr(true_df, pred_df, len(prototype_adata.var), False)
                pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], PR: {pr:.4f} EPR: {epr:.4f} | Weight change: {weight_change.item():.4f} | Loss: {loss.item():.4f} MSELoss: {mse_loss.item():.4f} L1Loss: {l1_loss.item():.4f}')
            else:
                pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}],  Weight change: {weight_change.item():.4f} MSE Loss: {mse_loss.item():.4f} L1 Loss: {l1_loss.item():.4f}')
            
            
        loss_list.append(loss.item())
        
    return G_hat, np.array(loss_list)


def get_ranked_edges(jacobian, gene_names, num_top=10000):
    gene_names = adata.uns['gene_name']
    
    df = pd.DataFrame(jacobian, index=gene_names, columns=gene_names).T
    
    stacked = df.stack()
    values = stacked.to_numpy().flatten()
    idx = np.argsort(values)
    #values[idx]
    top_idx = np.unique(np.concatenate([idx[:num_top], idx[-num_top:]]))
    gene1 = gene_names[top_idx // len(gene_names)]
    gene2 = gene_names[top_idx % len(gene_names)]
    result = pd.DataFrame([gene1, gene2, values[top_idx]], index=['Gene1', 'Gene2', 'EdgeWeight']).T
    result['absEdgeWeight'] = abs(result.EdgeWeight)
    result = result.sort_values('absEdgeWeight', ascending=False)
    return result

def infer_GRN(adata, time_key, layer_key=None, beta_grad=True, num_epochs=10000, lr=0.01, l1_penalty = 0.005, init_beta=1.0, min_beta=1.0, coverage_cutoff=5e-3, true_df=None, init_jacobian=None, A=None, device=torch.device('cpu')):
    if not 'velocity' in adata.layers.keys():
        raise KeyError('Please compute velocity first and store velocity in adata.layers')
    adata.uns['gene_name'] = adata.var.index
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
    G_hat, _ = optimize_global_GRN(adata, time_key, layer_key, beta_grad, num_epochs, lr, l1_penalty, init_beta, min_beta, coverage_cutoff, true_df, init_jacobian, A, device=device)
    grn = GRN(G_hat, adata.uns['gene_name'])
    return grn