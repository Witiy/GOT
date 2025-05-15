import torch.nn as nn
import torch
import ot
import numpy as np



class DensityLoss(nn.Module):
    def __init__(self, device=torch.device('cpu'), hinge_value=0.01):
        super().__init__()
        self.hinge_value = hinge_value
        self.device = device

    def __call__(self, source, target, groups = None, to_ignore = None, top_k = 5):
        if groups is not None:
            # for global loss
            c_dist = torch.stack([
                torch.cdist(source[i].to(self.device), target[i].to(self.device)) 
                # NOTE: check if this should be 1 indexed
                for i in range(1,len(groups))
                if groups[i] != to_ignore
            ])
        else:
            # for local loss
             c_dist = torch.stack([
                torch.cdist(source.to(self.device), target.to(self.device))                 
            ])
        values, _ = torch.topk(c_dist, top_k, dim=2, largest=False, sorted=False)
        values -= self.hinge_value
        values[values<0] = 0
        loss = torch.mean(values)
        return loss


    
class OTLoss(nn.Module):
    _valid = 'emd sinkhorn sinkhorn_knopp_unbalanced'.split()

    def __init__(self, which='emd', device=torch.device('cpu')):
        if which not in self._valid:
            raise ValueError(f'{which} not known ({self._valid})')
        elif which == 'emd':
            self.fn = lambda m, n, M: ot.emd(m, n, M)
        elif which == 'sinkhorn':
            self.fn = lambda m, n, M : ot.sinkhorn(m, n, M, 2.0)
        elif which == 'sinkhorn_knopp_unbalanced':
            self.fn = lambda m, n, M : ot.unbalanced.sinkhorn_knopp_unbalanced(m, n, M, 1.0, 1.0)
        else:
            pass
        self.device = device

    def __call__(self, source, target):
        
        mu = torch.from_numpy(ot.unif(source.size()[0])).float()
        nu = torch.from_numpy(ot.unif(target.size()[0])).float()
        M = torch.cdist(source, target)**2
        #print(torch.isnan(M).sum())
        pi = self.fn(mu, nu, M.detach().cpu())
        if type(pi) is np.ndarray:
            pi = torch.tensor(pi)
        elif type(pi) is torch.Tensor:
            pi = pi.clone().detach()
        pi = pi.to(self.device)
        M = M.to(pi.device)
        loss = torch.sum(pi * M)
        return loss


class XCentricLoss(nn.Module):
    def __init__(self , lambda_ot=1, lambda_density=5, hinge_value=0.1, top_k=5, device=torch.device('cpu'), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.ot_fn = OTLoss(device=device)
        self.density_fn = DensityLoss(device=device, hinge_value=hinge_value)
        self.top_k = top_k
        self.lambda_ot = lambda_ot
        self.lambda_density = lambda_density
        self.device = device
        
    def forward(self, X_pred, X, ts):
        
            
        ot_loss = sum([
                self.ot_fn(X_pred[i].to(self.device), X[i].to(self.device)) 
                for i in range(1, len(ts))
                
            ])
        
        density_loss = self.density_fn(X_pred, X, groups=ts, top_k=self.top_k)
        
        
        return self.lambda_ot * ot_loss + self.lambda_density * density_loss


        
                  




    