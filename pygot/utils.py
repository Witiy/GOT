from scipy.stats import pearsonr
from scipy.sparse import issparse
from tqdm import tqdm

def calcu_dynamic_corr(adata, time_key='expectation', layer_key=None):
        corrs = []
        if layer_key is None:
            if issparse(adata.X):
                X = adata.X.toarray()
            else:
                X = adata.X
        else:
            X = adata.layers[layer_key]
        t = adata.obs[time_key].to_numpy()
        for i in tqdm(range(X.shape[1])):
            corrs.append(pearsonr(X[:,i], t)[0])
        adata.var['corr'] = corrs