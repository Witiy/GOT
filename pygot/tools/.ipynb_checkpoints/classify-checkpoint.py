from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection  import cross_val_score
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scanpy as sc

def filter_cell_type(adata, cell_type_key, cutoff=20):
    idx = adata.obs.index
    ct_list = pd.unique(adata.obs[cell_type_key])
    print('Check each cell num of cell type')
    for cell_type in ct_list:
        n = len(adata[idx].obs.loc[adata[idx].obs[cell_type_key] == cell_type])
        print(cell_type, n)
        if n < cutoff:
            print('**** Delect ', cell_type+" ****")
            idx = adata[idx].obs.loc[adata[idx].obs[cell_type_key] != cell_type].index
    return adata[idx]

def calcu_proba_entropy(df):
    def entropy(x):
        
        x = np.sort(x[x>0])
        if len(x) == 1:
            return 0.
        else:
            x = x[-2:]
        
        #x = x[x>0]
        return -np.sum(x*np.log2(x))
    return df.apply(lambda x:entropy(x.to_numpy()), axis=1).to_numpy()

class CellTypeClassifier:
    def __init__(self, adata : sc.AnnData, 
                 time_key: str, embedding_key: str, cell_type_key: str,
                 ) -> None:
        self.time_key = time_key
        self.embedding_key = embedding_key

        time_series = np.sort(np.unique(adata.obs[time_key]))
        adata = filter_cell_type(adata, cell_type_key, cutoff=20)

        self.X = adata.obsm[embedding_key]
        self.y = adata.obs[cell_type_key].to_numpy()
        
        
    
    def fit(self, max_k=21):
        k_error = []
        k_range = range(20, max_k)
        print('Searching best k..')
        for k in tqdm(k_range):
            knn = KNeighborsClassifier(n_neighbors=k, )
            scores = cross_val_score(knn, self.X, self.y, cv=6, scoring='accuracy')
            k_error.append(1 - scores.mean())
        k_error = np.array(k_error)
        best_k = np.argmin(k_error)
        print('Best K: {}, error: {}'.format(k_range[best_k], k_error[best_k]))

        self.clf = KNeighborsClassifier(n_neighbors=k_range[best_k],)
        self.clf.fit(self.X, self.y)
        print('Traning Accuracy ', self.clf.score(self.X, self.y))
        pred_cell_type_proba = self.predict_proba(self.X)
        pred_cell_type = self.predict(self.X)
        
        wrong_ent = calcu_proba_entropy(pred_cell_type_proba.loc[self.y != pred_cell_type])
        correct_ent = calcu_proba_entropy(pred_cell_type_proba.loc[self.y == pred_cell_type])
        
        sns.distplot(correct_ent, label='correct prediction prob entropy')
        sns.distplot(wrong_ent, label='wrong prediction prob entropy')
        
        self.transition_cutoff = np.median(wrong_ent)
        plt.axvline(x=self.transition_cutoff, label='transition state cutoff', c='red')
        plt.legend()
        plt.title('Classifier Predition Entropy')

    def predict_proba(self, x:np.array):

        res = self.clf.predict_proba(x)
        return pd.DataFrame(res, columns=self.clf.classes_)
    
    def predict_uncertainty(self, x:np.array, specify=False):
        proba = self.predict_proba(x)
        pred_cell_type = self.predict(x)
        ent = calcu_proba_entropy(proba)
        res = []
        for i in range(len(ent)):
            if ent[i] > self.transition_cutoff:
                if specify:
                    idx = proba.loc[proba.index[i]].to_numpy().argsort()
                    ab = np.sort([proba.columns[idx[-1]], proba.columns[idx[-2]]])
                    a, b = ab[0], ab[1]
                    res.append( a+'<->'+b )
                else:
                    res.append('transition')
            else:
                res.append(pred_cell_type[i])        
        return res
    
    def predict(self, x:np.array):
        return self.clf.predict(x)


 