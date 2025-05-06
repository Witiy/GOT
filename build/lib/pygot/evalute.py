from itertools import product, permutations, combinations, combinations_with_replacement
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import numpy as np
import pandas as pd
import random
import copy
from .tools.traj.utils import cosine, find_neighbors
# This function is from BoolODE package
def computeScores(trueEdgesDF, predEdgeDF, 
                  directed = True, selfEdges = True):
    '''        
    Computes precision-recall and ROC curves
    using scikit-learn for a given set of predictions in the 
    form of a DataFrame.
    

    :param trueEdgesDF:   A pandas dataframe containing the true classes.The indices of this dataframe are all possible edgesin a graph formed using the genes in the given dataset. This dataframe only has one column to indicate the classlabel of an edge. If an edge is present in the reference network, it gets a class label of 1, else 0.
    :type trueEdgesDF: DataFrame
        
    :param predEdgeDF:   A pandas dataframe containing the edge ranks from the prediced network. The indices of this dataframe are all possible edges.This dataframe only has one column to indicate the edge weightsin the predicted network. Higher the weight, higher the edge confidence.
    :type predEdgeDF: DataFrame
    
    :param directed:   A flag to indicate whether to treat predictionsas directed edges (directed = True) or undirected edges (directed = False).
    :type directed: bool
    :param selfEdges:   A flag to indicate whether to includeself-edges (selfEdges = True) or exclude self-edges (selfEdges = False) from evaluation.
    :type selfEdges: bool
        
    :returns:
            - prec: A list of precision values (for PR plot)
            - recall: A list of precision values (for PR plot)
            - fpr: A list of false positive rates (for ROC plot)
            - tpr: A list of true positive rates (for ROC plot)
            - AUPRC: Area under the precision-recall curve
            - AUROC: Area under the ROC curve
    '''

    if directed:        
        # Initialize dictionaries with all 
        # possible edges
        if selfEdges:
            possibleEdges = list(product(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                         repeat = 2))
        else:
            possibleEdges = list(permutations(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                         r = 2))
        
        TrueEdgeDict = {'|'.join(p):0 for p in possibleEdges}
        PredEdgeDict = {'|'.join(p):0 for p in possibleEdges}
        
        # Compute TrueEdgeDict Dictionary
        # 1 if edge is present in the ground-truth
        # 0 if edge is not present in the ground-truth
        for key in TrueEdgeDict.keys():
            if len(trueEdgesDF.loc[(trueEdgesDF['Gene1'] == key.split('|')[0]) &
                   (trueEdgesDF['Gene2'] == key.split('|')[1])])>0:
                    TrueEdgeDict[key] = 1
                
        for key in PredEdgeDict.keys():
            subDF = predEdgeDF.loc[(predEdgeDF['Gene1'] == key.split('|')[0]) &
                               (predEdgeDF['Gene2'] == key.split('|')[1])]
            if len(subDF)>0:
                PredEdgeDict[key] = np.abs(subDF.EdgeWeight.values[0])

    # if undirected
    else:
        
        # Initialize dictionaries with all 
        # possible edges
        if selfEdges:
            possibleEdges = list(combinations_with_replacement(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                                               r = 2))
        else:
            possibleEdges = list(combinations(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                                               r = 2))
        TrueEdgeDict = {'|'.join(p):0 for p in possibleEdges}
        PredEdgeDict = {'|'.join(p):0 for p in possibleEdges}

        # Compute TrueEdgeDict Dictionary
        # 1 if edge is present in the ground-truth
        # 0 if edge is not present in the ground-truth

        for key in TrueEdgeDict.keys():
            if len(trueEdgesDF.loc[((trueEdgesDF['Gene1'] == key.split('|')[0]) &
                           (trueEdgesDF['Gene2'] == key.split('|')[1])) |
                              ((trueEdgesDF['Gene2'] == key.split('|')[0]) &
                           (trueEdgesDF['Gene1'] == key.split('|')[1]))]) > 0:
                TrueEdgeDict[key] = 1  

        # Compute PredEdgeDict Dictionary
        # from predEdgeDF

        for key in PredEdgeDict.keys():
            subDF = predEdgeDF.loc[((predEdgeDF['Gene1'] == key.split('|')[0]) &
                               (predEdgeDF['Gene2'] == key.split('|')[1])) |
                              ((predEdgeDF['Gene2'] == key.split('|')[0]) &
                               (predEdgeDF['Gene1'] == key.split('|')[1]))]
            if len(subDF)>0:
                PredEdgeDict[key] = max(np.abs(subDF.EdgeWeight.values))

                
                
    # Combine into one dataframe
    # to pass it to sklearn
    outDF = pd.DataFrame([TrueEdgeDict,PredEdgeDict]).T
    outDF.columns = ['TrueEdges','PredEdges']
    
    fpr, tpr, thresholds = roc_curve(y_true=outDF['TrueEdges'],
                                     y_score=outDF['PredEdges'], pos_label=1)

    prec, recall, thresholds = precision_recall_curve(y_true=outDF['TrueEdges'],
                                                      probas_pred=outDF['PredEdges'], pos_label=1)
    
    return prec, recall, fpr, tpr, auc(recall, prec), auc(fpr, tpr)

def compute_pr(true_df, pred_df):
    _, _, _, _, pr, roc = computeScores(true_df, pred_df, selfEdges=False, directed=True)
    return pr




def compute_epr(true_df, pred_df, gene_num, sign=True):
    true_df = true_df.loc[true_df.Gene1 != true_df.Gene2]
    pred_df = pred_df.loc[pred_df.Gene1 != pred_df.Gene2]
    pred_df = pred_df.loc[pred_df.index[:len(true_df)]]
    baseline = (len(true_df) * 1.) / ( (gene_num * (gene_num - 1)) )
    
    if sign:
        pred_df['Type'] = pred_df.EdgeWeight.apply(lambda x: '+' if x > 0 else '-')
        precision = len(set(true_df.apply(lambda x : x.Gene1 + '->' +x.Gene2 + x.Type, axis=1)) \
            & set(pred_df.apply(lambda x : x.Gene1 + '->' +x.Gene2 + x.Type, axis=1))) 
    else:
        precision = len(set(true_df.apply(lambda x : x.Gene1 + '->' +x.Gene2, axis=1)) \
            & set(pred_df.apply(lambda x : x.Gene1 + '->' +x.Gene2, axis=1))) 
    precision /= len(pred_df)
    #print(precision, baseline, precision / baseline)
    return precision / baseline, pred_df

def compute_random_pr(true_df, pred_df, pool, t=100):
    
    pr_list, roc_list = [], []
    for i in range(t):
        random.shuffle(pool)
        random_edge = copy.deepcopy(pool[:len(pred_df)])

        weight = pred_df.EdgeWeight.tolist()
        for j in range(len(weight)):
            random_edge[j].append(weight[j])
        random_edge = pd.DataFrame(random_edge, columns=pred_df.columns[:3])
        _, _, _, _, a, b = computeScores(true_df, random_edge, selfEdges=False, directed=True)
        pr_list.append(a)
        roc_list.append(b)
    
    return np.mean(pr_list)

def compute_epr_ratio(true_df, pred_df,  pool, t=100):
    true_df = true_df.loc[true_df.Gene1 != true_df.Gene2]
    epr, pred_df = compute_epr(true_df, pred_df)
    random_epr = compute_random_pr(true_df, pred_df, pool, t)
    return epr, epr / random_epr


def compute_pr_ratio(true_df, pred_df, pool, t=100):
    
    _, _, _, _, pr, roc = computeScores(true_df, pred_df, selfEdges=False, directed=True)
    random_pr = compute_random_pr(true_df, pred_df, pool, t)
    return pr, random_pr


def incluster_coherence(adata, cell_type_key, velocity_key='velocity'):
    n = len(adata)
    neighbors = find_neighbors(adata.obsp['distances'])
    cell_types = adata.obs[cell_type_key].to_numpy()
    incluster_neighbors =[np.array(neighbors[i])[cell_types[neighbors[i]] == cell_types[i]] for i in range(n)]
    v = adata.layers[velocity_key]
    v = v[:, np.isnan(v).sum(axis=0) == 0]
    icc = np.array([np.mean(cosine(v[i][None,:], v[incluster_neighbors[i]])) for i in range(n)])
    adata.obs['icc'] = icc
    mean_icc = np.nanmean(icc)
    print('Mean ICC:{:.4f}'.format(mean_icc))
    return mean_icc