import matplotlib.pyplot as plt
from pygot.tools.jacobian import GRN
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def plot_target_gene(grn : GRN, gene_name : str, top_k=5, fontsize=10, **kwargs):

    if not isinstance(grn.pvalues_df, pd.DataFrame):
        filtered_jacobian_df = grn.gene_jacobian_df
    else:
        filtered_jacobian_df = (grn.pvalues_df < 0.05).to_numpy().astype(int) * grn.gene_jacobian_df
    x = filtered_jacobian_df.loc[filtered_jacobian_df[gene_name] != 0][gene_name].to_numpy()
    y = filtered_jacobian_df.index[filtered_jacobian_df[gene_name] != 0]
    idx = x.argsort()
    x = x[idx]
    y = y[idx]
    plt.figure(**kwargs)
    
    plt.scatter(range(len(x)), x, color='grey', s=10)
    if top_k > 0:
        plt.scatter(range(len(x)-top_k, len(x)), x[-top_k:], color='red', s=10)
        for i in range(top_k):
            plt.text(len(x)-i-1, x[len(x) - i - 1], y[len(x)-i-1], fontsize=fontsize, ha='right', va='center')
    else:
        plt.scatter(range(abs(top_k)), x[:abs(top_k)], color='red', s=10)
        for i in range(abs(top_k)):
            plt.text(i, x[i], y[i], fontsize=fontsize, ha='right', va='center')
    
    plt.xlabel('Gene Rank')
    plt.ylabel('Gene Scores')
    plt.title('{} Target Gene'.format(gene_name))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()


def plot_regulatory_gene(grn : GRN, gene_name : str, top_k=5, fontsize=10, **kwargs):
    if not isinstance(grn.pvalues_df, pd.DataFrame):
        filtered_jacobian_df = grn.gene_jacobian_df
    else:
        filtered_jacobian_df = (grn.pvalues_df < 0.05).to_numpy().astype(int) * grn.gene_jacobian_df

    x = filtered_jacobian_df.loc[gene_name][filtered_jacobian_df.loc[gene_name] != 0].to_numpy()
    y = filtered_jacobian_df.columns[filtered_jacobian_df.loc[gene_name] != 0]
    idx = x.argsort()
    x = x[idx]
    y = y[idx]
    plt.figure(**kwargs)
    
    plt.scatter(range(len(x)), x, color='grey', s=10)
    if top_k > 0:
        plt.scatter(range(len(x)-top_k, len(x)), x[-top_k:], color='red', s=10)
        for i in range(top_k):
            plt.text(len(x)-i-1, x[len(x) - i - 1], y[len(x)-i-1], fontsize=fontsize, ha='right', va='center')
    else:
        plt.scatter(range(abs(top_k)), x[:abs(top_k)], color='red', s=10)
        for i in range(abs(top_k)):
            plt.text(i, x[i], y[i], fontsize=fontsize, ha='right', va='center')

    plt.xlabel('Gene Rank')
    plt.ylabel('Gene Scores')
    plt.title('{} Regulatory Gene'.format(gene_name))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()



def plot_grn(ancestors, phenotype_genes=None, title='', **kwargs):
    # 创建一个有向图
    G = nx.DiGraph()
    
    # 添加节点和边
    for node, parents in ancestors.items():
        G.add_node(node)
        for parent in parents:
            G.add_node(parent)
            G.add_edge(parent, node)  # 从父节点到子节点添加边
    node_color=["lightblue"]*len(ancestors)
    if not phenotype_genes is None:
        for i, g in enumerate(G.nodes()):
            if g in phenotype_genes:
                node_color[i] = '#E95F42'
    # 绘制图形
    pos = nx.spring_layout(G, seed=42)  # 节点布局，seed用于确定布局的随机性
    plt.figure(**kwargs)
    nx.draw(G, with_labels=True, node_size=1500, node_color=node_color, pos=nx.circular_layout(G))
    plt.title(title)
    plt.show()