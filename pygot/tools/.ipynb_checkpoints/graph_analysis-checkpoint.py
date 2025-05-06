from collections import deque, defaultdict
import numpy as np
def iscyclic(gene1, gene2, childrens):
    queue = deque([gene2])
    visited = {}
    while queue:
        current_node = queue.popleft()
        visited[current_node] = True
        if not current_node in childrens.keys():
            continue
        if len(childrens[current_node]) == 0:
            continue
        for child in childrens[current_node]:
            if child == gene1:
                return True
            else:
                if not child in visited.keys():
                    queue.append(child)
    return False
def search_network_for_phenotype_genes(grn,  phenotype_genes, percentile=95, topk=None,):
    ranked_edges = grn.ranked_edges
    cutoff = np.percentile(ranked_edges.loc[ranked_edges.absEdgeWeight > 0.01].absEdgeWeight, percentile)
    ancestors = {g:[] for g in phenotype_genes}
    childrens = {g:[] for g in phenotype_genes}
    queue = deque(phenotype_genes)
    visited = {}

    while queue:
        current_node = queue.popleft()
    
        visited[current_node] = True
        connections = ranked_edges.loc[ranked_edges['Gene2'] == current_node]
        connections = connections.loc[connections.absEdgeWeight > cutoff]
        if not topk is None:
            connections = connections.loc[connections.index[:topk]]
        regulators = connections.Gene1.tolist()
        for reg in regulators:
            if not iscyclic(reg, current_node, childrens):
                if not reg in ancestors:
                    ancestors[reg] = []
                if not current_node in ancestors:
                    ancestors[current_node] = []
                ancestors[current_node].append(reg)
                if not reg in childrens:
                    childrens[reg] = []
                childrens[reg].append(current_node)
                if (not reg in visited.keys()) and (not reg in queue):
                    queue.append(reg)
    return layer_genes(childrens), ancestors, childrens
                

def build_graph_and_indegree(data):
    graph = defaultdict(list)
    indegree = defaultdict(int)
    
    # Build the graph and compute indegrees
    for node, ancestors in data.items():
        for ancestor in ancestors:
            graph[ancestor].append(node)
            indegree[node] += 1
        if node not in indegree:
            indegree[node] = 0  # Ensure all nodes are in the indegree map

    return graph, indegree

def topological_sort(graph, indegree):
    # Initialize the queue with nodes having zero indegree
    zero_indegree = deque([node for node in indegree if indegree[node] == 0])
    layers = []
    
    while zero_indegree:
        current_layer = []
        for _ in range(len(zero_indegree)):
            node = zero_indegree.popleft()
            current_layer.append(node)
            for neighbor in graph[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    zero_indegree.append(neighbor)
        layers.append(current_layer)
    
    return layers

def layer_genes(data):
    graph, indegree = build_graph_and_indegree(data)
    layers = topological_sort(graph, indegree)
    return layers