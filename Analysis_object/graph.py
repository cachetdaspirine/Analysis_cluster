import numpy as np
import sys
import networkx as nx
sys.path.append('/home/hcleroy/PostDoc/aging_condensates/Simulation/Aging_Condensate/Gillespie_backend')
import Gillespie_backend as backend


def weight(distance):
    if distance != 0:
        return 1/distance
    else :
        return 0.
def make_adjacency_matrix(array):
    adjacency = np.zeros((array.shape[0], array.shape[0]),dtype=float)
    for i in range(array.shape[0]):
        for j in range(i+1,array.shape[0]):
            adjacency[i,j] = weight(np.linalg.norm(array[i]-array[j]))
            adjacency[j,i] = adjacency[i,j]
    return adjacency
def clustering_coefficient(A):
    graph = nx.from_numpy_matrix(A)
    weighted_clustering_coef = nx.clustering(graph,weight='weight')
    return sum(weighted_clustering_coef.values()) / len(weighted_clustering_coef)
def clustering_distrib(A):
    graph = nx.from_numpy_matrix(A)
    return nx.clustering(graph,weight='weight').values()

