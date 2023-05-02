import numpy as np

import math
from scipy.sparse import csr_matrix


import sknetwork as skn


from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
import random
cdict = {'blue':   ((0.0,  0.9, 0.9),
                    (0.5,  0.4, 0.4),
                    (1.0,  0.1, 0.1)),

         'green': ((0.0,  0.5, 0.5),
                   (0.5, 1, 1),
                   (1.0,  0.3, 0.3)),

         'alpha': ((0.0,  1, 1),
                   (0.5, 0.8, 0.8),
                   (1.0,  1, 1)),

         'red':  ((0.0,  0.4, 0.4),
                  (0.5,  0.5, 0.5),
                  (1.0,  0.9, 0.9)),
         }
def weight(distance):
    return 1/distance
def make_adjacency_matrix(array):
    adjacency = np.zeros((array.shape[0], array.shape[0]),dtype=float)
    for i in range(array.shape[0]):
        for j in range(i+1,array.shape[0]):
            adjacency[i,j] = weight(np.linalg.norm(array[i]-array[j]))
            adjacency[j,i] = adjacency[i,j]
    return adjacency
class Cluster:
    def __init__(self, array):
        """
        This function takes  an array of NxD number, where N is
        the number particles, and D the dimension. It first create
        and adjacency matrix, where the weight is proportional to
        the inverse of the distance between each points.
        """

        self.Nparticle = array.shape[0]
        self.dimension = array.shape[1]

        self.adjacency = make_adjacency_matrix(array)
    def make_cluster(self,resolution=1.):
        louvain = skn.clustering.Louvain(resolution=resolution)
        labels = louvain.fit_transform(self.adjacency)
        #propagation = skn.clustering.PropagationClustering()
        #labels = propagation.fit_transform(AdjacencyMatrix)
        return labels
def split_data_per_cluster(kmeans,data):
    """
    Given a kmeans object that we used to split the data into clusters.
    The data are then splitted according to the cluster where the belong.
    """
    data_out = []
    for i in range(kmeans.n_clusters):   
        kmeans.labels_
        data_out.append(data[np.argwhere(kmeans.labels_==i)[:,0]])
    return data_out
def compute_dist_vector(array):
    """
    return a 1D vector of distance between the points in the array.
    The shape of the array must be Nx3, where N is the number of points
    """
    if array.shape[0]==1:
        # return an unique 0 to allow the computation of the mean
        return np.array([0.])
    dists = np.zeros(array.shape[0]*(array.shape[0]-1)//2,dtype=float)
    k=0
    for i in range(array.shape[0]):
        for j in range(i):
            dists[k] = np.linalg.norm(array[i]-array[j])
            k+=1
    return dists
def compute_mutual_distance(A1,A2):
    """
    return a 1D vector of distance between the two arrays.
    """
    dists = np.zeros(A1.shape[0]*A2.shape[0],dtype=float)
    k=0
    for a1 in A1:
        for a2 in A2:
            dists[k] = np.linalg.norm(a1-a2)
            k+=1
    return dists