import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.spatial import distance_matrix
sys.path.append('/home/hcleroy/PostDoc/aging_condensates/Simulation/Gillespie/Gillespie_backend/')
import Gillespie_backend as Gil

def compute_pair_prob_function(points_sets, weights, num_bins, max_distance,dimension = 3):
    n_sets = len(points_sets)
    assert n_sets == len(weights), "The number of point sets and weights must be the same"

    all_distances = []
    all_weights = []
    total_weighted_pairs = 0
    
    for points, weight in zip(points_sets, weights):
        dist_matrix = distance_matrix(points, points)
        distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        all_distances.extend(distances)
        
        n_points = len(points)
        n_pairs = n_points * (n_points - 1) // 2
        all_weights.extend([weight] * n_pairs)
        
        total_weighted_pairs += weight * n_pairs

    hist, bin_edges = np.histogram(all_distances, bins=num_bins, range=(0, max_distance), weights=all_weights)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    if dimension == 3:
        shell_volumes = (4 / 3) * np.pi * ((bin_centers + bin_widths)**3 - bin_centers**3)
    elif dimension ==2:
        shell_volumes = 4*np.pi * ((bin_centers + bin_widths)**2 - bin_centers**2)
    elif dimension==1:
        shell_volumes = 1
    else:
        raise NotImplementedError
    normalized_hist = hist / (total_weighted_pairs * shell_volumes)

    return bin_centers, normalized_hist

"""
def compute_2_body_dist_prob(R,bins=100):
    maxR = np.max(R)
    ddist = maxR/(bins-1)
    def I(dist):
        return int(dist/ddist)
    def v_shell(dist):
        return 4/3*np.pi*((dist+ddist)**3 - dist**3)
    X = np.linspace(0,maxR,bins)
    PR = np.zeros(bins,dtype=float)
    for r in R:
        for r1,r2 in zip(r[:r.shape[0]-1],r[1:]):
            if any(r1!=r2):
                try:
                    PR[I(np.linalg.norm(r1-r2))] +=1/v_shell(np.linalg.norm(r1-r2))*1/(R.shape[0]*R.shape[1])
                except IndexError:
                    pass
                    #PR.resize(I(np.linalg.norm(r1-r2))+1)
                    #PR[I(np.linalg.norm(r1-r2))] +=1/v_shell(np.linalg.norm(r1-r2))*1/(R.shape[0]*R.shape[1])
                    #X = np.linspace(0,maxR,I(np.linalg.norm(r1-r2))+1)
    return X,PR
def compute_2_body_dist_prob_from_dist(D,bins=100):
    maxD = np.max(D)
    ddist = maxD/(bins-1)
    def I(dist):
        return int(dist/ddist)
    def v_shell(dist):
        return 4/3*np.pi*((dist+ddist)**3 - dist**3)
    X = np.linspace(0,maxD,bins)
    PR = np.zeros(bins,dtype=float)
    for D in D:
        if D!=0:
            try:
                PR[I(D)] += 1/v_shell(D)*1/(D.shape[0]*D.shape[1])
            except IndexError:
                pass
    return X,PR
"""

def Concatenate_dists(Syss):
    """
    This function takes a list of replicas, and return a single array with all the distances between linkers
    """
    dists = np.zeros((Syss.__len__(),Syss[0].Nlinker*(Syss[0].Nlinker-1)//2),dtype=float)
    # first compute the distance between linkers for each replica:
    for n in range(Syss.__len__()):
        k=0
        for i in range(Syss[n].Nlinker):
            for j in range(0,i):
                dists[n,k] = np.linalg.norm(Syss[n].get_r()[i]-Syss[n].get_r()[j])
                k+=1
    #return a concatenate version of all the distances.
    return np.concatenate(dists,axis=0)
#np.concatenate(np.array([Rs_replica[j][i*dt:(i+1)*dt] for j in range(Nreplica)]),axis=0)
def twobody_dist_prob_from_dist(D,bins=100):
    """
    This function takes a list of distance, and compute the probability associated which each distance.
    It accounts for the fact that the linkers are in a 3d. Thus the probability of being at a distance
    D has to be corrected by the volume of the shell.
    """
    maxD = np.max(D)
    ddist = maxD/(bins-1)
    def I(dist):
        return int(dist/ddist)
    def v_shell(dist):
        return 4/3*np.pi*((dist+ddist)**3 - dist**3)
    X = np.linspace(0,maxD,bins)
    PR = np.zeros(bins,dtype=float)
    for d in D:
        if d!=0:
            try:
                PR[I(d)] += 1/v_shell(d)*1/(D.shape[0])
            except IndexError:
                pass
    return X,PR

def Time_to_reach_equilibrium(ell_tot,rho0,BindingEnergy,kdiff,seed,Nlinker,dimension,compute_step,step_size,epsilon,Nreplica,PLOT=False):
    """
    This function create a system, and make it evolve. It computes the pair correlation function on the fly every compute_step.
    Whenever the norm of the difference of two succesives pair correlation functions is smaller than epsilon : stop the simulation and
    return the time.

    We have the possibility to use several replica in parallel to compute the pair correlation function. This is especially important
    whenever there are very few linkers, or whenever the system is supposed to evolve quickly.

    compute_step : is the number of step ( the last ones) used to compute the pair correlation function
    step_size : is the number of steps of equilibration between each measurement
    """
    np.random.seed(seed)
    # make an array of Gillespie simulations:
    Syss = [Gil.Gillespie(  ell_tot=ell_tot,
                                rho0=rho0,
                                BindingEnergy=BindingEnergy,
                                kdiff=kdiff,
                                seed = s,
                                Nlinker=Nlinker,
                                dimension=dimension) for s in np.random.randint(0,100000,Nreplica)]
    # initialize the two probability distributions:
    D = Concatenate_dists(Syss)
    X,g1 = twobody_dist_prob_from_dist(D,bins=100)
    g2 = np.zeros(100,dtype=float)
    # setup a counter to avoid infinit loop
    counter=0
    # t is the time of the simulation:
    t=0
    # the simulation keeps going until the system is equilibrated
    while Norm(X,g1-g2)/Norm(X,g1+g2)>epsilon and counter < 100:
        # Dists in the concatenation of all the distances between linkers of all the replica for each steps in compute_step.
        Dists = []
        # make a serie of step between each computation of the two body distrib
        for i in range(compute_step//step_size):
            # time is used to compute the average time of all moves
            time = []
            for Sys in Syss:
                movetype,Dt = Sys.evolve(step_size)
                time.append(np.sum(Dt))
            # add up the average time of the moves to the total time
            t += np.mean(time)
            # extend the Dists array with the concatenation of the distances between linkers of all replicas
            Dists.extend(Concatenate_dists(Syss))
        Dists=np.array(Dists)
        # keep track of the previous g
        g2 = g1
        # actualize the new g
        X,g1 = twobody_dist_prob_from_dist(Dists,bins=100)
        print(Norm(X,g1-g2)/Norm(X,g1+g2))
        counter+=1
    if PLOT: 
        return t,X,g1,g2,Syss
    return t