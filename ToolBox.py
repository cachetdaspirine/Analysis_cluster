import numpy as np
import math

MaxEnt = lambda N,L : 1.5* (N-1) * (np.log(3/(2*np.pi*L/N)) -1)

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

Ec = lambda L,N : -3/2*np.log(L/N* np.pi/3)

def V(L,Nlinker,dimension):
    if Nlinker ==1:
        return 4/3*np.pi*(L/6)**1.5
    else:
        return  2*((L/6)**1.5/Nlinker**0.5*3/4*np.pi)

def nonzero_points(arr):
    """Returns all points in `arr` that are not equal to [0, 0, 0].
    
    Args:
        arr (numpy.ndarray): A 2D or 3D array of shape (Lx, N, 3) or (N, 3)
    
    Returns:
        numpy.ndarray: A 2D array of shape (M, 3) where M is² the number of non-zero points in `arr`
    """
    arr = np.atleast_2d(arr)
    nonzero_mask = np.any(arr != [0,0,0], axis=1)
    nonzero_indices = np.argwhere(nonzero_mask)
    nonzero_points = arr[nonzero_mask]
    result = nonzero_points.reshape((-1, 3))
    return result


def linker_entropy(move, S):
    """
    This function takes two arrays 'move' and 'S' as input and returns a dictionary. The keys in the dictionary
    represent the number of bound linkers, and the values are 2D NumPy arrays containing the corresponding
    entropy values and their indices in the input array 'S'.
    
    Parameters:
    - move (list): An array where each entry is an integer between 0 and 3 (inclusive). A value of 3 represents
                   a linker binding, while a value of 0 represents a linker unbinding. Values 1 and 2 indicate
                   no change in the number of bound linkers.
    - S (list): An array containing the entropy of the system at each step in 'move'.
    
    Returns:
    - entropy_dict (dict): A dictionary where each key represents the number of bound linkers and the corresponding
                           value is a 2D NumPy array. Each row of the array contains the entropy for that number of bound linkers
                           and the index of the entropy value in the input array 'S'.
    """
    entropy_dict = {}
    bound_linkers = 0

    for i in range(len(move)):
        if move[i] == 3:
            bound_linkers += 1
        elif move[i] == 0:
            bound_linkers -= 1
            if bound_linkers < 0:
                bound_linkers = 0

        if bound_linkers not in entropy_dict:
            entropy_dict[bound_linkers] = []

        entropy_dict[bound_linkers].append((S[i], i))

    for key in entropy_dict:
        entropy_dict[key] = np.array(entropy_dict[key])

    return entropy_dict

def moving_average(X, Y, window_size):
    """
    Compute a moving average.
    
    Args:
        X: np.array of x values
        Y: np.array of y values
        window_size: size of the window for moving average.
        
    Returns:
        X_av: X values corresponding to the averaged Y values
        Y_av: Averaged Y values
    """
    window = np.ones(int(window_size))/float(window_size)
    Y_av = np.convolve(Y, window, 'valid')

    # For moving average with 'valid' mode, the length of Y_av is reduced 
    # at the beginning and end. We have to remove the corresponding X values.
    cut_size_start = (window_size - 1) // 2
    cut_size_end = (window_size - 1) - cut_size_start
    X_av = X[cut_size_start: -cut_size_end or None]

    return X_av,Y_av