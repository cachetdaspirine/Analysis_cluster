import numpy as np

def nonzero_points(arr):
    """Returns all points in `arr` that are not equal to [0, 0, 0].
    
    Args:
        arr (numpy.ndarray): A 2D or 3D array of shape (Lx, N, 3) or (N, 3)
    
    Returns:
        numpy.ndarray: A 2D array of shape (M, 3) where M is the number of non-zero points in `arr`
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
