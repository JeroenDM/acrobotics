import numpy as np


def is_in_range(x, lower, upper):
    if x > upper or x < lower:
        return False
    else:
        return True


def check_rxyz_input(rxyz):
    if np.allclose(abs(rxyz[1]), np.pi / 2):
        return False
    if not is_in_range(rxyz[0], -np.pi, np.pi):
        return False
    if not is_in_range(rxyz[1], -np.pi / 2, np.pi / 2):
        return False
    if not is_in_range(rxyz[2], -np.pi, np.pi):
        return False
    return True


def create_grid(r):
    """ Create an N dimensional grid from N arrays

    Based on N lists of numbers we create an N dimensional grid containing
    all possible combinations of the numbers in the different lists.
    An array can also be a single float if their is now tolerance range.

    Parameters
    ----------
    r : list of numpy.ndarray of float
        A list containing numpy vectors (1D arrays) representing a sampled
        version of a range along an axis.

    Returns
    -------
    numpy.ndarray
        Array with shape (M, N) where N is the number of input arrays and
        M the number of different combinations of the data in the input arrays.

    Examples
    --------
    >>> a = [np.array([0, 1]), np.array([1, 2, 3]), 99]
    >>> create_grid(a)
    array([[ 0,  1, 99],
           [ 1,  1, 99],
           [ 0,  2, 99],
           [ 1,  2, 99],
           [ 0,  3, 99],
           [ 1,  3, 99]])
    """
    grid = np.meshgrid(*r)
    grid = [grid[i].flatten() for i in range(len(r))]
    grid = np.array(grid).T
    return grid
