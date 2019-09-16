import numpy as np

from acrolib.quaternion import Quaternion

from acrobotics.util import rot_x, rot_y, rot_z


def quat_distance(qa: Quaternion, qb: Quaternion):
    """ Half of the rotation angle to bring qa to qb."""
    return np.arccos(np.abs(qa.elements @ qb.elements))


def tf_inverse(T):
    """ Efficient inverse of a homogenous transform.

    (Normal matrix inversion would be a bad idea.)
    Returns a copy, not inplace!
    """
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].transpose()
    Ti[:3, 3] = np.dot(-Ti[:3, :3], T[:3, 3])
    return Ti


def xyz_intrinsic_to_rot_mat(rxyz):
    r_x, r_y, r_z = rxyz[0], rxyz[1], rxyz[2]
    return rot_x(r_x) @ rot_y(r_y) @ rot_z(r_z)


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


def rotation_matrix_to_rxyz(R):
    r11, r12, r13 = R[0]
    r21, r22, r23 = R[1]
    r31, r32, r33 = R[2]

    r_x = np.arctan2(-r23, r33)
    r_y = np.arctan2(r13, np.sqrt(r11 ** 2 + r12 ** 2))
    r_z = np.arctan2(-r12, r11)

    return [r_x, r_y, r_z]


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
