import numpy as np
from .ik_result import IKResult
from acrolib.geometry import rotation_matrix_to_rpy

TOLERANCE = 1e-12


def nearZero(value):
    return abs(value) < TOLERANCE


def ik(tf, links, tf_base) -> IKResult:
    """ Analytic inverse kinematics for 3 link robot
        
        Parameters
        ----------
        T : numpy.ndarray
            4 by 4 homgeneous transform matrix of the end-effector pose
        
        Returns
        -------
        IKResult
            Class with two attributes, success and solutions.
            Solutions constains a list with ndarrays of joint positions.
        
        Notes
        -----
        Joint limits [-pi, pi] are implied by the way the solution is calculated. 
        """
    # transform pose p to local base frame of this robot
    # explicilty write calculations for speed
    # R = rotation(self.base[2])

    # cos = np.cos(self.base[2])
    # sin = np.sin(self.base[2])

    # x = (p[0] - self.base[0]) * cos + (p[1] - self.base[1]) * sin
    # y = -(p[0] - self.base[0]) * sin + (p[1] - self.base[1]) * cos
    # phi = p[2] - self.base[2]

    rpy = rotation_matrix_to_rpy(tf[:3, :3])

    # check if the transform is valid for this planar robot
    # z-position zero and all but the z-rotation zero
    if not nearZero(rpy[0]) or not nearZero(rpy[1]) or not nearZero(tf[2, 3]):
        return IKResult(False)

    # # define variables for readability
    x, y, phi = (tf[0, 3], tf[1, 3], rpy[2])
    l1, l2, l3 = (links[0].dh.a, links[1].dh.a, links[2].dh.a)

    # print(f"IK solver: {x} {y} {phi}")

    # # initialize output
    q_up = np.zeros(3)
    q_do = np.zeros(3)
    # # q_up = [0, 0, 0]
    # # q_do = [0, 0, 0]
    reachable = False

    # start calculations
    if (l1 + l2 + l3) >= np.sqrt(x ** 2 + y ** 2):
        # coordinates of end point second link (w)
        pwx = x - l3 * np.cos(phi)
        pwy = y - l3 * np.sin(phi)
        rws = pwx ** 2 + pwy ** 2  # squared distance to second joint
        if (l1 + l2) >= np.sqrt(rws):
            # calculate cosine q2
            c2 = (rws - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
            # c2 is already guaranteed to be < 1
            # but it can still be smaller than -1
            # for example points close the the robot base that are
            # not reachable
            if c2 > -1:
                reachable = True
                # if c2 exactly 1, it can be a little bit bigger at this point
                if abs(c2 - 1) < TOLERANCE:
                    # first two links aligned and streched out
                    q_up[0] = np.arctan2(pwy, pwx)
                    q_up[1] = 0.0
                    q_up[2] = phi - q_up[0] - q_up[1]
                    return {"success": True, "q": [q_up]}
                elif abs(-c2 - 1) < TOLERANCE:
                    # first two links aligned and folded
                    q_up[0] = np.arctan2(pwy, pwx)
                    q_do[0] = np.arctan2(pwy, pwx)
                    q_up[1] = np.pi
                    q_do[1] = -np.pi
                    q_up[2] = phi - q_up[0] - q_up[1]
                    q_do[2] = phi - q_do[0] - q_do[1]
                    return {"success": True, "q": [q_up, q_do]}
                else:
                    # general reachable case
                    s2 = np.sqrt(1 - c2 ** 2)
                    q_up[1] = np.arctan2(s2, c2)  # elbow up
                    q_do[1] = np.arctan2(-s2, c2)  # elbow down
                # calculate q1
                temp = l1 + l2 * c2
                s1_up = (temp * pwy - l2 * s2 * pwx) / rws
                c1_up = (temp * pwx + l2 * s2 * pwy) / rws
                s1_do = (temp * pwy + l2 * s2 * pwx) / rws
                c1_do = (temp * pwx - l2 * s2 * pwy) / rws
                q_up[0] = np.arctan2(s1_up, c1_up)
                q_do[0] = np.arctan2(s1_do, c1_do)
                # finally q3
                q_up[2] = phi - q_up[0] - q_up[1]
                q_do[2] = phi - q_do[0] - q_do[1]

    if reachable:
        return IKResult(True, [q_up, q_do])
    else:
        return IKResult(False)
