import numpy as np

from abc import ABC
from collections import namedtuple
from matplotlib import animation
from typing import List
from acrobotics.geometry import Scene
from acrobotics.link import Link
from acrolib.plotting import plot_reference_frame

JointLimit = namedtuple("JointLimit", ["lower", "upper"])


class IKResult:
    def __init__(self, success: bool, solutions: List[np.ndarray] = None):
        self.success = success
        if self.success:
            assert solutions is not None
            self.solutions = solutions


class Tool(Scene):
    """ Geometric shapes with added atribute tool tip transform tf_tt
     relative to the last link.
    """

    def __init__(self, shapes, tf_shapes, tf_tool_tip):
        """
        tf_tool_tip relative to last link robot.
        """
        super().__init__(shapes, tf_shapes)
        self.tf_tool_tip = tf_tool_tip


class RobotKinematics:
    """ Robot kinematics and shape

    (inital joint values not implemented)
    """

    def __init__(self, links: List[Link], joint_limits: List[JointLimit] = None):
        self.links = links
        self.ndof = len(links)

        # set default joint limits
        if joint_limits is None:
            self.joint_limits = [JointLimit(-np.pi, np.pi)] * self.ndof
        else:
            self.joint_limits = joint_limits

        # pose of base with respect to the global reference frame
        # this is independent from the geometry of the base,
        # for the whole robot
        self.tf_base = np.eye(4)
        # pose of the tool tip relative to last link robot.
        self.tf_tool_tip = None

    def fk(self, q) -> np.ndarray:
        """ Return end effector frame, either last link, or tool frame
        if tool available
        """
        T = self.tf_base
        for i in range(0, self.ndof):
            Ti = self.links[i].get_link_relative_transform(q[i])
            T = T @ Ti
        if self.tf_tool_tip is not None:
            T = np.dot(T, self.tf_tool_tip)
        return T

    def fk_all_links(self, q) -> List[np.ndarray]:
        """ Return link frames (not base or tool)
        """
        tf_links = []
        T = self.tf_base
        for i in range(0, self.ndof):
            Ti = self.links[i].get_link_relative_transform(q[i])
            T = T @ Ti
            tf_links.append(T)
        return tf_links

    def ik(self, transformation_matrix) -> IKResult:
        raise NotImplementedError

    def estimate_max_extension(self):
        max_ext = 0
        for link in self.links:
            max_ext += abs(link.dh.a) + abs(link.dh.d)
        return max_ext

    def plot_kinematics(self, ax, q, *arg, **kwarg):
        # base frame (0)
        plot_reference_frame(ax, self.tf_base)

        # link frames (1-ndof)
        tf_links = self.fk_all_links(q)
        points = [tf[0:3, 3] for tf in tf_links]
        points = np.array(points)
        points = np.vstack((self.tf_base[0:3, 3], points))
        ax.plot(points[:, 0], points[:, 1], points[:, 2], "o-")
        for tfi in tf_links:
            plot_reference_frame(ax, tfi)

        # tool tip frame
        if self.tf_tool_tip is not None:
            tf_tt = np.dot(tf_links[-1], self.tf_tool_tip)
            plot_reference_frame(ax, tf_tt)


class RobotCasadiKinematics(ABC):
    ndof: int
    links: List[Link]
    tf_base: np.ndarray
    tf_tool_tip: np.ndarray

    def fk_casadi(self, q):
        T = self.tf_base
        for i in range(0, self.ndof):
            Ti = self.links[i].get_link_relative_transform_casadi(q[i])
            T = T @ Ti
        if self.tf_tool_tip is not None:
            T = T @ self.tf_tool_tip
        return T

    def fk_all_links_casadi(self, q):
        """ Return link frames (not base or tool)
        """
        tf_links = []
        T = self.tf_base
        for i in range(0, self.ndof):
            Ti = self.links[i].get_link_relative_transform_casadi(q[i])
            T = T @ Ti
            tf_links.append(T)
        return tf_links


class Robot(RobotKinematics, RobotCasadiKinematics):
    def __init__(self, links, joint_limits=None):
        super().__init__(links, joint_limits)

        # defaul has no fixed base geometry, no tool and
        self.geometry_base = None
        self.geometry_tool = None

        self.do_check_self_collision = True
        # self collision matrix
        # default: do not check link neighbours, create band structure matrix
        temp = np.ones((self.ndof, self.ndof), dtype="bool")
        self.collision_matrix = np.tril(temp, k=-3) + np.triu(temp, k=3)

        # keep track of most likly links to be in collision
        self.collision_priority = list(range(self.ndof))

        # loggers to get performance criteria
        self.cc_checks = 0

    @property
    def tool(self):
        return self.geometry_tool

    @tool.setter
    def tool(self, new_tool: Tool):
        self.tf_tool_tip = new_tool.tf_tool_tip
        self.geometry_tool = new_tool

    def _check_self_collision(self, tf_links, geom_links):
        for i, ti, gi in zip(range(self.ndof), tf_links, geom_links):
            for j, tj, gj in zip(range(self.ndof), tf_links, geom_links):
                if self.collision_matrix[i, j]:
                    if gi.is_in_collision(gj, tf_self=ti, tf_other=tj):
                        return True

        # do not check tool against the last link where it is mounted
        if self.geometry_tool is not None:
            tf_tool = tf_links[-1]
            for tf_link, geom_link in zip(tf_links[:-1], geom_links[:-1]):
                if geom_link.is_in_collision(
                    self.geometry_tool, tf_self=tf_link, tf_other=tf_tool
                ):
                    return True
        return False

    def _is_in_limits(self, q):
        for qi, limit in zip(q, self.joint_limits):
            if qi > limit.upper or qi < limit.lower:
                return False
        return True

    @staticmethod
    def _linear_interpolation_path(q_start, q_goal, max_q_step):
        q_start, q_goal = np.array(q_start), np.array(q_goal)

        q_diff = np.linalg.norm(q_goal - q_start)
        num_steps = int(np.ceil(q_diff / max_q_step))

        S = np.linspace(0, 1, num_steps)
        return [(1 - s) * q_start + s * q_goal for s in S]

    def is_in_self_collision(self, q):
        geom_links = [l.geometry for l in self.links]
        tf_links = self.fk_all_links(q)
        return self._check_self_collision(tf_links, geom_links)

    def is_in_collision(self, q, collection):
        self.cc_checks += 1
        if collection is not None:
            geom_links = [l.geometry for l in self.links]
            tf_links = self.fk_all_links(q)

            # check collision with tool first
            if self.geometry_tool is not None:
                tf_tool = tf_links[-1]
                if self.geometry_tool.is_in_collision(collection, tf_self=tf_tool):
                    return True

            # check collision of fixed base geometry
            base = self.geometry_base
            if base is not None:
                if base.is_in_collision(collection, tf_self=self.tf_base):
                    return True

            # check collision for all links
            for i in self.collision_priority:
                if geom_links[i].is_in_collision(collection, tf_self=tf_links[i]):
                    # move current index to front of priority list
                    self.collision_priority.remove(i)
                    self.collision_priority.insert(0, i)
                    return True

        if self.do_check_self_collision:
            if self._check_self_collision(tf_links, geom_links):
                return True
        return False

    def is_path_in_collision_discrete(
        self, q_start, q_goal, collection, max_q_step=0.1
    ):
        """ Check for collision with linear interpolation between start and goal.
        """
        for q in self._linear_interpolation_path(q_start, q_goal, max_q_step):
            if self.is_in_collision(q, collection):
                return True
        return False

    def is_path_in_collision(self, q_start, q_goal, collection: Scene):
        """ Check for collision using the continuous collision checking
        stuff from fcl.
        - We do not check for self collision on a path.
        - Base is assumed not to move.
        """
        geom_links = [l.geometry for l in self.links]
        tf_links = self.fk_all_links(q_start)
        tf_links_target = self.fk_all_links(q_goal)

        # check collision with tool first
        if self.geometry_tool is not None:
            if self.geometry_tool.is_path_in_collision(
                tf_links[-1], tf_links_target[-1], collection
            ):
                return True

        # Base is assumed to be always fixed
        base = self.geometry_base
        if base is not None:
            if base.is_in_collision(collection, tf_self=self.tf_base):
                return True

        # check collision for all links
        for i in self.collision_priority:
            if geom_links[i].is_path_in_collision(
                tf_links[i], tf_links_target[i], collection
            ):
                # move current index to front of priority list
                self.collision_priority.remove(i)
                self.collision_priority.insert(0, i)
                return True
        return False

    def plot(self, ax, q, *arg, **kwarg):
        if self.geometry_base is not None:
            self.geometry_base.plot(ax, self.tf_base, *arg, **kwarg)

        tf_links = self.fk_all_links(q)
        for i, link in enumerate(self.links):
            link.plot(ax, tf_links[i], *arg, **kwarg)

        if self.geometry_tool is not None:
            self.geometry_tool.plot(ax, tf=tf_links[-1], *arg, **kwarg)

    def plot_path(self, ax, joint_space_path):
        alpha = np.linspace(1, 0.2, len(joint_space_path))
        for i, qi in enumerate(joint_space_path):
            self.plot(ax, qi, c=(0.1, 0.2, 0.5, alpha[i]))

    def animate_path(self, fig, ax, joint_space_path):
        def get_emtpy_lines(ax):
            lines = []
            for l in self.links:
                for s in l.geometry.shapes:
                    lines.append(s.get_empty_plot_lines(ax, c=(0.1, 0.2, 0.5)))
            if self.geometry_tool is not None:
                for s in self.geometry_tool.shapes:
                    lines.append(s.get_empty_plot_lines(ax, c=(0.1, 0.2, 0.5)))
            return lines

        def update_lines(frame, q_path, lines):
            tfs = self.fk_all_links(q_path[frame])
            cnt = 0
            for tf_l, l in zip(tfs, self.links):
                for tf_s, s in zip(l.geometry.tf_s, l.geometry.s):
                    Ti = np.dot(tf_l, tf_s)
                    lines[cnt] = s.update_plot_lines(lines[cnt], Ti)
                    cnt = cnt + 1

            if self.geometry_tool is not None:
                for tf_s, s in zip(self.geometry_tool.tf_s, self.geometry_tool.s):
                    tf_j = np.dot(tfs[-1], tf_s)
                    lines[cnt] = s.update_plot_lines(lines[cnt], tf_j)
                    cnt = cnt + 1

        ls = get_emtpy_lines(ax)
        N = len(joint_space_path)
        self.animation = animation.FuncAnimation(
            fig, update_lines, N, fargs=(joint_space_path, ls), interval=200, blit=False
        )
