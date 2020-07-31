# This init file is a work in progress,
# I do not understand all the details of Python init files yet.
#
# Move stuff to top level module so we can do
#
# import acrobotics as ab
#
# scene = ab.Scene(...)
# shape = ab.Box(...)
#
# from acrobotics.shapes import Box
# from acrobotics.geometry import Scene
#
# and having too many import statements at the top of each file
# It is mainly meant for users, not for use in source files in
# the lirbary itself.
#
# I explicitly import all classes and functions to avoid importing
# names that where imported in the module file, and therefore importing
# the same name through multiple different paths.
#
# Also this clearly shows what is meant to be user facing.
# I should add _ to non user facing stuff...
#
#
from .robot import Tool
from .robot_examples import (
    PlanarArm,
    SphericalArm,
    SphericalWrist,
    AnthropomorphicArm,
    Arm2,
    Kuka,
    KukaOnRail,
)
from .tool_examples import torch, torch2, torch3
from .shapes import Box, Cylinder
from .geometry import Scene

# Path construction
from .path.path_pt import TolPositionPt, TolEulerPt, TolQuatPt, FreeOrientationPt
from .path.factory import create_line, create_circle, create_arc
from .path.tolerance import (
    NoTolerance,
    Tolerance,
    QuaternionTolerance,
    SymmetricTolerance,
)

# Planning settings and solvers
from .path.sampling import SamplingSetting, SearchStrategy
from .planning.settings import OptSettings, SolverSettings
from .planning.types import SolveMethod, CostFuntionType, PlanningSetup
from .planning.solver import solve
