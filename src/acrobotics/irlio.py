import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from acrolib.geometry import xyz_intrinsic_to_rot_mat

# from acrobotics.path import TolerancedNumber, TolEulerPt


class Commands:
    MOVEP = "movep"
    MOVEJ = "movej"
    MOVELIN = "movel"
    SET_REFERENCE = "set-reference"


class Sections:
    VARS = "variables"
    COMMANDS = "commands"
    CONSTRAINTS = "constraints"


def strs_to_floats(lst):
    return [float(e) for e in lst]


def parse_file(filepath):
    """ Parse a txt file with the industrial robot like
    task specification.
    There are three sections:
    - variables
    - commands
    - constraints
    """
    with open(filepath) as file:
        data = file.readlines()
        output = {"variables": {}, "commands": [], "constraints": {}}
        output["variables"]["default"] = np.eye(4)

        current_key = None
        current_ref = output["variables"]["default"]
        for line in data:
            line = line.rstrip()
            if line in output.keys():
                current_key = line
                continue
            if current_key is None:
                print("File did not start with 'variables' or 'commands'.")
            if line == "":
                continue
            if Commands.SET_REFERENCE in line:
                _, ref = line.split(" ")
                try:
                    current_ref = output["variables"][ref]
                    continue
                except:
                    print("Trying to set reference to unkown pose: {}".format(ref))

            if current_key == "variables":
                _, name, value = parse_variable(line, current_ref)
                output[current_key][name] = value
            elif current_key == "commands":
                output[current_key].append(parse_command(line))
            elif current_key == "constraints":
                ctype, name, lower, upper = parse_constraint(line)
                output[current_key][name] = {"type": ctype, "min": lower, "max": upper}

    # output["variables"] = {
    #     k: transform_to_dict(v) for k, v in output["variables"].items()
    # }

    return output


def guess_degree_or_radians(values):
    """ Assume that for an average larger than 2*pi, the values where given in degrees."""
    TRESHOLD = 2 * np.pi
    not_zero = [abs(v) for v in values if v != 0]
    if len(not_zero) == 0:
        # all values are zero, the guess does not mather
        return "rad"
    elif sum(not_zero) / len(not_zero) > TRESHOLD:
        return "deg"
    else:
        return "rad"


def parse_rotation(values):
    """ Parse the 3 / 4 values for rotation. Only roll pitch yaw support for now. """
    if len(values) == 4:
        raise NotImplementedError("Quaterion input not supported yet.")
    elif len(values) == 3:
        if guess_degree_or_radians(values) == "rad":
            return xyz_intrinsic_to_rot_mat(values)
        else:
            values = [v * np.pi / 180 for v in values]
            return xyz_intrinsic_to_rot_mat(values)
    else:
        raise ValueError(
            "Wrong number of values for the rotation: {}".format(str(values))
        )


def parse_variable(line, reference_frame):
    """
    Assign pose or joint values to variable names.
        config <name> <q1> <q2> ...
        pose <name> <x> <y> <z> <qx> <qy> <qz> <qw>
    For a pose, premultiply with the reference frame in wich
    it is expressed.
    """
    # print("Parsing var line: {}".format(line))
    v = line.split(" ")
    assert len(v) >= 3
    vtype = v[0]
    name = v[1]
    if vtype == "config":
        value = [float(e) for e in v[2:]]
    elif vtype == "pose":
        all_values = [float(e) for e in v[2:]]
        value = np.eye(4)
        value[:3, :3] = parse_rotation(all_values[3:])
        value[:3, 3] = all_values[:3]
        value = np.dot(reference_frame, value)
    else:
        print("Unkown variable type " + vtype)
    return vtype, name, value


def parse_command(line):
    """
    Motion commands using named configs / poses and constraints.
        movep <goal>
        movej <goal>
        movelin <goal>
        movecirc <goal>
    After all of the above commands we can add path constraints
        movelin <goal> constraints c1
    TODO: also allow goal constraints, not only path constraints.
    """
    # print("Parsing command line: {}".format(line))
    v = line.split(" ")
    assert len(v) >= 2
    if len(v) == 2:
        return {"type": v[0], "goal": v[1]}
    else:
        assert len(v) >= 4
        assert v[2] == "con"
        return {"type": v[0], "goal": v[1], "constraints": v[3:]}


def parse_constraint(line):
    """
    Constraints section:
        xyz <name> symmetric <x> <y> <z>
        xyz <name> minmax <xmin> <ymin> <zmin> <xmax> <ymax> <zmax>
        rpy <name> symmetric <r> <p> <y>
        rpy <name> minmax <rmin> <pmin> <ymin> <rmax> <pmax> <ymax>
    """
    # print("Parsing con line: {}".format(line))
    v = line.split(" ")
    assert len(v) >= 5
    ctype, name, bounds = v[0], v[1], v[2]
    if bounds == "symmetric":
        upper = strs_to_floats(v[3:])
        lower = [-e for e in upper]
    elif bounds == "minmax":
        lower = strs_to_floats(v[3:6])
        upper = strs_to_floats(v[6:])

    return ctype, name, lower, upper


def extract_weld_lines(irl_data):
    v = irl_data["variables"]
    c = irl_data["constraints"]
    previous_command = {}
    lines = []
    for command in irl_data["commands"]:
        if command["type"] == Commands.MOVELIN:
            tf_start = v[previous_command["goal"]]
            tf_goal = v[command["goal"]]

            con = []
            if "constraints" in command.keys():
                for constraint in command["constraints"]:
                    con.append(c[constraint])

            lines.append({"start": tf_start, "goal": tf_goal, "con": con})

        previous_command = command.copy()
    return lines


def tf_interpolations(tf_a, tf_b, num_points):
    """ Linear interpolation between two transform.

    Implemented using ugly numpy magic.
    """
    s = np.linspace(0, 1, num_points)
    S = np.repeat(s[:, None], 3, axis=1)
    start_goal = Rotation.from_matrix([tf_a[:3, :3], tf_b[:3, :3]])
    slerp = Slerp([0, 1], start_goal)
    rotations = slerp(s)
    translations = (1 - S) * tf_a[:3, 3] + S * tf_b[:3, 3]
    return translations, rotations


# def weld_lines_to_path(weld_lines, num_points):
#     paths = []
#     for wl in weld_lines:
#         path = []
#         for t, r in zip(*tf_interpolations(wl["start"], wl["goal"], num_points)):
#             rpy = r.as_euler("XYZ")
#             if len(wl["con"]) > 0:
#                 con0 = wl["con"][0]
#                 rpy_tol = []
#                 for v, lower, upper in zip(rpy, con0["min"], con0["max"]):
#                     if lower == upper:
#                         rpy_tol.append(v)
#                     else:
#                         rpy_tol.append(TolerancedNumber(lower, upper, v))
#                 path.append(TolEulerPt(t, rpy_tol))
#             else:
#                 path.append(TolEulerPt(t, rpy))
#         paths.append(path)
#     return paths


# def import_irl_paths(filepath, num_points=10):
#     irl_data = parse_file(filepath)
#     weld_lines = extract_weld_lines(irl_data)
#     return weld_lines_to_path(weld_lines, num_points)
