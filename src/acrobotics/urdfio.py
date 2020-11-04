import numpy as np
from string import Template
from urdfpy import URDF
from acrolib.geometry import rotation_matrix_to_rpy

from acrobotics.shapes import Box
from acrobotics.geometry import Scene

# look up how paths work in libraries
# with open("templates/box.template.xml") as file:
#     _box_template = Template(file.read())
# with open("templates/joint.template.xml") as file:
#     _joint_template = Template(file.read())
# with open("templates/workobject.template.urdf") as file:
#     _urdf_template = Template(file.read())

_box_template = Template(
    """<link name="$name">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="$size"/>
    </geometry>
    <material name="green"/>
  </visual>
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="$size"/>
    </geometry>
  </collision>
</link>"""
)

_joint_template = Template(
    """<joint name="world_to_$name" type="fixed">
  <parent link="world"/>
  <child link="$name"/>
  <origin xyz="$xyz" rpy="$rpy" />
</joint>"""
)

_urdf_template = Template(
    """<?xml version="1.0"?>
<robot name="$name" xmlns:xacro="http://wiki.ros.org/xacro">
<link name="world"/>

<material name="green">
    <color rgba="0 0.8 0 1.0"/>
</material>

$content
</robot>"""
)


def list_to_string(lst):
    """ Convert [1.0, 2, 0] ==> '1.0 2 0'  """
    return " ".join([str(v) for v in lst])


def create_urdf_data(scene, names):
    """ Convert acrobotics scene to urdf_data """

    sizes = []
    positions = []
    rotations = []
    for box, tf_box in zip(scene.shapes, scene.tf_s):
        sizes.append([box.dx, box.dy, box.dz])
        positions.append(tf_box[:3, 3])
        rotations.append(rotation_matrix_to_rpy(tf_box[:3, :3]))

    boxes = []
    for name, size, xyz, rpy in zip(names, sizes, positions, rotations):
        new_box = {}
        new_box["name"] = name
        new_box["size"] = list_to_string(size)
        new_box["xyz"] = list_to_string(xyz)
        new_box["rpy"] = list_to_string(rpy)
        boxes.append(new_box)

    return boxes


def export_urdf(scene, name, path):
    """ Convert a Scene object to an urdf file.

    params
    ------
    scene : acrobotics.geometry.Scene
      A scene containing shapes and their transforms in the world frame.
    name : str
      The filename for the urdf, extension ".urdf" is optional.
      It will be added if not present.
    path : str
      Relative or absolute path to where the urdf file should be saved.
    """
    # add '.urdf' extension if it was not added by the user
    if not name.endswith(".urdf"):
        name += ".urdf"

    shape_names = ["shape_{}".format(i) for i in range(len(scene.shapes))]
    urdf_data = create_urdf_data(scene, shape_names)

    content = "\n"
    for box in urdf_data:
        content += _box_template.substitute(box) + "\n"
        content += _joint_template.substitute(box) + "\n"

        with open("{}/{}".format(path, name), "w") as file:
            file.write(_urdf_template.substitute({"name": name, "content": content}))


def parse_link(link):
    """ Assume a link has only a single collision object.
        Assume this collision object is a box.
        Assume the link named "world" has no collision objects.
    
    Parameters
    ----------
    link: a urdfpy.urdf.Link object
    """
    assert len(link.collisions) == 1
    c = link.collisions[0]
    assert c.geometry.box is not None

    size = c.geometry.box.size
    return Box(size[0], size[1], size[2])


def import_urdf(name, path):
    # add '.urdf' extension if it was not added by the user
    if not name.endswith(".urdf"):
        name += ".urdf"

    urdf = URDF.load("{}/{}".format(path, name))
    root = urdf.base_link.name

    shapes = {}
    for link in urdf.links:
        if len(link.collisions) == 0:
            continue
        else:
            shapes[link.name] = parse_link(link)

    tfs = []
    final_shapes = []
    for joint in urdf.joints:
        if joint.parent == root:
            tfs.append(joint.origin)
            final_shapes.append(shapes[joint.child])
    return Scene(final_shapes, tfs)
