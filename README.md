[![Build Status](https://travis-ci.org/JeroenDM/acrobotics.svg?branch=master)](https://travis-ci.org/JeroenDM/acrobotics) [![codecov](https://codecov.io/gh/JeroenDM/acrobotics/branch/master/graph/badge.svg)](https://codecov.io/gh/JeroenDM/acrobotics)

# Acrobotics

Quickly test motion planning ideas is the goal, and Python seems like a great language for rapid prototyping. There are great libraries for robot simulation and related task, but installing them is can be a hassle and very dependent on operating system and python version.
The drawback is that I have to write a lot of stuff myself. I'm not sure if it is useful to do this. But it will be fun and I will learn a bunch.

This library provides robot kinematics and collision checking for serial kinematic chains. The idea is that this library can be easily swapped by another one providing the same functionality.

The acro part comes from [ACRO](https://iiw.kuleuven.be/onderzoek/acro) a robotics research group at KU Leuven in Belgium.

## Installation

```bash
pip install acrobotics
```

Or for development

```bash
git clone https://github.com/JeroenDM/acrobotics.git
cd acrobotics
python setup.py develop
```

No Windows support for the moment because [python-fcl](https://pypi.org/project/python-fcl/) is not supported. :(
In the future I will possibly switch to [pybullet](https://pypi.org/project/pybullet/). In the meantime, use [windows subsystem for linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10). MacOS is not tested yet.

## Gettings started

(Code for example below: [examples/getting_started.py](examples/getting_started.py))

This library has three main tricks.

### Robot kinematics
`T = robot.fk(joint_values)`
`IKSolution = robot.ik(T)`

Forward kinematics are implemented in a generic `RobotKinematics` class.
```python
from acrobotics.robot_examples import Kuka

robot = Kuka()

joint_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
T_fk = robot.fk(joint_values)
```

Analytical inverse kinematics only for specific robots:
```python
ik_solution = robot.ik(T_fk)  # T_fk is a numpy 4x4 array

print(f"Inverse kinematics successful? {ik_solution.success}")
for q in ik_solution.solutions:
    print(q)
```
```bash
Inverse kinematics successful? True
[ 0.1        -1.0949727   2.84159265  2.87778828  0.79803563 -1.99992985]
[ 0.1        -1.0949727   2.84159265 -0.26380438 -0.79803563  1.1416628 ]
[0.1 0.2 0.3 0.4 0.5 0.6]
[ 0.1         0.2         0.3        -2.74159265 -0.5        -2.54159265]
```

### Collision checking
`bool = robot.is_in_collision(joint_values, planning_scene)`

First create a planning scene with obstacles the robot can collide with.
```python
from acrobotics.geometry import Scene
from acrobotics.shapes import Box

table = Box(2, 2, 0.1)
T_table = translation(0, 0, -0.2)

obstacle = Box(0.2, 0.2, 1.5)
T_obs = translation(0, 0.5, 0.55)

scene = Scene([table, obstacle], [T_table, T_obs])
```

Then create a list of robot configurations for wich you want to check collision with the planning scene.
```python
import numpy as np

q_start = np.array([0.5, 1.5, -0.3, 0, 0, 0])
q_goal = np.array([2.5, 1.5, 0.3, 0, 0, 0])
q_path = np.linspace(q_start, q_goal, 10)
```

And then you could do:
```python
print([robot.is_in_collision(q, scene) for q in q_path])
```
```bash
[False, False, False, False, True, True, True, True, False, False]
```

### Visualization
`robot.plot(axes_handle, joint_values)`
`robot.animate_path(figure_handle, axes_handle, joint_path)`

```python
from acrolib.plotting import get_default_axes3d

fig, ax = get_default_axes3d()

scene.plot(ax, c="green")
robot.animate_path(fig, ax, q_path)
```

![animation](examples/robot_animation.gif)

## More details

There's a more in depth explanation in the jupyter-notebooks in the examples folder.

## And motion planning?

Comming soon.
