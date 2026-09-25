# Kimech

**Planar mechanism kinematics for Python.**

Kimech models planar rigid-body mechanisms declaratively and solves position,
velocity, and acceleration kinematics for one-DOF mechanisms with revolute and
prismatic joints.

The public workflow is centered on four concepts:

1. build a {py:class}`~kimech.Mechanism`;
2. prescribe one coordinate with a {py:class}`~kimech.KinematicDriver`;
3. call {py:func}`~kimech.solve`;
4. inspect the resulting {py:class}`~kimech.KinematicSolution`.

```python
from kimech import KinematicDriver, Mechanism, solve

mechanism = Mechanism("example")
# ... add links, points, and joints ...

driver = KinematicDriver(
    crank_joint,
    position=positions,
)

solution = solve(
    mechanism,
    driver=driver,
    initial_guess=initial_guess,
)
```

Kimech is currently a **one-DOF planar R/P kinematics library**. Dynamics,
multiple simultaneous Drivers, branch enumeration, pseudo-arclength
continuation, and motion-law objects are outside the 0.6.0 scope.

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started
```

```{toctree}
:maxdepth: 2
:caption: User guide

concepts
time-aware-kinematics
examples
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference
```
