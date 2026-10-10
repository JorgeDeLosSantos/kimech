# Kimech

**Planar mechanism kinematics for Python.**

Kimech models planar rigid-body mechanisms declaratively and solves position,
velocity, and acceleration kinematics for fully prescribed planar mechanisms
with revolute and prismatic joints, including Multi-DOF systems.

The public workflow is centered on four concepts:

1. build a {py:class}`~kimech.Mechanism`;
2. prescribe independent coordinates with one or more {py:class}`~kimech.KinematicDriver` objects;
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
    drivers=driver,
    initial_guess=initial_guess,
)
```

The **unreleased 0.7.0 development branch** supports multiple simultaneous
independent drivers for planar R/P mechanisms. Dynamics, automatic driver
selection, branch enumeration, pseudo-arclength continuation and continuous
motion-law objects remain outside this release.

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started
```

```{toctree}
:maxdepth: 2
:caption: User guide

concepts
kinematic-analysis
visualization
time-aware-kinematics
examples
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference
```
