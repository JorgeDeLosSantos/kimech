# Getting started

## Installation

For a local development checkout:

```bash
pip install -e .
```

For plotting and animation support:

```bash
pip install -e ".[viz]"
```

## Your first four-bar

```python
import numpy as np

from kimech import KinematicDriver, Mechanism, solve

mechanism = Mechanism("four_bar")

ground_a = mechanism.ground.add_point("A", (0.0, 0.0))
ground_d = mechanism.ground.add_point("D", (0.30, 0.0))

crank = mechanism.add_link("crank")
crank_a = crank.add_point("A", (0.0, 0.0))
crank_b = crank.add_point("B", (0.08, 0.0))

coupler = mechanism.add_link("coupler")
coupler_b = coupler.add_point("B", (0.0, 0.0))
coupler_c = coupler.add_point("C", (0.22, 0.0))
point_p = coupler.add_point("P", (0.10, 0.05))

rocker = mechanism.add_link("rocker")
rocker_c = rocker.add_point("C", (0.0, 0.0))
rocker_d = rocker.add_point("D", (0.18, 0.0))

crank_joint = mechanism.revolute(ground_a, crank_a, name="crank")
mechanism.revolute(crank_b, coupler_b)
mechanism.revolute(coupler_c, rocker_c)
mechanism.revolute(rocker_d, ground_d)

driver = KinematicDriver(
    crank_joint,
    position=np.linspace(0.8, 1.3, 60),
)

initial_guess = {
    crank: (0.0, 0.0, 0.8),
    coupler: (0.05, 0.06, 0.2),
    rocker: (0.30, 0.0, 2.2),
}

solution = solve(
    mechanism,
    driver=driver,
    initial_guess=initial_guess,
)

path = solution.point_positions(point_p)
```

A scalar Driver position still returns a {py:class}`~kimech.KinematicSolution`
with one sample. Access its configuration with `solution[0]`.

## What to read next

Read [Core concepts](concepts.md) for the model/result vocabulary, then
[Time-aware kinematics](time-aware-kinematics.md) if your prescribed coordinate
has physical velocity or acceleration.
