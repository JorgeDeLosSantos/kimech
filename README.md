# Kimech

Kimech is a small Python library for modeling and solving the kinematics of planar rigid-body mechanisms.

It provides declarative rigid-body models with revolute and prismatic joints, position solving with warm-start continuation, analytic velocity and acceleration kinematics, result queries, and schematic plotting and animation.

Kimech `0.2.0` supports position, velocity, and acceleration analysis for one-DOF planar R/P mechanisms with one prescribed joint coordinate. The `0.1.0` position-kinematics baseline is recorded in [`docs/design.md`](docs/design.md), the `0.2.0` design baseline in [`docs/design-0.2.0.md`](docs/design-0.2.0.md), and [`docs/api.md`](docs/api.md) documents the current implemented public API.

## Installation

Install the core package, which depends on NumPy and SciPy:

```bash
pip install -e .
```

Install the optional visualization dependencies (Matplotlib and Pillow) for plotting, animation, and GIF output:

```bash
pip install -e ".[viz]"
```

For development, install both development and visualization dependencies and run the tests:

```bash
pip install -e ".[dev,viz]"
pytest
```

## Quick start

The following builds a four-bar linkage and solves a differential kinematic sweep through the generic API:

```python
import numpy as np

from kimech import Mechanism, solve

mechanism = Mechanism("four_bar")
ground = mechanism.ground

ground_a = ground.add_point("A", (0.0, 0.0))
ground_d = ground.add_point("D", (0.30, 0.0))

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

input_joint = mechanism.revolute(ground_a, crank_a, name="input")
mechanism.revolute(crank_b, coupler_b)
mechanism.revolute(coupler_c, rocker_c)
mechanism.revolute(rocker_d, ground_d)

initial_guess = {
    crank: (0.0, 0.0, 0.8),
    coupler: (0.05, 0.06, 0.2),
    rocker: (0.30, 0.0, 2.2),
}
values = np.linspace(0.8, 1.3, 60)

solution = solve(
    mechanism,
    input=input_joint,
    values=values,
    input_velocity=1.5,
    input_acceleration=0.0,
    initial_guess=initial_guess,
)

positions = solution.point_path(point_p)
velocities = solution.point_velocities(point_p)
accelerations = solution.point_accelerations(point_p)
```

`values` are configuration parameters, not timestamps. `input_velocity` and `input_acceleration` are physical derivatives with respect to a common external time variable. Scalar differential inputs are broadcast across a sweep.

Position-only solving remains valid by omitting the differential inputs.

## Visualization

Visualization remains presentation-only and does not define physical time:

```python
import matplotlib.pyplot as plt
from kimech.visualization import animate

animation = animate(solution, fps=30)
plt.show()
```

## Examples

[`examples/four_bar.py`](examples/four_bar.py) and [`examples/slider_crank.py`](examples/slider_crank.py) are complete examples built with the generic public API.

See [`CHANGELOG.md`](CHANGELOG.md) for release changes and intentional breaking renames.
