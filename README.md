# Kimech

Kimech is a small Python library for modeling and solving the kinematics of planar rigid-body mechanisms.

It provides declarative rigid-body models with revolute and prismatic joints, position solving with warm-start continuation, result queries, and schematic plotting and animation.

Kimech `0.1.0` is the initial position-kinematics MVP. The project remains young, and its API may evolve in future versions. The conceptual design baseline lives in [`docs/design.md`](docs/design.md), while [`docs/api.md`](docs/api.md) documents the current implemented public API.

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

The following builds and animates a four-bar linkage through the generic API:

```python
import matplotlib.pyplot as plt
import numpy as np

from kimech import Mechanism, solve
from kimech.visualization import animate

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
values = np.linspace(0.8, 0.8 + 2 * np.pi, 180, endpoint=False)

solution = solve(
    mechanism,
    input=input_joint,
    values=values,
    initial_guess=initial_guess,
)

animation = animate(solution, fps=30)
plt.show()
```

Keep a reference to the returned animation until display or saving is complete, as shown above.

## Examples

[`examples/four_bar.py`](examples/four_bar.py) and [`examples/slider_crank.py`](examples/slider_crank.py) are complete examples built with the generic public API.
