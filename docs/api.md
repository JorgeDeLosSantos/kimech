# Kimech — Package structure and public API

> Status: current implemented API for `0.1.0.dev0`.
>
> Kimech remains in early development and this API may evolve before `0.1.0`. This document describes the implementation as it exists now; [`design.md`](design.md) records the conceptual and architectural baseline.

## 1. Overview

Kimech models planar rigid-body mechanisms declaratively, solves position configurations with one prescribed joint coordinate, exposes entity-based result queries, and provides optional Matplotlib visualization.

A typical sweep looks like:

```python
import numpy as np

from kimech import Mechanism, solve
from kimech.visualization import animate

mechanism = Mechanism("four_bar")
# ... declare links, points, and joints ...

values = np.linspace(
    0.8,
    0.8 + 2 * np.pi,
    180,
    endpoint=False,
)
solution = solve(
    mechanism,
    input=input_joint,
    values=values,
    initial_guess=initial_guess,
)
path = solution.point_path(point_p)

animation = animate(solution, fps=30)
```

The values parameterize configurations; they are not interpreted as physical time.

## 2. Package structure

```text
src/
└── kimech/
    ├── __init__.py
    ├── model.py
    ├── joints.py
    ├── solution.py
    ├── solver.py
    ├── validation.py
    ├── errors.py
    ├── _geometry.py
    ├── _constraints.py
    └── visualization/
        ├── __init__.py
        ├── plot.py
        └── animation.py
```

The package is intentionally flat. Mechanism-specific classes, solver class hierarchies, backend registries, and plugin systems are not part of `0.1.0.dev0`.

### Module responsibilities

- `model.py` owns `Mechanism`, `Link`, `Ground`, and `Point`, including topology and local point geometry. Models do not store current poses or perform numerical solves.
- `joints.py` owns immutable `RevoluteJoint` and `PrismaticJoint` domain objects.
- `_geometry.py` contains private planar numerical helpers.
- `_constraints.py` assembles private residual and analytic Jacobian contributions. It receives explicit `links: tuple[Link, ...]` and joint snapshots; tuple order defines the layout of the generalized coordinate vector `q`.
- `solver.py` validates the problem, snapshots `mechanism.links` and `mechanism.joints`, packs poses in link snapshot order, assembles residual/Jacobian callbacks, calls `scipy.optimize.root(..., method="hybr")`, independently verifies the final residual, and warm-starts sweeps.
- `solution.py` owns solver-independent kinematic state and result queries through `Configuration` and `KinematicSolution`. It does not wrap SciPy result objects or expose diagnostic containers.
- `validation.py` provides lightweight structural validation and mobility estimation.
- `errors.py` defines the public Kimech exception hierarchy.
- `visualization` renders configurations and solutions with Matplotlib without mutating them.

Private modules and names beginning with `_` are implementation details, not public API.

## 3. Imports and dependencies

The public core surface is available from `kimech`:

```python
from kimech import (
    Configuration,
    Ground,
    InvalidModelError,
    KimechError,
    KinematicSolution,
    KinematicSolveError,
    Link,
    Mechanism,
    Point,
    PrismaticJoint,
    RevoluteJoint,
    ValidationReport,
    solve,
)
```

Visualization is imported separately:

```python
from kimech.visualization import animate, plot
```

The core dependencies are NumPy and SciPy. Visualization is an optional extra containing Matplotlib and Pillow:

```bash
pip install -e ".[viz]"
```

Top-level `import kimech` does not require Matplotlib.

## 4. Model API

### `Mechanism`

```python
mechanism = Mechanism(name="four_bar")
```

The implemented interface is:

```python
class Mechanism:
    @property
    def name(self) -> str | None: ...

    @property
    def ground(self) -> Ground: ...

    @property
    def links(self) -> tuple[Link, ...]: ...

    @property
    def joints(self) -> tuple[RevoluteJoint | PrismaticJoint, ...]: ...

    def add_link(self, name: str) -> Link: ...

    def revolute(
        self,
        point_a: Point,
        point_b: Point,
        *,
        name: str | None = None,
    ) -> RevoluteJoint: ...

    def prismatic(
        self,
        point_a: Point,
        point_b: Point,
        *,
        axis_a: Sequence[float],
        axis_b: Sequence[float],
        name: str | None = None,
    ) -> PrismaticJoint: ...

    def mobility(self) -> int: ...

    def validate(self) -> ValidationReport: ...

    def __getitem__(self, name: str) -> Link: ...
```

`links` and `joints` are tuples in deterministic creation order. Both prismatic axes are explicit and required.

### `Link`, `Ground`, and `Point`

Links and ground own points:

```python
crank = mechanism.add_link("crank")
point_a = crank.add_point("A", (0.0, 0.0))
same_point = crank["A"]
```

`Link` exposes `mechanism`, `name`, `points`, `add_point()`, and name-based `__getitem__()`. `Ground` exposes the same point-owning operations plus `mechanism`, `name`, and `points`; its name is `"ground"`.

A `Point` exposes `name`, `body`, and `local`. `local` returns a safe NumPy copy with shape `(2,)`.

### Joints

`RevoluteJoint` stores `point_a`, `point_b`, and optional `name`.

`PrismaticJoint` stores `point_a`, `point_b`, optional `name`, and the two local axes. `axis_a` and `axis_b` are normalized during construction and stored as immutable two-component values. Safe NumPy copies are available through:

```python
joint.axis_a_array
joint.axis_b_array
```

Joint objects are immutable metadata. They do not assemble residuals or expose coordinate-vector indices.

## 5. Solving positions

```python
solve(
    mechanism,
    *,
    input,
    values,
    initial_guess,
)
```

`input` must be a revolute or prismatic joint belonging to the mechanism. `values` determines the return type:

- a scalar returns a `Configuration`;
- a one-dimensional, non-empty sequence returns a `KinematicSolution`.

Input values are finite configuration parameters, not physical timestamps. For a sweep, user order is preserved and each accepted configuration becomes the initial guess for the next value. The first failure aborts the sweep.

`initial_guess` may be either:

- a mapping containing exactly every mobile link, with each value an `(x, y, theta)` pose; or
- a compatible `Configuration` from the same mechanism whose saved link layout contains all links in the solve snapshot.

Partial mapping guesses and raw packed coordinate vectors are not accepted. The initial guess selects the assembly branch but need not already satisfy the prescribed coordinate.

The solver currently supports a single prescribed input and requires the resulting position system to be square (structural mobility one for the current R/P model). No public SciPy methods, options, or tolerances are exposed.

### Solve errors

`InvalidModelError` reports structurally invalid models or solve problems. `KinematicSolveError` is a lightweight exception raised when a numerical configuration is not accepted. Its message includes useful available context: the input index for a sweep, input value, independently recomputed residual infinity norm, and SciPy solver message. It has no documented structured payload.

## 6. Result API

### `Configuration`

```python
class Configuration:
    @property
    def mechanism(self) -> Mechanism: ...

    @property
    def input_joint(self) -> RevoluteJoint | PrismaticJoint | None: ...

    @property
    def input_value(self) -> float | None: ...

    @property
    def coordinates(self) -> np.ndarray: ...

    def pose(self, body: Link | Ground) -> np.ndarray: ...

    def position(self, point: Point) -> np.ndarray: ...

    def joint_coordinate(
        self,
        joint: RevoluteJoint | PrismaticJoint,
    ) -> float: ...
```

`input_joint` and `input_value` identify the prescribed coordinate for solver-created configurations. Both may be `None` when a `Configuration` is constructed directly.

Shapes and behavior:

- `coordinates` has shape `(3*n,)` for mobile links and returns a copy;
- `pose(body)` returns a safe `(x, y, theta)` array;
- `pose(mechanism.ground)` returns `np.zeros(3)`;
- `position(point)` returns a derived global `(2,)` array;
- `joint_coordinate(joint)` returns the natural unwrapped revolute angle or signed prismatic displacement.

### `KinematicSolution`

```python
class KinematicSolution:
    @property
    def mechanism(self) -> Mechanism: ...

    @property
    def input_joint(self) -> RevoluteJoint | PrismaticJoint: ...

    @property
    def input_values(self) -> np.ndarray: ...

    @property
    def coordinates(self) -> np.ndarray: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Configuration: ...

    def point_path(self, point: Point) -> np.ndarray: ...

    def link_poses(self, body: Link | Ground) -> np.ndarray: ...

    def joint_coordinates(
        self,
        joint: RevoluteJoint | PrismaticJoint,
    ) -> np.ndarray: ...
```

`solution[index]` returns a `Configuration` carrying the same mechanism, input joint, and corresponding input value. Integer indexing is supported; slicing is not.

Expected shapes are:

- `input_values`: `(N,)`;
- `coordinates`: `(N, 3*n)`;
- `point_path(point)`: `(N, 2)`;
- `link_poses(body)`: `(N, 3)`;
- `joint_coordinates(joint)`: `(N,)`.

Array properties return copies, and query arrays are safe derived values. No pandas dependency or diagnostic wrapper object is used.

### Result snapshot semantics

Result objects retain the link layout captured when they are constructed or solved. This keeps each link's association with its three coordinates stable for the lifetime of the result, even if the mechanism is later extended. Queries still require entities compatible with that retained layout.

## 7. Validation

```python
report = mechanism.validate()
```

`ValidationReport` is a lightweight immutable value with:

```python
report.mobility
report.errors
report.warnings
report.is_valid
```

`mobility` is the planar lower-pair structural estimate. `errors` and `warnings` are tuples of messages; `is_valid` is true when there are no errors. Validation does not promise complete detection of redundant constraints, singularities, or special geometric degeneracies.

## 8. Visualization

Visualization requires the optional `[viz]` dependencies and is intentionally Matplotlib-only.

### Static plotting

```python
plot(
    config,
    *,
    ax=None,
)
```

`plot()` accepts only a `Configuration` and returns `(fig, ax)`. If `ax` is supplied, Kimech draws into it and returns its figure and the same axes. Auxiliary model points are included in the schematic; trajectory overlays are not part of this API.

Static SVG output uses the returned Matplotlib figure:

```python
fig, ax = plot(config)
fig.savefig("mechanism.svg")
```

### Animation

```python
animate(
    solution,
    *,
    fps=30,
    ax=None,
)
```

`animate()` accepts only a `KinematicSolution` and returns `matplotlib.animation.FuncAnimation`. One configuration corresponds to one frame. Frames are uniformly spaced for presentation, `fps` controls playback speed, and `input_values` do not represent physical time. Playback repeats.

The viewport remains fixed over the full rendered motion. Schematic artists are created once and updated rather than recreated on each frame; the prismatic guide length remains constant over a solution.

Kimech constructs the schematic. Matplotlib remains responsible for display and file output. Kimech does not call `plt.show()` automatically:

```python
import matplotlib.pyplot as plt

animation = animate(solution)
plt.show()
```

Keep a reference to the returned animation until display or saving is complete. GIF output uses Matplotlib's animation object and Pillow from `[viz]`:

```python
animation = animate(solution, fps=30)
animation.save("mechanism.gif", writer="pillow")
```

Kimech has no dedicated SVG or GIF exporter and no renderer/backend abstraction.

## 9. Tests and examples

The current test layout mirrors public and internal responsibilities:

```text
tests/
├── test_animation.py
├── test_constraints.py
├── test_four_bar.py
├── test_joints.py
├── test_model.py
├── test_slider_crank.py
├── test_solution.py
├── test_solver.py
├── test_validation.py
└── test_visualization.py
```

The four-bar and slider-crank tests exercise the generic infrastructure; there are no mechanism-specific implementations. Complete generic API examples live in:

```text
examples/
├── four_bar.py
└── slider_crank.py
```

## 10. Deliberately absent API

There is no public `Driver`, `JointCoordinate`, `PositionSolver`, `SolverOptions`, `Constraint`, `Backend`, `Renderer`, `Analysis`, `FourBar`, or `SliderCrank` class. Mechanisms and links do not carry mutable current-pose state. These omissions keep the current API aligned with the position-kinematics scope.

## 11. Resolved implementation decisions

- Raw array properties return copies, and result-query arrays are safe derived values.
- `plot()` returns `(fig, ax)`.
- `animate()` returns Matplotlib `FuncAnimation`.
- Matplotlib and Pillow are supplied through the optional `[viz]` extra.
- No public characteristic-length override or internal characteristic-length scaling is present.
- Scalar and one-dimensional sequence inputs retain their distinct `Configuration` / `KinematicSolution` return behavior.
- Kinematic results do not contain diagnostic wrapper objects.
- Examples use the generic model and solver API rather than special mechanism classes.