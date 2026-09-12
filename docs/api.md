# Kimech — Package structure and public API

> Status: public API for `0.2.0`.
>
> Kimech remains a young project, and this API may evolve in future versions. [`design.md`](design.md) records the `0.1.0` position-kinematics baseline and [`design-0.2.0.md`](design-0.2.0.md) records the differential-kinematics design baseline.

## 1. Overview

Kimech models planar rigid-body mechanisms declaratively, solves position configurations with one prescribed joint coordinate, and optionally solves analytic velocity and acceleration kinematics for the same accepted configurations.

A differential sweep looks like:

```python
import numpy as np

from kimech import Mechanism, solve

mechanism = Mechanism("four_bar")
# ... declare links, points, and joints ...

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

`values` parameterize configurations; they are not interpreted as physical time. Differential inputs are physical derivatives with respect to a common external time variable.

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
    ├── _differential.py
    └── visualization/
        ├── __init__.py
        ├── plot.py
        └── animation.py
```

The package remains intentionally flat. Mechanism-specific solver classes, backend registries, and plugin systems are not part of `0.2.0`.

### Module responsibilities

- `model.py` owns `Mechanism`, `Link`, `Ground`, and `Point`, including topology and local point geometry. Models do not store solved state or perform numerical solves.
- `joints.py` owns immutable `RevoluteJoint` and `PrismaticJoint` domain objects.
- `_geometry.py` contains private planar numerical helpers.
- `_constraints.py` assembles private residual, Jacobian, and analytic second-order constraint contributions.
- `_differential.py` solves private velocity and acceleration linear systems and independently verifies their residuals.
- `solver.py` validates and normalizes the solve problem, performs complete position continuation first, then optional velocity and acceleration phases, and returns result objects.
- `solution.py` owns solver-independent kinematic state and derived entity queries through `Configuration` and `KinematicSolution`.
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

The implemented interface includes:

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
    def revolute(...) -> RevoluteJoint: ...
    def prismatic(...) -> PrismaticJoint: ...
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

A `Point` exposes `name`, `body`, and `local`. `local` returns a safe NumPy copy with shape `(2,)`.

### Joints

`RevoluteJoint` stores `point_a`, `point_b`, and optional `name`.

`PrismaticJoint` additionally stores normalized local `axis_a` and `axis_b`. `axis_a` defines the sign of the natural prismatic coordinate and its derivatives.

## 5. Solving

```python
solve(
    mechanism,
    *,
    input,
    values,
    initial_guess,
    input_velocity=None,
    input_acceleration=None,
)
```

`input` must be a revolute or prismatic joint belonging to the mechanism. The shape of `values` determines the return type:

- scalar `values` returns `Configuration`;
- one-dimensional non-empty `values` returns `KinematicSolution`.

The requested kinematic level is determined by optional differential inputs:

```text
values only
    -> position

values + input_velocity
    -> position + velocity

values + input_velocity + input_acceleration
    -> position + velocity + acceleration
```

`input_acceleration` without `input_velocity` is invalid. `None` means a differential level was not requested; `0.0` is a valid physical derivative and requests that level.

For a sweep, `input_velocity` and `input_acceleration` may each be either:

- a scalar, explicitly broadcast to every sample; or
- a one-dimensional array with exactly the same length as `values`.

General NumPy broadcasting is not part of the public contract. All input data are validated before the position solve begins.

### Position semantics

For sweeps, user order is preserved and each accepted position configuration warm-starts the next position solve. Differential phases run only after the complete position history has been accepted, so requesting velocity or acceleration does not alter branch continuation.

`initial_guess` may be either:

- a mapping containing exactly every mobile link with an `(x, y, theta)` pose; or
- a compatible `Configuration` from the same mechanism.

The current model supports one prescribed input and square mobility-one R/P solve systems.

### Differential formulation

Position satisfies

\[
\Phi(q,u)=0.
\]

Velocity solves

\[
J(q)\dot q=b_v,
\]

where the geometric rows of `b_v` are zero and the driver row contains the prescribed input velocity.

Acceleration solves

\[
J(q)\ddot q=b_a(q,\dot q,\ddot u).
\]

Kimech reuses the analytic position Jacobian and computes analytic second-order bias terms. It does not obtain production derivatives by finite-differencing neighboring configurations, and it does not construct a public `J_dot` or Hessian API.

## 6. Time semantics

`values` are configuration parameters, not a time array.

The dot notation has its standard physical meaning. `input_velocity`, `input_acceleration`, generalized differential state, and derived differential queries are derivatives with respect to one common external physical time variable.

A user may externally sample a time law and pass corresponding `u`, `u_dot`, and `u_ddot` arrays. Kimech itself does not store the sample times.

Animation `fps` remains presentation-only and is not a physical integration step.

## 7. `Configuration`

A `Configuration` always stores generalized position state and may store velocity and acceleration state.

Important properties:

```python
config.mechanism
config.input_joint
config.input_value
config.input_velocity
config.input_acceleration

config.coordinates
config.coordinate_velocities
config.coordinate_accelerations

config.has_velocity
config.has_acceleration
```

Generalized arrays have shape `(3*n,)`, ordered as `(x, y, theta)` for each mobile body in the retained link snapshot.

Acceleration state implies velocity state. Querying unavailable differential state raises `ValueError` rather than returning `None`.

### Body queries

```python
config.body_pose(body)          # (x, y, theta)
config.body_velocity(body)      # (vx, vy, omega)
config.body_acceleration(body)  # (ax, ay, alpha)
```

Ground pose is the zero-coordinate identity. Ground differential state is zero only when the corresponding result capability exists.

### Point queries

```python
config.position(point)      # (2,)
config.velocity(point)      # (2,)
config.acceleration(point)  # (2,)
```

Point velocity and acceleration are derived from rigid-body state, including tangential and centripetal contributions.

### Joint queries

```python
config.joint_coordinate(joint)
config.joint_velocity(joint)
config.joint_acceleration(joint)
```

These operate on the same natural joint coordinate family:

- revolute: relative unwrapped angle and its derivatives;
- prismatic: signed displacement along `axis_a` and its derivatives.

## 8. `KinematicSolution`

A `KinematicSolution` stores an ordered history of accepted state.

Important properties:

```python
solution.mechanism
solution.input_joint
solution.input_values
solution.input_velocities
solution.input_accelerations

solution.coordinates
solution.coordinate_velocities
solution.coordinate_accelerations

solution.has_velocity
solution.has_acceleration
```

Shapes are:

- `input_values`: `(N,)`;
- optional differential input histories: `(N,)`;
- `coordinates`: `(N, 3*n)`;
- optional generalized differential histories: `(N, 3*n)`.

### Indexing

```python
config = solution[i]
```

Integer indexing returns a `Configuration` preserving all position, velocity, acceleration and prescribed-input metadata available at sample `i`. Slicing is not supported.

### History queries

```python
solution.point_path(point)             # (N, 2)
solution.point_velocities(point)        # (N, 2)
solution.point_accelerations(point)     # (N, 2)

solution.body_poses(body)               # (N, 3)
solution.body_velocities(body)          # (N, 3)
solution.body_accelerations(body)       # (N, 3)

solution.joint_coordinates(joint)       # (N,)
solution.joint_velocities(joint)        # (N,)
solution.joint_accelerations(joint)     # (N,)
```

Array properties and query results are safe values/copies and do not expose mutable internal state.

## 9. Intentional breaking rename in `0.2.0`

Because Kimech remains pre-`1.0`, the result API was standardized without compatibility aliases:

```text
Configuration.pose()
    -> Configuration.body_pose()

KinematicSolution.link_poses()
    -> KinematicSolution.body_poses()
```

See [`CHANGELOG.md`](../CHANGELOG.md).

## 10. Errors

Public exception hierarchy:

```text
KimechError
├── InvalidModelError
└── KinematicSolveError
```

`InvalidModelError` reports structurally invalid models or solve problems.

`KinematicSolveError` reports numerical failures at position, velocity, or acceleration level. Differential failures identify the stage and, for sweeps, the input index/value when available. Kimech independently verifies accepted residuals rather than trusting the underlying numerical routine alone.

No public singularity exception or condition-number policy exists in `0.2.0`.

## 11. Validation

```python
report = mechanism.validate()
```

`ValidationReport` provides:

```python
report.mobility
report.errors
report.warnings
report.is_valid
```

The mobility value is the planar lower-pair structural estimate. Validation does not promise complete detection of redundant constraints, special geometric degeneracies, toggles, or singularities.

## 12. Visualization

Visualization requires `[viz]` and remains Matplotlib-only.

### Static plotting

```python
fig, ax = plot(config)
fig.savefig("mechanism.svg")
```

### Animation

```python
animation = animate(solution, fps=30)
```

One configuration corresponds to one frame. `fps` controls presentation playback speed only and is not a physical time step. Kimech does not call `plt.show()` automatically.

GIF output can be delegated to Matplotlib/Pillow:

```python
animation.save("mechanism.gif", writer="pillow")
```

## 13. Result snapshot semantics

Result objects retain the link layout captured when they are constructed or solved. This keeps the mapping between links and generalized state stable even if the mechanism object is later extended. Queries require entities compatible with that retained snapshot.

## 14. Examples and tests

The examples deliberately separate geometric motion/animation from quantitative differential analysis:

```text
examples/
├── four_bar.py
├── four_bar_analysis.py
├── slider_crank.py
└── slider_crank_analysis.py
```

- `four_bar.py` and `slider_crank.py` focus on position solving and animation.
- `four_bar_analysis.py` plots rocker angle, angular velocity, angular acceleration, and coupler-point differential magnitudes versus prescribed crank angle.
- `slider_crank_analysis.py` plots slider displacement, velocity, and acceleration versus crank angle and demonstrates equivalent prismatic-input reconstruction.

The test suite covers model/constraint behavior, position solving, differential result state, velocity and acceleration solves, analytic second-order terms, four-bar and slider-crank acceptance, prismatically driven inverse analysis, visualization, and package metadata.

## 15. Deliberately absent API

`0.2.0` does not provide public abstractions for multiple inputs, motion laws, time histories, dynamics, forces, masses/inertias, formal singularity diagnostics, adaptive continuation, branch enumeration, renderer/backend registries, or mechanism-specific solver classes.
