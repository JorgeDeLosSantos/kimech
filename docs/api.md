# Kimech — Package structure and public API

> Status: public API for `0.2.0`.
>
> Kimech remains a young pre-`1.0` project. [`design.md`](design.md) records the `0.1.0` position-kinematics baseline and [`design-0.2.0.md`](design-0.2.0.md) records the differential-kinematics design baseline. This document describes the implemented public API.

## 1. Overview

Kimech models planar rigid-body mechanisms declaratively and solves one-DOF kinematics with one prescribed revolute or prismatic joint coordinate.

`0.2.0` supports:

- position configurations;
- generalized and entity-level velocities;
- generalized and entity-level accelerations;
- scalar solves and ordered sweeps;
- warm-start continuation for position sweeps;
- Matplotlib plotting and animation.

A differential sweep looks like:

```python
solution = solve(
    mechanism,
    input=input_joint,
    values=values,
    input_velocity=omega,
    input_acceleration=alpha,
    initial_guess=initial_guess,
)

positions = solution.point_path(point)
velocities = solution.point_velocities(point)
accelerations = solution.point_accelerations(point)
```

`values` parameterizes configurations and is not a time array. Differential quantities are physical derivatives with respect to a common external time variable supplied implicitly through `input_velocity` and `input_acceleration`.

## 2. Package structure

```text
src/kimech/
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

Private modules and names beginning with `_` are implementation details.

### Module responsibilities

- `model.py`: `Mechanism`, `Link`, `Ground`, `Point` and topology/local geometry.
- `joints.py`: immutable `RevoluteJoint` and `PrismaticJoint` metadata.
- `_geometry.py`: private planar numerical helpers.
- `_constraints.py`: residuals, analytic Jacobian and analytic second-order constraint contributions.
- `_differential.py`: private linear velocity and acceleration solves.
- `solver.py`: public problem validation, position continuation, differential phases and result construction.
- `solution.py`: solver-independent `Configuration` and `KinematicSolution` state/query API.
- `validation.py`: structural validation and mobility estimation.
- `errors.py`: public Kimech exception hierarchy.
- `visualization`: Matplotlib schematic plotting and animation.

## 3. Public imports and dependencies

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

Core dependencies are NumPy and SciPy. Visualization is optional:

```bash
pip install -e ".[viz]"
```

## 4. Model API

### `Mechanism`

```python
mechanism = Mechanism(name="four_bar")
```

The implemented surface includes:

```python
mechanism.name
mechanism.ground
mechanism.links
mechanism.joints
mechanism.add_link(name)
mechanism.revolute(point_a, point_b, name=None)
mechanism.prismatic(point_a, point_b, axis_a=..., axis_b=..., name=None)
mechanism.mobility()
mechanism.validate()
mechanism[name]
```

`links` and `joints` are deterministic creation-order tuples.

### `Link`, `Ground`, and `Point`

Links and ground own named points:

```python
crank = mechanism.add_link("crank")
point_a = crank.add_point("A", (0.0, 0.0))
same_point = crank["A"]
```

`Point.local` returns a safe `(2,)` NumPy copy.

### Joints

`RevoluteJoint` stores `point_a`, `point_b`, and optional `name`.

`PrismaticJoint` additionally stores normalized local axes `axis_a` and `axis_b`; safe NumPy copies are available through `axis_a_array` and `axis_b_array`.

Joint objects are immutable metadata and do not store mutable kinematic state.

## 5. Solving kinematics

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

`input` must be a revolute or prismatic joint belonging to the mechanism.

### Return type

The shape of `values` determines the return type:

- scalar `values` -> `Configuration`;
- one-dimensional non-empty sequence -> `KinematicSolution`.

A length-one sequence still returns `KinematicSolution`.

### Differential solve levels

```text
values
    -> position

values + input_velocity
    -> position + velocity

values + input_velocity + input_acceleration
    -> position + velocity + acceleration
```

`input_acceleration` requires `input_velocity`.

`None` means that differential level is not requested. Numeric zero is a valid prescribed derivative and requests the corresponding solve level.

### Differential input shapes

If `values` is scalar, supplied differential inputs must be scalar.

If `values` has shape `(N,)`, each differential input may be:

- a scalar, explicitly broadcast to all `N` configurations; or
- an array with exact shape `(N,)`.

General NumPy broadcasting is not part of the API contract. All supplied values must be finite.

### Time semantics

`values` is a configuration parameter sequence, not a time grid. `input_velocity` and `input_acceleration` are physical derivatives with respect to one common external time variable. Kimech does not receive or store that time array in `0.2.0`.

### Initial guess and continuation

`initial_guess` may be:

- a mapping containing exactly every mobile link with `(x, y, theta)` poses; or
- a compatible `Configuration` from the same mechanism.

For sweeps, positions are solved in user order and each accepted position becomes the next position initial guess. Velocity and acceleration are solved afterward and do not affect branch continuation.

The current public solver supports a single prescribed input and requires the resulting system to be square, corresponding to the current structural mobility-one R/P model.

## 6. Differential formulation

Position satisfies

\[
\Phi(q,u)=0.
\]

Velocity is solved analytically from

\[
J(q)\dot q=b_v.
\]

Acceleration is solved from

\[
J(q)\ddot q=b_a(q,\dot q,\ddot u).
\]

The same analytic position Jacobian is reused at all three levels. Kimech does not estimate production velocities or accelerations by finite-differencing neighboring solved configurations.

## 7. `Configuration`

A `Configuration` always contains position state and may additionally contain generalized velocity and acceleration state.

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

`coordinate_velocities` raises `ValueError` when velocity state is unavailable. `coordinate_accelerations` behaves analogously.

### Body queries

```python
config.body_pose(body)
config.body_velocity(body)
config.body_acceleration(body)
```

For a mobile link these return respectively

```text
(x, y, theta)
(vx, vy, omega)
(ax, ay, alpha)
```

with shape `(3,)`.

Ground pose is the zero-coordinate identity pose. Ground velocity/acceleration return zeros when the corresponding differential capability exists.

### Point queries

```python
config.position(point)
config.velocity(point)
config.acceleration(point)
```

Each returns a global `(2,)` array.

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

Complete examples:

```text
examples/
├── four_bar.py
└── slider_crank.py
```

The test suite covers model/constraint behavior, position solving, differential result state, velocity and acceleration solves, analytic second-order terms, four-bar and slider-crank acceptance, prismatically driven inverse analysis, visualization, and package metadata.

## 15. Deliberately absent API

`0.2.0` does not provide public abstractions for multiple inputs, motion laws, time histories, dynamics, forces, masses/inertias, formal singularity diagnostics, adaptive continuation, branch enumeration, renderer/backend registries, or mechanism-specific solver classes.
