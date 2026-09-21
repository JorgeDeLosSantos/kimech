# Kimech — Package structure and public API

> Status: current API for `0.4.0`.
>
> Kimech remains a young project, and this API may evolve in future versions. [`design.md`](design.md) records the `0.1.0` position-kinematics baseline, [`design-0.2.0.md`](design-0.2.0.md) the differential-kinematics baseline, and [`design-0.4.0.md`](design-0.4.0.md) the solver-robustness design baseline.

## 1. Overview

Kimech models planar rigid-body mechanisms declaratively, solves position configurations with one prescribed joint coordinate, and optionally solves analytic velocity and acceleration kinematics for the same accepted configurations.

A differential sweep looks like:

```python
import numpy as np

from kimech import Mechanism, solve

mechanism = Mechanism("four_bar")
# ... declare links, points, and joints ...

input_positions = np.linspace(0.8, 1.3, 60)
solution = solve(
    mechanism,
    input_joint=input_joint,
    input_position=input_positions,
    input_velocity=1.5,
    input_acceleration=0.0,
    initial_guess=initial_guess,
)

positions = solution.point_positions(point_p)
velocities = solution.point_velocities(point_p)
accelerations = solution.point_accelerations(point_p)
```

`input_position` parameterizes configurations; it is not interpreted as physical time. It is the natural coordinate of `input_joint`: relative angle for a revolute joint and signed displacement for a prismatic joint. Differential inputs are physical derivatives with respect to a common external time variable.

## 2. Package structure

```text
src/
└── kimech/
    ├── __init__.py
    ├── model.py
    ├── joints.py
    ├── solution.py
    ├── diagnostics.py
    ├── solver.py
    ├── validation.py
    ├── errors.py
    ├── _geometry.py
    ├── _constraints.py
    ├── _differential.py
    ├── _scaling.py
    └── visualization/
        ├── __init__.py
        ├── plot.py
        └── animation.py
```

The package remains intentionally flat. Mechanism-specific solver classes, backend registries, and plugin systems are not part of `0.4.0`.

### Module responsibilities

- `model.py` owns `Mechanism`, `Link`, `Ground`, and `Point`, including topology and local point geometry. Models do not store solved state or perform numerical solves.
- `joints.py` owns immutable `RevoluteJoint` and `PrismaticJoint` domain objects.
- `_geometry.py` contains private planar numerical helpers.
- `_constraints.py` assembles private residual, Jacobian, and analytic second-order constraint contributions.
- `_differential.py` solves private velocity and acceleration linear systems and independently verifies their residuals.
- `_scaling.py` constructs private dimensionless numerical scaling used consistently by position, velocity, and acceleration solves.
- `solver.py` validates and normalizes the solve problem, performs complete position continuation first, then optional velocity and acceleration phases, and returns result objects.
- `solution.py` owns solver-independent kinematic state and derived entity queries through `Configuration` and `KinematicSolution`.
- `diagnostics.py` owns structured numerical histories exposed through `SolveDiagnostics`.
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
    SolveDiagnostics,
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
    input_joint,
    input_position,
    initial_guess,
    input_velocity=None,
    input_acceleration=None,
) -> KinematicSolution
```

`input_joint` must be a revolute or prismatic joint belonging to the mechanism. `input_position` is that joint's prescribed natural coordinate: relative angle for a revolute joint and signed displacement for a prismatic joint.

`input_position` may be either a finite scalar or a finite, non-empty one-dimensional sequence. `solve()` always returns a `KinematicSolution`:

- scalar `input_position` produces a one-sample solution;
- one-dimensional `input_position` produces an ordered multi-sample solution.

For a scalar solve, access the configuration with `solution[0]`.

The requested kinematic level is determined by optional differential inputs:

```text
input_position only
    -> position

input_position + input_velocity
    -> position + velocity

input_position + input_velocity + input_acceleration
    -> position + velocity + acceleration
```

`input_acceleration` without `input_velocity` is invalid. `None` means a differential level was not requested; `0.0` is a valid physical derivative and requests that level.

When `input_position` is scalar, each requested differential input must also be scalar. For a position sweep, `input_velocity` and `input_acceleration` may each be either:

- a scalar, explicitly broadcast to every sample; or
- a one-dimensional array with exactly the same length as `input_position`.

General NumPy broadcasting is not part of the public contract. All prescribed input data are validated before the position solve begins.

### Position semantics

For sweeps, user order is preserved and each accepted position configuration warm-starts the next position solve. Differential phases run only after the complete position history has been accepted, so requesting velocity or acceleration does not alter branch continuation.

`initial_guess` is required because it selects the numerical starting state and, in mechanisms with multiple assembly branches, helps select the intended branch. It may be either:

- a mapping containing exactly every mobile link with an `(x, y, theta)` pose; or
- a compatible `Configuration` from the same mechanism snapshot.

Kimech internally solves the position problem in dimensionless scaled coordinates and validates accepted configurations with a dimensionless residual tolerance. Public positions and derived quantities remain in the user's original units.

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

`input_position` samples are configuration parameters, not a time array.

The dot notation has its standard physical meaning. `input_velocity`, `input_acceleration`, generalized differential state, and derived differential queries are derivatives with respect to one common external physical time variable.

A user may externally sample a time law and pass corresponding `u`, `u_dot`, and `u_ddot` arrays. Kimech itself does not store the sample times.

Animation `fps` remains presentation-only and is not a physical integration step.

## 7. `Configuration`

A `Configuration` always stores generalized position state and may store velocity and acceleration state.

Important properties:

```python
config.mechanism
config.input_joint
config.input_position
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
config.point_position(point)      # (2,)
config.point_velocity(point)      # (2,)
config.point_acceleration(point)  # (2,)
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
solution.input_positions
solution.input_velocities
solution.input_accelerations

solution.coordinates
solution.coordinate_velocities
solution.coordinate_accelerations

solution.has_velocity
solution.has_acceleration
```

Shapes are:

- `input_positions`: `(N,)`;
- optional differential input histories: `(N,)`;
- `coordinates`: `(N, 3*n)`;
- optional generalized differential histories: `(N, 3*n)`.

### Indexing

```python
config = solution[i]
```

Integer indexing returns a `Configuration` preserving all position, velocity, acceleration and prescribed-input metadata available at sample `i`. Slicing returns a new `KinematicSolution` over the selected samples, and iteration yields `Configuration` objects in solution order.

### History queries

```python
solution.point_positions(point)         # (N, 2)
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

`KinematicSolution` behaves as an immutable sequence by protocol:

```python
config = solution[i]       # Configuration
subset = solution[a:b]     # KinematicSolution
reverse = solution[::-1]   # KinematicSolution

for config in solution:
    ...
```

Fancy indexing and mutation operations such as item assignment, `append`, or `extend` are not part of the public API. Empty `KinematicSolution` objects are valid containers, although `solve()` does not produce them.

## 9. Intentional breaking changes

Because Kimech remains pre-`1.0`, API cleanups are applied without compatibility aliases.

For `0.3.0`:

```text
solve(..., input=..., values=...)
    -> solve(..., input_joint=..., input_position=...)

solve(... scalar position ...) -> Configuration
    -> solve(... scalar position ...) -> KinematicSolution(len=1)

Configuration.input_value
    -> Configuration.input_position

KinematicSolution.input_values
    -> KinematicSolution.input_positions
```

The earlier `0.2.0` result-method renames remain:

```text
Configuration.pose()
    -> Configuration.body_pose()

KinematicSolution.link_poses()
    -> KinematicSolution.body_poses()
```

See [`CHANGELOG.md`](../CHANGELOG.md).

## 10. Solve diagnostics

Solutions returned by `solve()` include structured numerical diagnostics:

```python
diagnostics = solution.diagnostics

condition = diagnostics.condition_numbers
sigma_min = diagnostics.min_singular_values
rank = diagnostics.ranks
subdivisions = diagnostics.subdivision_counts
strategies = diagnostics.strategies
attempts = diagnostics.corrector_attempts
residuals = diagnostics.residual_norms
```

These histories have shape `(N,)`, one value for each accepted configuration.

The metrics are evaluated from Kimech's **dimensionless scaled solve Jacobian**,

\[
\hat J = D_\Phi^{-1} J D_q,
\]

rather than from the raw dimensional Jacobian. Consequently, the diagnostics are intended to remain comparable when a mechanism is expressed in different but consistent linear units.

The reported matrix is the Jacobian of the complete driven position problem: all joint-constraint rows plus the prescribed-input row. Therefore these quantities diagnose the conditioning and rank of the **chosen solve formulation**. They may depend on which joint is selected as `input_joint` and should not be interpreted as a driver-independent classification of the mechanism.

The initial diagnostics are deliberately descriptive rather than prescriptive:

- `condition_numbers` reports the 2-norm condition number of the scaled Jacobian;
- `min_singular_values` reports its smallest singular value;
- `ranks` reports numerical rank using the standard floating-point SVD tolerance;
- `subdivision_counts` reports how many internal accepted continuation points were required to recover each requested sample. A value of zero means no adaptive subdivision was needed;
- `strategies` reports the path that produced each requested position sample: `"initial_guess"`, `"predictor"`, `"warm_start"`, or `"subdivision"`;
- `corrector_attempts` reports how many nonlinear position-corrector calls were made while obtaining that requested sample, including failed recovery attempts and internal subdivision solves;
- `residual_norms` reports the final infinity norm of the dimensionless scaled position residual for each accepted requested sample.

An exactly rank-deficient Jacobian can therefore appear as `condition_number = inf`, `min_singular_value = 0`, and a reduced rank. Kimech does not yet apply a universal threshold for declaring a configuration "near singular".

Diagnostics are preserved when slicing a `KinematicSolution`. Manually constructed `KinematicSolution` objects may omit diagnostics, in which case `solution.diagnostics is None`.

The strategy names describe continuation behavior rather than a numerical backend. `"predictor"` means a first-order tangent prediction was accepted by the nonlinear corrector. `"warm_start"` means the previous accepted configuration was used directly; this includes cases where tangent prediction was unavailable and therefore collapsed to the previous state. `"subdivision"` means one or more hidden intermediate input positions were needed before the requested sample could be reached.

Kimech deliberately does not expose SciPy-specific counters such as `nfev` or `njev` as part of `SolveDiagnostics`. Process diagnostics are intended to remain meaningful if the nonlinear backend changes.

## 11. Adaptive subdivision

Position sweeps use predictor-corrector continuation. If a requested sample cannot be solved from the predictor, Kimech retries from the previous accepted configuration. If both attempts fail and there is a previous accepted sample, Kimech may recursively bisect the input interval and solve internal intermediate configurations.

The subdivision is an internal recovery mechanism:

- user-supplied `input_positions` are never expanded in the returned solution;
- internal intermediate configurations are discarded after they help reach the requested target;
- subdivision depth is bounded internally;
- if recovery fails, the original requested-target `KinematicSolveError` is preserved.

Adaptive subdivision does not make unreachable targets solvable and does not cross folds where the selected `input_joint` ceases to be a valid local parameter. See `docs/study-0.4.0-singularity-diagnostics.md` for examples involving slider-crank dead-centers and four-bar rocker toggles.

## 12. Errors

Public exception hierarchy:

```text
KimechError
├── InvalidModelError
└── KinematicSolveError
```

`InvalidModelError` reports structurally invalid models or solve problems.

`KinematicSolveError` reports numerical failures at position, velocity, or acceleration level. Differential failures identify the stage and, for sweeps, the input index/value when available. Kimech independently verifies accepted residuals rather than trusting the underlying numerical routine alone.

No public singularity exception or condition-number policy exists in `0.4.0`.

## 13. Validation

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

## 14. Visualization

Visualization requires `[viz]` and remains Matplotlib-only.

### Static plotting

```python
fig, ax = plot(config)
fig.savefig("mechanism.svg")
```

Kimech renders a schematic rigid-body scaffold rather than physical/CAD geometry. Structural joint points define the primary scaffold. Mobile links with fewer than two structural points fall back to their declared body points so plate-like bodies remain visually coherent. Auxiliary points on an already-defined scaffold remain markers and receive lightweight visual connectors to that scaffold. Ground does not use this fallback.

Joint glyph sizes are scaled from effective body scaffolds rather than arbitrary remote auxiliary points. The plot bounds still include all rendered geometry.

### Animation

```python
animation = animate(solution, fps=30)
```

Optional progressive point traces can be requested explicitly:

```python
animation = animate(
    solution,
    fps=30,
    trace_points=[foot],
)
```

`trace_points` must be a collection of mechanism `Point` objects. Duplicate point objects are collapsed. A trace grows from the first configuration through the current frame and follows the stored solution order; it does not imply a physical time parameter.

One configuration corresponds to one frame. `fps` controls presentation playback speed only and is not a physical time step. Kimech does not call `plt.show()` automatically.

GIF output can be delegated to Matplotlib/Pillow:

```python
animation.save("mechanism.gif", writer="pillow")
```

## 15. Result snapshot semantics

Public `Configuration` and `KinematicSolution` constructors validate the structure, shape, finiteness, and entity compatibility of supplied state. They do not certify that manually supplied coordinates satisfy the mechanism constraints. Results returned by `solve()` contain states accepted by the solver.

For `Configuration`, any prescribed-input metadata (`input_position`, `input_velocity`, or `input_acceleration`) requires `input_joint`. Input acceleration additionally requires input velocity.

Result objects retain the link layout captured when they are constructed or solved. This keeps the mapping between links and generalized state stable even if the mechanism object is later extended. Queries require entities compatible with that retained snapshot.

## 16. Examples and tests

The examples deliberately separate geometric motion/animation from quantitative differential analysis:

```text
examples/
├── four_bar.py
├── four_bar_analysis.py
├── slider_crank.py
├── slider_crank_analysis.py
└── slider_crank_analysis_comparison.py
```

- `four_bar.py` and `slider_crank.py` focus on position solving and animation.
- `four_bar_analysis.py` plots rocker angle, angular velocity, angular acceleration, and coupler-point differential magnitudes versus prescribed crank angle.
- `slider_crank_analysis.py` plots slider displacement, velocity, and acceleration versus crank angle and demonstrates equivalent prismatic-input reconstruction.
- `slider_crank_analysis_comparison.py` compares position, velocity, and acceleration with an independent closed-form slider-crank solution.

The test suite covers model/constraint behavior, position solving, differential result state, velocity and acceleration solves, analytic second-order terms, four-bar and slider-crank acceptance, prismatically driven inverse analysis, visualization, and package metadata.

## 17. Deliberately absent API

`0.4.0` does not provide public abstractions for multiple inputs, motion laws, time histories, dynamics, forces, masses/inertias, pseudo-arclength continuation, branch enumeration, renderer/backend registries, or mechanism-specific solver classes. The release adds descriptive scaled-Jacobian and solve-process diagnostics plus bounded internal adaptive subdivision without defining a universal near-singularity policy.
