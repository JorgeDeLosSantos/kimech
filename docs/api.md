# Kimech — Package structure and public API

> Status: development API for `0.6.0`.
>
> Kimech remains a young project, and this API may evolve in future versions. [`design-0.5.0.md`](design-0.5.0.md) records the current consolidation baseline; earlier design documents remain available for historical context.

## 1. Overview

Kimech models planar rigid-body mechanisms declaratively, solves position configurations with one prescribed joint coordinate, and optionally solves analytic velocity and acceleration kinematics for the same accepted configurations.

A differential sweep looks like:

```python
import numpy as np

from kimech import KinematicDriver, Mechanism, solve

mechanism = Mechanism("four_bar")
# ... declare links, points, and joints ...

input_positions = np.linspace(0.8, 1.3, 60)
driver = KinematicDriver(
    input_joint,
    position=input_positions,
    velocity=1.5,
    acceleration=0.0,
)

solution = solve(
    mechanism,
    driver=driver,
    initial_guess=initial_guess,
)

positions = solution.point_positions(point_p)
velocities = solution.point_velocities(point_p)
accelerations = solution.point_accelerations(point_p)
```

`KinematicDriver.position` parameterizes configurations; it is not interpreted as physical time. In 0.6.0 a driver targets the natural coordinate of one revolute or prismatic joint. Optional `velocity` and `acceleration` values are physical derivatives with respect to a common external time variable.

## 2. Package structure

```text
src/
└── kimech/
    ├── __init__.py
    ├── model.py
    ├── joints.py
    ├── driver.py
    ├── topology.py
    ├── sensitivity.py
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

The package remains intentionally flat. Mechanism-specific solver classes, backend registries, and plugin systems are not part of `0.5.0`.

### Module responsibilities

- `model.py` owns `Mechanism`, `Link`, `Ground`, and `Point`, including topology and local point geometry. Models do not store solved state or perform numerical solves.
- `joints.py` owns immutable `RevoluteJoint` and `PrismaticJoint` domain objects.
- `driver.py` owns `KinematicDriver`, the prescribed kinematic coordinate and its optional differential data.
- `topology.py` owns immutable-from-the-caller's-perspective structural topology snapshots through `MechanismTopology`.
- `sensitivity.py` provides explicit input-coordinate sensitivity analysis through `input_sensitivity()` and `InputSensitivity`.
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
    KinematicDriver,
    InputSensitivity,
    InvalidModelError,
    KimechError,
    KinematicSolution,
    KinematicSolveError,
    Link,
    Mechanism,
    MechanismTopology,
    Point,
    PrismaticJoint,
    RevoluteJoint,
    SolveDiagnosticSummary,
    SolveDiagnostics,
    SolveFailureContext,
    ValidationReport,
    input_sensitivity,
    solve,
)
```

Visualization is imported separately:

```python
from kimech.visualization import animate, plot, plot_topology
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
    def topology(self) -> MechanismTopology: ...
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


### Structural topology

```python
topology = mechanism.topology()
```

`MechanismTopology` is a structural snapshot of the mechanism modeled semantically as an undirected body-joint multigraph. Bodies are vertices and joints are edges; parallel joints between the same pair of bodies retain their identities.

The snapshot exposes:

```python
topology.mechanism
topology.bodies
topology.joints

topology.incident_joints(body)
topology.adjacent_bodies(body)
topology.joints_between(body_a, body_b)
topology.degree(body)

topology.connected_components
topology.is_connected
topology.cycle_rank
```

Ordering is deterministic. `bodies` contains ground first and then mobile links in creation order; joint-valued queries preserve joint creation order. `degree(body)` counts incident joints rather than unique neighboring bodies.

`cycle_rank` is the undirected multigraph quantity `E - V + C`. It is purely structural and is not mobility, constraint rank, or an assembly-mode count.

Topology objects use snapshot semantics: extending the source mechanism later does not change an existing `MechanismTopology`. Calling `mechanism.topology()` again produces a fresh structural snapshot.

## 5. Solving

A solve receives one explicit `KinematicDriver`:

```python
driver = KinematicDriver(
    input_joint,
    position=input_positions,
    velocity=1.5,
    acceleration=0.0,
)

solution = solve(
    mechanism,
    *,
    driver=driver,
    initial_guess=initial_guess,
)
```

In 0.6.0, `KinematicDriver` targets the natural coordinate of one revolute or prismatic joint. The target joint must belong to the mechanism snapshot.

`position` is required and may be either a finite scalar or a finite, non-empty one-dimensional sequence. Position determines the number of requested samples. `solve()` always returns a `KinematicSolution`:

- scalar `position` produces a one-sample solution;
- one-dimensional `position` produces an ordered multi-sample solution.

For a scalar solve, access the configuration with `solution[0]`.

The requested kinematic level is determined by optional driver differential data:

```text
position only
    -> position

position + velocity
    -> position + velocity

position + velocity + acceleration
    -> position + velocity + acceleration
```

`acceleration` without `velocity` is invalid. `None` means a differential level was not requested; `0.0` is a valid physical derivative and requests that level.

When driver position is scalar, requested differential data must also be scalar. For a position sweep, `velocity` and `acceleration` may each be either:

- a scalar, explicitly broadcast to every sample; or
- a one-dimensional array with exactly the same length as `position`.

General NumPy broadcasting is not part of the public contract. Prescribed driver data are validated by `KinematicDriver` before solving begins.

### Position semantics

For sweeps, user order is preserved and each accepted position configuration warm-starts the next position solve. Differential phases run only after the complete position history has been accepted, so requesting velocity or acceleration does not alter branch continuation.

`initial_guess` is required because it selects the numerical starting state and, in mechanisms with multiple assembly branches, helps select the intended branch. It may be either:

- a mapping containing exactly every mobile link with an `(x, y, theta)` pose; or
- a compatible `Configuration` from the same mechanism snapshot.

Kimech internally solves the position problem in dimensionless scaled coordinates and validates accepted configurations with a dimensionless residual tolerance. Public positions and derived quantities remain in the user's original units.

The current model supports one `KinematicDriver` and square mobility-one R/P solve systems.

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

`KinematicDriver.position` samples are configuration parameters, not a time array.

The dot notation has its standard physical meaning. Driver `velocity` and `acceleration`, generalized differential state, and derived differential queries are derivatives with respect to one common external physical time variable.

A user may externally sample a time law and pass corresponding `position`, `velocity`, and `acceleration` arrays to `KinematicDriver`. Explicit solve-time samples are not yet stored in this 0.6-C block.

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

## 9. Input-coordinate sensitivity

Input-coordinate sensitivity is an explicit downstream analysis of an accepted solution:

```python
solution = solve(...)
sensitivity = input_sensitivity(solution)
```

It evaluates the local derivative

\[
\frac{dq}{du}
\]

at every requested configuration, where `u` is the natural coordinate of the selected `input_joint`. Sensitivity is not a time derivative and does not require `input_velocity` or `input_acceleration`.

`InputSensitivity` exposes:

```python
sensitivity.mechanism
sensitivity.input_joint
sensitivity.input_positions
sensitivity.coordinate_derivatives

sensitivity.body_pose_derivatives(body)
sensitivity.point_position_derivatives(point)
sensitivity.joint_coordinate_derivatives(joint)
```

For `N` configurations and `n` mobile bodies, `coordinate_derivatives` has shape `(N, 3*n)`. The selected input joint has coordinate derivative one up to numerical precision.

The values are returned in the user's coordinate convention. With a revolute input, translational components are length per input angle; with a prismatic input, translational components are length per input displacement. Kimech does not attach a unit package.

Sensitivity reuses the dimensionless scaled driven Jacobian formulation. If a reliable tangent cannot be computed at a requested sample, `input_sensitivity()` raises `KinematicSolveError` with structured failure context. The original position solution remains valid and unchanged.

Sensitivity is deliberately not computed by every `solve()` call and no `compute_sensitivity` flag is provided. This keeps analysis optional and avoids redefining a valid position solution when the selected driven coordinate becomes locally singular.

See [`study-0.5.0-input-sensitivity.md`](study-0.5.0-input-sensitivity.md).

## 10. Intentional breaking changes

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

## 11. Solve diagnostics

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

A compact descriptive summary is available through:

```python
summary = solution.diagnostics.summary()

summary.sample_count
summary.worst_condition_index
summary.worst_condition_number
summary.minimum_singular_value_index
summary.minimum_singular_value
summary.minimum_rank
summary.max_subdivision_index
summary.max_subdivision_count
summary.max_corrector_attempt_index
summary.max_corrector_attempts
summary.strategy_counts
```

`SolveDiagnosticSummary` reports extrema and solve effort only. It deliberately does not define a universal singularity threshold or an `is_singular` flag.

## 12. Adaptive subdivision

Position sweeps use predictor-corrector continuation. If a requested sample cannot be solved from the predictor, Kimech retries from the previous accepted configuration. If both attempts fail and there is a previous accepted sample, Kimech may recursively bisect the input interval and solve internal intermediate configurations.

The subdivision is an internal recovery mechanism:

- user-supplied `input_positions` are never expanded in the returned solution;
- internal intermediate configurations are discarded after they help reach the requested target;
- subdivision depth is bounded internally;
- if recovery fails, the original requested-target `KinematicSolveError` is preserved.

Adaptive subdivision does not make unreachable targets solvable and does not cross folds where the selected `input_joint` ceases to be a valid local parameter. See `docs/study-0.4.0-singularity-diagnostics.md` for examples involving slider-crank dead-centers and four-bar rocker toggles.

## 13. Errors

Public exception hierarchy:

```text
KimechError
├── InvalidModelError
└── KinematicSolveError
```

`InvalidModelError` reports structurally invalid models or solve problems.

`KinematicSolveError` reports numerical failures at position, velocity, or acceleration level. Its message remains human-readable and backward-compatible, while `error.context` may provide a structured immutable `SolveFailureContext`:

```python
try:
    solution = solve(...)
except KinematicSolveError as error:
    context = error.context
```

When available, the context records the kinematic stage, requested input index and position, independently checked residual norm, scaled-Jacobian condition number / minimum singular value / rank, and requested-sample recovery information such as attempted strategies and corrector-attempt count.

Unavailable quantities are represented by `None`; Kimech does not fabricate diagnostics when a reliable candidate or Jacobian cannot be evaluated. The structured context describes Kimech semantics and does not expose SciPy-specific counters.

No public singularity exception, universal condition-number threshold, or binary singularity policy is defined.

## 14. Validation

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

## 15. Visualization

Visualization requires `[viz]` and remains Matplotlib-only.

### Static plotting

```python
fig, ax = plot(config)
fig.savefig("mechanism.svg")
```

Kimech renders a schematic rigid-body scaffold rather than physical/CAD geometry. Structural joint points define the primary scaffold. Mobile links with fewer than two structural points fall back to their declared body points so plate-like bodies remain visually coherent. Auxiliary points on an already-defined scaffold remain markers and receive lightweight visual connectors to that scaffold. Ground does not use this fallback.

Joint glyph sizes are scaled from effective body scaffolds rather than arbitrary remote auxiliary points. The plot bounds still include all rendered geometry.

### Topology plotting

```python
fig, ax = plot_topology(mechanism)
# or:
fig, ax = plot_topology(mechanism.topology())
```

`plot_topology()` renders the structural body/joint multigraph rather than physical mechanism geometry. Body-node positions are deterministic display coordinates and do not encode link dimensions, local point coordinates, or a solved configuration.

Ground and mobile bodies are visually distinct. Revolute and prismatic joints use different glyphs, and multiple joints between the same two bodies are rendered as separate curved connections so joint identity is preserved. Disconnected components are laid out separately.

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

## 16. Result snapshot semantics

Public `Configuration` and `KinematicSolution` constructors validate the structure, shape, finiteness, and entity compatibility of supplied state. They do not certify that manually supplied coordinates satisfy the mechanism constraints. Results returned by `solve()` contain states accepted by the solver.

For `Configuration`, any prescribed-input metadata (`input_position`, `input_velocity`, or `input_acceleration`) requires `input_joint`. Input acceleration additionally requires input velocity.

Result objects retain the link layout captured when they are constructed or solved. Solve-generated `KinematicSolution` objects also retain the associated joint snapshot for downstream analyses such as input sensitivity. This keeps the mapping between entities and stored state stable even if the mechanism object is later extended. Queries require entities compatible with the retained snapshot.

## 17. Examples and tests

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

The test suite covers model/constraint behavior, position solving, differential result state, velocity and acceleration solves, analytic second-order terms, four-bar and slider-crank acceptance, prismatically driven inverse analysis, topology and topology visualization, structured failure observability, input sensitivity, complex-mechanism regression acceptance, visualization, and package metadata.

## 18. Deliberately absent API

`0.6.0` currently provides one `KinematicDriver` but not multiple simultaneous drivers, general multi-DOF solving, motion laws, explicit time histories, dynamics, forces, masses/inertias, pseudo-arclength continuation, branch enumeration, renderer/backend registries, or mechanism-specific solver classes. The release adds structural topology introspection, structured solve observability, and explicit one-input sensitivity without defining a universal near-singularity policy.
