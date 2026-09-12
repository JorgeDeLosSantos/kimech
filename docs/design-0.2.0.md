# Kimech — Design baseline for `0.2.0`

> Status: reviewed planning baseline for `0.2.0`.
>
> This document defines the intended scope and design direction for Kimech `0.2.0`. It extends the `0.1.0` position-kinematics baseline recorded in [`design.md`](design.md). Until `0.2.0` is implemented and released, [`api.md`](api.md) remains the source of truth for the current public API.

## 1. Version purpose

Kimech `0.2.0` extends the library from position kinematics to **differential kinematics** for the same class of planar rigid-body mechanisms already supported by `0.1.0`.

The version adds first- and second-order kinematic state without broadening the mechanism model itself.

Conceptually, the supported state grows from

\[
\mathbf q
\]

to

\[
\mathbf q,\qquad \dot{\mathbf q},\qquad \ddot{\mathbf q}.
\]

The intended identity of the release is:

> **Kimech `0.2.0` provides position, velocity, and acceleration kinematics for one-DOF planar mechanisms with revolute and prismatic joints.**

## 2. Scope

`0.2.0` should support:

- all `0.1.0` planar rigid-body models;
- revolute and prismatic joints;
- one prescribed natural joint coordinate;
- the existing position-solve semantics from `0.1.0`;
- prescribed input velocity;
- prescribed input acceleration;
- generalized coordinate velocities and accelerations;
- body pose, velocity, and acceleration queries;
- point position, velocity, and acceleration queries;
- joint coordinate, velocity, and acceleration queries;
- scalar solves returning `Configuration`;
- sweep solves returning `KinematicSolution`;
- analytic differential kinematics based on the existing constraint Jacobian;
- explicit failure when the requested differential state cannot be solved reliably.

The mechanism model remains intentionally unchanged. Differential state belongs to the solved kinematic problem and to result objects, not to `Mechanism`, `Link`, `Ground`, `Point`, or joint metadata.

## 3. Explicitly out of scope

The following remain outside `0.2.0`:

- mechanisms with structural mobility greater than one;
- multiple prescribed inputs;
- additional public joint types;
- explicit time arrays;
- motion-law objects such as `u(t)`;
- numerical differentiation of sampled trajectories as the production method;
- numerical time integration;
- dynamics;
- forces and torques;
- mass and inertia properties;
- formal singularity analysis;
- public condition-number diagnostics;
- characteristic-length scaling;
- adaptive continuation;
- automatic branch discovery or branch switching;
- richer visualization of velocity or acceleration vectors;
- additional visualization backends;
- partial-solution public APIs;
- Hessian objects or explicit public `J_dot` APIs.

Solver robustness and richer singular/toggle diagnostics remain candidates for a later release.

## 4. High-level solve API

`solve()` remains the single public high-level solving entry point.

The intended `0.2.0` signature is conceptually:

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

The arguments mean:

- `input` identifies the prescribed revolute or prismatic joint;
- `values` contains the prescribed natural joint coordinate `u`;
- `input_velocity` contains `u_dot` when velocity kinematics is requested;
- `input_acceleration` contains `u_ddot` when acceleration kinematics is requested;
- `initial_guess` retains the same branch-selection role as in `0.1.0`.

The requested solution level is determined only by presence or absence of the differential input data:

```text
values only
    -> position

values + input_velocity
    -> position + velocity

values + input_velocity + input_acceleration
    -> position + velocity + acceleration
```

`input_acceleration` without `input_velocity` is invalid.

A numeric zero is a physical value, not absence of data. Therefore `input_velocity=0.0` requests velocity kinematics with zero prescribed input velocity, whereas `input_velocity=None` means that velocity kinematics is not requested.

## 5. Input shape and broadcasting rules

The shape of `values` determines the problem shape.

### 5.1 Scalar problem

If `values` is scalar:

- `input_velocity`, when supplied, must also be scalar;
- `input_acceleration`, when supplied, must also be scalar;
- the result is a `Configuration`.

Zero-dimensional NumPy numeric arrays follow the same scalar semantics already used for `values`.

### 5.2 Sweep problem

If `values` is one-dimensional with shape `(N,)`:

- `input_velocity` may be a scalar or an array with shape `(N,)`;
- `input_acceleration` may be a scalar or an array with shape `(N,)`;
- scalar differential inputs are broadcast explicitly to all configurations;
- non-scalar inputs must have exactly shape `(N,)`;
- the result is a `KinematicSolution`.

General NumPy broadcasting is not part of the public contract. In particular, an array with shape `(1,)` is not treated as a scalar broadcast value.

All prescribed values must be numeric and finite. The complete input specification is validated and normalized before any position solve begins.

## 6. Time semantics

The `0.1.0` distinction remains in force: `values` parameterizes configurations and is not itself interpreted as a time array.

The dot notation in `0.2.0` nevertheless has its ordinary physical meaning. The quantities

\[
\dot u,\quad \ddot u,\quad \dot{\mathbf q},\quad \ddot{\mathbf q}
\]

are derivatives with respect to a **single common external physical time variable**. Kimech receives the derivative values but does not need to receive or store the corresponding time samples.

A user may externally sample a motion law using a time array and pass the resulting `u`, `u_dot`, and `u_ddot` arrays to Kimech. All supplied differential quantities for a sample are understood to refer to the same instant and the same external time variable.

Animation FPS remains a presentation parameter and does not become a physical time step.

## 7. Differential kinematics formulation

The position problem remains

\[
\Phi(\mathbf q,u)=0.
\]

The analytic position Jacobian already used by `0.1.0` is

\[
J(\mathbf q)=\frac{\partial \Phi}{\partial \mathbf q}.
\]

### 7.1 Velocity

Differentiating the constraint equations gives

\[
J(\mathbf q)\dot{\mathbf q}+\Phi_u\dot u=0.
\]

Because the prescribed-coordinate equation has the form

\[
c_J(\mathbf q)-u=0,
\]

while the geometric joint constraints do not depend explicitly on `u`, the velocity system can be written as

\[
\boxed{J(\mathbf q)\dot{\mathbf q}=\mathbf b_v(\dot u)}.
\]

`b_v` is zero for the geometric constraints and contains the prescribed input velocity in the driver row.

Velocity is solved directly from the differentiated constraints. It is not estimated from differences between neighboring configurations.

### 7.2 Acceleration

Differentiating again gives a linear system in the unknown generalized accelerations:

\[
\boxed{J(\mathbf q)\ddot{\mathbf q}=\mathbf b_a(\mathbf q,\dot{\mathbf q},\ddot u)}.
\]

The implementation reuses the same position Jacobian and computes only the known second-order terms required on the right-hand side.

Kimech should not construct explicit Hessian tensors or store an explicit `J_dot` matrix merely to obtain the contraction needed by the acceleration equations. Instead, each internal constraint contribution should provide its analytic acceleration bias terms directly.

Finite differences may be used in tests as an independent numerical oracle, but not as the production differential-kinematics method.

## 8. Point differential kinematics

For a point with local coordinates `s` rigidly attached to a mobile link,

\[
\mathbf r=\mathbf p+R(\theta)\mathbf s.
\]

Its velocity is

\[
\mathbf v=\dot{\mathbf p}+\omega R(\theta)E\mathbf s,
\]

and its acceleration is

\[
\mathbf a=\ddot{\mathbf p}+\alpha R(\theta)E\mathbf s-\omega^2R(\theta)\mathbf s.
\]

Point velocity and acceleration queries are derived from the solved generalized state rather than stored redundantly.

For ground points the physical velocity and acceleration are zero. However, differential queries still require the corresponding result capability: a position-only result does not answer velocity queries merely because the requested entity happens to be fixed.

## 9. Joint differential coordinates

Joint velocities and accelerations are the first and second time derivatives of the same natural coordinates already defined in `0.1.0`.

### 9.1 Revolute joint

For

\[
\phi_J=\theta_B-\theta_A,
\]

Kimech defines

\[
\dot\phi_J=\omega_B-\omega_A,
\qquad
\ddot\phi_J=\alpha_B-\alpha_A.
\]

### 9.2 Prismatic joint

To avoid conflating axis vectors with accelerations, denote the unit axis carried by side A as

\[
\hat{\mathbf e}_A=R(\theta_A)\mathbf a_A^L.
\]

For

\[
s_J=\hat{\mathbf e}_A^T(\mathbf r_{P_B}-\mathbf r_{P_A}),
\]

its velocity on a valid prismatic configuration is

\[
\dot s_J=\hat{\mathbf e}_A^T(\mathbf v_{P_B}-\mathbf v_{P_A}).
\]

Let

\[
\mathbf d=\mathbf r_{P_B}-\mathbf r_{P_A}=s_J\hat{\mathbf e}_A.
\]

Projection of the relative point acceleration along the moving axis gives

\[
\boxed{
\ddot s_J
=\hat{\mathbf e}_A^T(\mathbf a_{P_B}-\mathbf a_{P_A})+s_J\omega_A^2
}.
\]

The existing `axis_a` orientation continues to define the positive direction of

\[
s_J,\qquad \dot s_J,\qquad \ddot s_J.
\]

`axis_b` continues to define the orientation relation enforced by the prismatic joint; it does not redefine the sign of the natural coordinate. Joint argument order therefore remains semantically meaningful.

## 10. Result model

`Configuration` and `KinematicSolution` remain the primary public result types. No `VelocitySolution`, `AccelerationSolution`, or separate differential-result hierarchy is introduced.

### 10.1 `Configuration`

A configuration always contains position state and may additionally contain differential state:

```text
q                       always
q_dot                   optional
q_ddot                  optional
```

It may also retain prescribed-input metadata:

```text
input_joint             optional
input_value             optional
input_velocity          optional
input_acceleration      optional
```

The core state invariant is

\[
\ddot{\mathbf q}\ \Rightarrow\ \dot{\mathbf q}.
\]

For prescribed-input metadata,

\[
\ddot u\ \Rightarrow\ \dot u.
\]

Solver-created differential results always contain the matching prescribed differential input metadata. Directly constructed result objects may contain differential state without prescribed-input provenance.

Public arrays remain safe copies and the object remains conceptually immutable.

### 10.2 `KinematicSolution`

A solution stores the dense state histories

\[
Q\in\mathbb R^{N\times3n},
\]

and, when available,

\[
\dot Q\in\mathbb R^{N\times3n},
\qquad
\ddot Q\in\mathbb R^{N\times3n}.
\]

Scalar differential input values supplied through `solve()` are normalized internally to full `(N,)` histories.

Derived body, point, and joint quantities are computed from the stored generalized state rather than duplicated eagerly.

### 10.3 Indexing

`solution[i]` returns a `Configuration` containing all kinematic state available at sample `i`, together with the corresponding prescribed-input metadata when that metadata is available.

Indexing must not discard differential information.

## 11. Result capability checks and unavailable data

Both result types expose:

```python
has_velocity
has_acceleration
```

These are boolean indicators of **generalized kinematic state availability**, not merely of prescribed-input metadata.

The following rules apply:

- `has_velocity` is true exactly when generalized velocity state is present;
- `has_acceleration` is true exactly when generalized acceleration state is present;
- acceleration state implies velocity state;
- body, point, joint, and generalized-state differential queries require the corresponding capability;
- unavailable kinematic state raises a clear error rather than returning `None`.

Prescribed-input fields are provenance/problem metadata and may be absent on directly constructed solver-independent result objects. This is distinct from absence of generalized kinematic state.

## 12. Public result API direction

The intended public naming is organized by entity type.

### 12.1 Generalized state

```python
config.coordinates
config.coordinate_velocities
config.coordinate_accelerations

solution.coordinates
solution.coordinate_velocities
solution.coordinate_accelerations
```

Shapes are `(3*n,)` for a `Configuration` and `(N, 3*n)` for a `KinematicSolution`. For each mobile body, the three entries are respectively translational x/y state and angular state.

### 12.2 Prescribed input metadata

```python
config.input_joint
config.input_value
config.input_velocity
config.input_acceleration

solution.input_joint
solution.input_values
solution.input_velocities
solution.input_accelerations
```

For solver-created sweep results, differential input metadata is stored as normalized `(N,)` histories when requested.

### 12.3 Bodies

```python
config.body_pose(body)
config.body_velocity(body)
config.body_acceleration(body)

solution.body_poses(body)
solution.body_velocities(body)
solution.body_accelerations(body)
```

`body` means `Link | Ground`.

For a mobile link these return, respectively,

\[
[x,\ y,\ \theta],\qquad
[\dot x,\ \dot y,\ \omega],\qquad
[\ddot x,\ \ddot y,\ \alpha]
\]

for the link's local-frame origin and orientation. Configuration-level shapes are `(3,)`; history shapes are `(N, 3)`.

For ground, `body_pose()` is the zero-coordinate identity pose and differential body state is zero when the corresponding differential capability exists.

### 12.4 Points

```python
config.position(point)
config.velocity(point)
config.acceleration(point)

solution.point_path(point)
solution.point_velocities(point)
solution.point_accelerations(point)
```

Configuration-level point quantities have shape `(2,)`; history quantities have shape `(N, 2)`.

### 12.5 Joints

```python
config.joint_coordinate(joint)
config.joint_velocity(joint)
config.joint_acceleration(joint)

solution.joint_coordinates(joint)
solution.joint_velocities(joint)
solution.joint_accelerations(joint)
```

Configuration-level joint quantities are scalars; history quantities have shape `(N,)`.

No public `q`, `q_dot`, or `q_ddot` aliases are required.

## 13. Direct result construction

Result objects remain solver-independent and may be constructed with externally obtained kinematic state.

The intended constructor direction is:

```python
Configuration(
    mechanism,
    coordinates,
    *,
    coordinate_velocities=None,
    coordinate_accelerations=None,
    input_joint=None,
    input_value=None,
    input_velocity=None,
    input_acceleration=None,
)
```

and conceptually:

```python
KinematicSolution(
    mechanism,
    input_joint,
    input_values,
    coordinates,
    *,
    coordinate_velocities=None,
    coordinate_accelerations=None,
    input_velocities=None,
    input_accelerations=None,
)
```

Direct `KinematicSolution` construction uses already normalized histories: optional differential arrays must have the exact required shapes and are not subject to the scalar broadcasting convenience of `solve()`.

The constructor must reject generalized acceleration state without generalized velocity state. Prescribed acceleration metadata requires prescribed velocity metadata. Exact consistency checks for optional provenance metadata should remain lightweight; result constructors are not replacements for solving or constraint validation.

## 14. Planned `0.2.0` naming cleanup

Because Kimech remains pre-`1.0` and the result API is expanding around a body/point/joint taxonomy, `0.2.0` standardizes the existing body pose names.

The intentional breaking changes are:

```text
Configuration.pose()
    -> Configuration.body_pose()

KinematicSolution.link_poses()
    -> KinematicSolution.body_poses()
```

No deprecated aliases are planned for this early release.

This rename is a repository-wide migration. In the same implementation block, internal callers, solver initial-guess handling, visualization, tests, examples, and playground clients must be updated so the branch remains internally consistent and CI remains green.

The `0.1.0` design document remains historical and is not rewritten to use the new names.

## 15. Solve phases

Although the public API remains a single `solve()` call, the implementation uses distinct internal phases.

For a sweep:

```text
validate and normalize complete input specification
                    |
                    v
solve complete position sweep using continuation
                    |
                    v
                    Q
                    |
                    +--> if requested: solve all velocity states -> Q_dot
                    |
                    +--> if requested: solve all acceleration states -> Q_ddot
```

Position continuation remains exclusively a position-level process. Differential state must not influence branch tracking.

Therefore, for identical `values` and `initial_guess`, adding `input_velocity` or `input_acceleration` must not alter the accepted position matrix `Q`.

Each differential sample is local once its configuration is known. Velocity and acceleration do not require continuation from the preceding differential sample.

## 16. Internal package direction

The public package remains small and flat. A private differential module may be introduced if it keeps responsibilities clear:

```text
src/kimech/
    model.py
    joints.py
    solution.py
    solver.py
    validation.py
    errors.py
    _geometry.py
    _constraints.py
    _differential.py      # candidate private module
```

A reasonable responsibility split is:

- `_constraints.py` owns geometric residuals, the analytic position Jacobian, and analytic differential constraint contributions/bias terms;
- `_differential.py` owns private linear velocity and acceleration solution routines;
- `solver.py` validates the public problem, performs position continuation, invokes differential phases when requested, and constructs result objects;
- `solution.py` stores solver-independent kinematic state and exposes derived queries.

No public `VelocitySolver`, `AccelerationSolver`, or solver-class hierarchy is introduced. The exact private helper names remain implementation details.

## 17. Differential solve verification

The linear algebra routine alone does not determine whether a differential state is accepted.

A candidate velocity must:

- have the expected shape;
- contain only finite values;
- satisfy

\[
\mathbf r_v=J\dot{\mathbf q}-\mathbf b_v
\]

under an appropriate private residual criterion.

A candidate acceleration is checked analogously using

\[
\mathbf r_a=J\ddot{\mathbf q}-\mathbf b_a.
\]

The exact private acceptance tolerances are selected and tested during implementation. They are not exposed in the public `solve()` signature in `0.2.0`.

No condition-number rejection threshold is introduced. A large but finite differential response near a critical configuration may be physically meaningful and should not be rejected merely because its magnitude is large.

## 18. Failure semantics

`KinematicSolveError` remains the public exception for numerical failures in all kinematic solve phases.

No separate public exceptions such as `VelocitySolveError`, `AccelerationSolveError`, or `SingularityError` are introduced in `0.2.0`.

Differential solve failures translate lower-level linear algebra failures into `KinematicSolveError` rather than leaking implementation-specific exceptions such as `numpy.linalg.LinAlgError`.

Error messages should identify available context including:

- solve stage: position, velocity, or acceleration;
- input index for a sweep;
- prescribed input value;
- residual norm when available;
- useful numerical failure reason when available.

A failed differential solve should not automatically be labeled a physical mechanism singularity. Formal interpretation of singular/toggle configurations is deferred.

The requested solve level is all-or-nothing:

```text
requested position
    -> valid Q required

requested position + velocity
    -> valid Q and Q_dot required

requested position + velocity + acceleration
    -> valid Q, Q_dot, and Q_ddot required
```

If velocity or acceleration fails, `solve()` does not silently downgrade the returned result to a lower kinematic level. Partial-solution payloads are not part of the `0.2.0` public error API.

## 19. Acceptance mechanisms

The generic infrastructure continues to be validated against ordinary `Mechanism` models rather than mechanism-specific solver classes.

### 19.1 Four-bar linkage

A four-bar mechanism should support a regular prescribed crank sweep with:

- position state;
- prescribed crank velocity;
- prescribed crank acceleration;
- generalized velocities and accelerations;
- body differential state;
- coupler-point velocity and acceleration;
- revolute joint velocity and acceleration.

The position path must be unchanged from the position-only solve for the same prescribed coordinate sequence and initial branch guess.

### 19.2 Slider-crank mechanism

The slider-crank should exercise both revolute and prismatic differential semantics.

Acceptance should include comparison against independent analytical relations where practical for

\[
x(\theta),\qquad \dot x,\qquad \ddot x.
\]

The fixed-guide case should verify that the prismatic joint coordinate, velocity, and acceleration have the expected signed behavior.

The existing inverse case with a prismatically driven mechanism should also be exercised on a regular admissible interval so that `s`, `s_dot`, and `s_ddot` are validated as prescribed inputs rather than only as output queries.

## 20. Test strategy

Testing should include several complementary levels.

### 20.1 Regression

All existing `0.1.0` position behavior should remain valid apart from the intentional body-pose method rename. Position-only `solve()` must not require differential data.

### 20.2 Analytic identities

Useful invariants include:

- doubling prescribed input velocity at a fixed regular configuration doubles generalized velocity;
- zero prescribed input velocity produces zero first-order response at a regular one-DOF configuration;
- with zero prescribed input acceleration, scaling input velocity by a factor `k` scales the velocity-dependent generalized acceleration contribution by `k**2`;
- with zero prescribed input velocity, scaling prescribed input acceleration scales generalized acceleration linearly;
- body and point differential queries agree with rigid-body formulas;
- revolute joint derivatives agree with relative angular derivatives;
- prismatic joint derivatives agree with derivatives of the existing natural coordinate.

### 20.3 Numerical differentiation as test oracle

Finite differences may independently check analytic implementation details, including:

- point velocity expressions;
- point acceleration expressions;
- joint coordinate derivatives;
- acceleration bias terms;
- directional behavior corresponding to `J_dot q_dot` without exposing or storing `J_dot` in production code.

Finite-difference tolerances belong only to tests and must not define the production algorithm.

### 20.4 End-to-end acceptance

Four-bar and slider-crank tests should exercise complete `solve()` calls with position, velocity, and acceleration requested together.

## 21. Definition of done

Kimech `0.2.0` is considered functionally complete when the same generic mechanism model and solver infrastructure can reliably produce position, velocity, and acceleration for the accepted one-DOF planar R/P mechanisms, and when users can query the resulting state consistently at body, point, and joint levels without mechanism-specific equations or mechanism-specific solver classes.

The release should preserve the small-library design principles established in `0.1.0`: declarative models, solver-independent result objects, analytic kinematics, minimal public abstraction, and explicit failure rather than silent degradation.

## 22. Proposed implementation blocks

A likely implementation sequence is:

1. **Result model and repository-wide naming update**
   - rename body pose queries and migrate every repository caller in the same block;
   - extend `Configuration` and `KinematicSolution` with optional differential state;
   - add capability flags and differential input metadata;
   - add body/point/joint differential queries operating on supplied state;
   - add focused result-model tests while keeping the existing suite green.

2. **Velocity infrastructure**
   - add input normalization for `input_velocity`;
   - assemble velocity right-hand sides;
   - solve and verify `J q_dot = b_v`;
   - integrate velocity histories into `solve()`.

3. **Acceleration infrastructure**
   - implement analytic second-order bias terms;
   - add input normalization for `input_acceleration`;
   - solve and verify `J q_ddot = b_a`;
   - integrate acceleration histories into `solve()`.

4. **Acceptance and regression hardening**
   - four-bar differential tests;
   - slider-crank analytical comparisons;
   - prismatically driven inverse acceptance case;
   - finite-difference verification tests;
   - error-path and shape-validation tests.

5. **Documentation and release cleanup**
   - update `api.md` for the implemented `0.2.0` API;
   - update README and user-facing examples where additional differential demonstrations are useful;
   - update version metadata;
   - document intentional breaking renames and release scope.

The repository-wide rename itself is completed in block 1; block 5 is for final user-facing documentation and release presentation, not for repairing stale code clients.

These blocks are planning units, not public architecture. They may be split into smaller feature branches or pull requests during implementation.

## 23. Open implementation details

The following details are intentionally left to implementation evidence and do not block the version design:

- exact private helper/function names;
- whether all differential helpers live in `_differential.py` or some remain in `_constraints.py`;
- exact private residual tolerances for velocity and acceleration acceptance;
- exact wording and exception subclass choice, if any, for unavailable-state access that is not a solve failure;
- lightweight consistency rules for optional prescribed-input provenance on directly constructed result objects.

These details should be resolved while preserving the public semantics and invariants recorded above.
