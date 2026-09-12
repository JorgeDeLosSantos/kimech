# Kimech — Design baseline for `0.2.0`

> Status: planning draft for `0.2.0`.
>
> This document defines the intended scope and design direction for Kimech `0.2.0`. It extends the `0.1.0` position-kinematics baseline recorded in [`design.md`](design.md). Until `0.2.0` is implemented and released, [`api.md`](api.md) remains the source of truth for the current public API.

## 1. Version purpose

Kimech `0.2.0` extends the library from position kinematics to **differential kinematics** for the same class of planar rigid-body mechanisms already supported by `0.1.0`.

The version should add first- and second-order kinematic state without broadening the mechanism model itself.

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
- position-only solves exactly as in `0.1.0`;
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
- numerical differentiation of sampled trajectories as the core method;
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

The meaning of the arguments is:

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

A numeric zero is a physical value, not absence of data. Therefore:

```python
input_velocity=0.0
```

requests velocity kinematics with zero prescribed input velocity, whereas:

```python
input_velocity=None
```

means that velocity kinematics is not requested.

## 5. Input shape and broadcasting rules

The shape of `values` determines the problem shape.

### 5.1 Scalar problem

If `values` is scalar:

- `input_velocity`, when supplied, must also be scalar;
- `input_acceleration`, when supplied, must also be scalar;
- the result is a `Configuration`.

### 5.2 Sweep problem

If `values` is one-dimensional with shape `(N,)`:

- `input_velocity` may be a scalar or an array with shape `(N,)`;
- `input_acceleration` may be a scalar or an array with shape `(N,)`;
- scalar differential inputs are broadcast explicitly to all configurations;
- non-scalar inputs must have exactly shape `(N,)`;
- the result is a `KinematicSolution`.

General NumPy broadcasting is not part of the public contract. For example, an array with shape `(1,)` is not treated as a scalar broadcast value.

All prescribed values must be numeric and finite.

The complete input specification is validated and normalized before any position solve begins.

## 6. Input values still do not imply time

The `0.1.0` semantic distinction remains in force:

```python
values = ...
```

parameterizes configurations and does not itself represent physical time.

`0.2.0` adds the ability to associate each prescribed coordinate value with independently supplied

\[
u,\qquad \dot u,\qquad \ddot u,
\]

but Kimech does not need to know how those values were generated.

A user may externally sample a motion law using a time array and pass the resulting `u`, `u_dot`, and `u_ddot` arrays to Kimech, but time remains outside the core model and result semantics.

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
J(\mathbf q)\dot{\mathbf q}
+
\Phi_u\dot u
=0.
\]

Because the prescribed-coordinate equation has the form

\[
c_J(\mathbf q)-u=0,
\]

while the geometric joint constraints do not depend explicitly on `u`, the velocity system can be written as

\[
\boxed{
J(\mathbf q)\dot{\mathbf q}
=
\mathbf b_v(\dot u)
}
\]

where `b_v` is zero for the geometric constraints and contains the prescribed input velocity in the driver row.

Velocity is solved directly from the differentiated constraints. It is not estimated from differences between neighboring configurations.

### 7.2 Acceleration

Differentiating again gives a linear system in the unknown generalized accelerations:

\[
\boxed{
J(\mathbf q)\ddot{\mathbf q}
=
\mathbf b_a(\mathbf q,\dot{\mathbf q},\ddot u)
}
\]

The implementation should reuse the same position Jacobian and compute only the known second-order terms required on the right-hand side.

Kimech should not construct explicit Hessian tensors or store an explicit `J_dot` matrix merely to obtain the contraction needed by the acceleration equations.

Instead, each internal constraint contribution should provide its analytic acceleration bias terms directly.

Finite differences may be used in tests as an independent numerical oracle, but not as the production differential-kinematics method.

## 8. Point differential kinematics

For a point with local coordinates `s` rigidly attached to a mobile link,

\[
\mathbf r
=
\mathbf p + R(\theta)\mathbf s.
\]

Its velocity is

\[
\mathbf v
=
\dot{\mathbf p}
+
\omega R(\theta)E\mathbf s,
\]

and its acceleration is

\[
\mathbf a
=
\ddot{\mathbf p}
+
\alpha R(\theta)E\mathbf s
-
\omega^2R(\theta)\mathbf s.
\]

Point velocity and acceleration queries should be derived from the solved generalized state rather than stored redundantly.

Ground points have zero velocity and zero acceleration.

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
\]

and

\[
\ddot\phi_J=\alpha_B-\alpha_A.
\]

### 9.2 Prismatic joint

For

\[
s_J
=
\mathbf a_A^T(\mathbf r_B-\mathbf r_A),
\]

with

\[
\mathbf a_A=R(\theta_A)\mathbf a_A^L,
\]

its velocity on a valid prismatic configuration is

\[
\dot s_J
=
\mathbf a_A^T(\mathbf v_B-\mathbf v_A).
\]

Its acceleration is the second derivative of the same signed coordinate. If

\[
\mathbf d=\mathbf r_B-\mathbf r_A=s_J\mathbf a_A,
\]

then projection along the moving axis gives

\[
\ddot s_J
=
\mathbf a_A^T(\mathbf a_B-\mathbf a_A^{P})
+s_J\omega_A^2,
\]

where the accelerations refer to the two prismatic reference points.

The existing `axis_a` orientation continues to define the positive direction of

\[
s_J,\qquad \dot s_J,\qquad \ddot s_J.
\]

Joint argument order therefore remains semantically meaningful.

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

The following invariants hold:

- `q_ddot` cannot exist without `q_dot`;
- `input_acceleration` cannot exist without `input_velocity`;
- public arrays remain safe copies;
- the object remains conceptually immutable.

A user may construct a `Configuration` directly with valid differential state; differential information is not restricted to solver-created objects.

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

Scalar differential input values supplied by the user are normalized internally to full `(N,)` histories.

Derived body, point, and joint quantities are computed from the stored generalized state rather than duplicated eagerly.

### 10.3 Indexing

`solution[i]` returns a `Configuration` containing all kinematic state available at sample `i`:

\[
q_i,
\]

optionally

\[
\dot q_i,
\]

optionally

\[
\ddot q_i,
\]

and the corresponding prescribed-input metadata

\[
u_i,\qquad \dot u_i,\qquad \ddot u_i.
\]

Indexing must not discard differential information.

## 11. Result capability checks

Both result types should expose:

```python
has_velocity
has_acceleration
```

These are boolean capability indicators.

Requesting unavailable differential data should fail explicitly rather than return `None`.

For example, a position-only result should raise a clear error for a velocity query such as:

```python
config.velocity(point)
```

and a position-plus-velocity result should raise a clear error for an acceleration query.

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

### 12.3 Bodies

```python
config.body_pose(body)
config.body_velocity(body)
config.body_acceleration(body)

solution.body_poses(body)
solution.body_velocities(body)
solution.body_accelerations(body)
```

`body` means `Link | Ground`. Ground returns zero pose velocity and zero pose acceleration.

### 12.4 Points

```python
config.position(point)
config.velocity(point)
config.acceleration(point)

solution.point_path(point)
solution.point_velocities(point)
solution.point_accelerations(point)
```

### 12.5 Joints

```python
config.joint_coordinate(joint)
config.joint_velocity(joint)
config.joint_acceleration(joint)

solution.joint_coordinates(joint)
solution.joint_velocities(joint)
solution.joint_accelerations(joint)
```

No public `q`, `q_dot`, or `q_ddot` aliases are required.

## 13. Planned `0.2.0` naming cleanup

Because Kimech remains pre-`1.0` and the result API is expanding around a body/point/joint taxonomy, `0.2.0` should standardize the existing body pose names.

The planned breaking changes are:

```text
Configuration.pose()
    -> Configuration.body_pose()

KinematicSolution.link_poses()
    -> KinematicSolution.body_poses()
```

No deprecated aliases are planned for this early release. Documentation, examples, and tests should be updated together.

The resulting body family is intentionally symmetric:

```text
body_pose
body_velocity
body_acceleration
```

with plural history methods on `KinematicSolution`.

## 14. Solve phases

Although the public API remains a single `solve()` call, the implementation should use distinct internal phases.

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

## 15. Internal package direction

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

No public `VelocitySolver`, `AccelerationSolver`, or solver-class hierarchy is introduced.

The exact private helper names remain implementation details.

## 16. Differential solve verification

The linear algebra routine alone does not determine whether a differential state is accepted.

After solving velocity, Kimech should independently verify

\[
\mathbf r_v
=
J\dot{\mathbf q}-\mathbf b_v
\]

and require an appropriate residual criterion.

Likewise, acceleration should verify

\[
\mathbf r_a
=
J\ddot{\mathbf q}-\mathbf b_a.
\]

The exact private tolerances should be selected and tested during implementation. They are not exposed in the public `solve()` signature in `0.2.0`.

No condition-number rejection threshold is introduced. A large but finite differential response near a critical configuration may be physically meaningful and should not be rejected merely because its magnitude is large.

## 17. Failure semantics

`KinematicSolveError` remains the public exception for numerical failures in all kinematic solve phases.

No separate public exceptions such as `VelocitySolveError`, `AccelerationSolveError`, or `SingularityError` are introduced in `0.2.0`.

Differential solve failures should translate lower-level linear algebra failures into `KinematicSolveError` rather than leaking implementation-specific exceptions such as `numpy.linalg.LinAlgError`.

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

If velocity or acceleration fails, `solve()` does not silently downgrade the returned result to a lower kinematic level.

Partial-solution payloads are not part of the `0.2.0` public error API.

## 18. Acceptance mechanisms

The generic infrastructure should continue to be validated against ordinary `Mechanism` models rather than mechanism-specific solver classes.

### 18.1 Four-bar linkage

A four-bar mechanism should support a valid prescribed crank sweep with:

- position state;
- prescribed crank velocity;
- prescribed crank acceleration;
- generalized velocities and accelerations;
- body differential state;
- coupler-point velocity and acceleration;
- revolute joint velocity and acceleration.

The position path must be identical to the position-only solve for the same prescribed coordinate sequence and initial branch guess.

### 18.2 Slider-crank mechanism

The slider-crank should exercise both revolute and prismatic differential semantics.

Acceptance should include comparison against independent analytical relations where practical for:

\[
x(\theta),\qquad \dot x,\qquad \ddot x.
\]

The fixed-guide case should verify that the prismatic joint coordinate, velocity, and acceleration have the expected signed behavior.

The existing inverse case with a prismatically driven mechanism should also be exercised so that `s`, `s_dot`, and `s_ddot` are validated as prescribed inputs rather than only as output queries.

## 19. Test strategy

Testing should include several complementary levels.

### 19.1 Regression

All existing `0.1.0` position behavior should remain valid apart from the intentional body-pose method rename.

Position-only `solve()` must not require differential data.

### 19.2 Analytic identities

Useful invariants include:

- doubling prescribed input velocity at a fixed configuration doubles generalized velocity;
- zero prescribed input velocity produces the corresponding zero first-order response for a regular one-DOF configuration;
- body and point differential queries agree with rigid-body formulas;
- revolute joint derivatives agree with relative angular derivatives;
- prismatic joint derivatives agree with derivatives of the existing natural coordinate.

### 19.3 Numerical differentiation as test oracle

Finite differences may independently check analytic implementation details, including:

- point velocity expressions;
- point acceleration expressions;
- joint coordinate derivatives;
- acceleration bias terms;
- directional behavior corresponding to `J_dot q_dot` without exposing or storing `J_dot` in production code.

Finite-difference tolerances belong only to tests and must not define the production algorithm.

### 19.4 End-to-end acceptance

Four-bar and slider-crank tests should exercise complete `solve()` calls with position, velocity, and acceleration requested together.

## 20. Definition of done

Kimech `0.2.0` is considered functionally complete when the same generic mechanism model and solver infrastructure can reliably produce:

```text
position
velocity
acceleration
```

for the accepted one-DOF planar R/P mechanisms, and when users can query the resulting state consistently at three entity levels:

```text
body
point
joint
```

without mechanism-specific equations or mechanism-specific solver classes.

The release should preserve the small-library design principles established in `0.1.0`: declarative models, solver-independent result objects, analytic kinematics, minimal public abstraction, and explicit failure rather than silent degradation.

## 21. Proposed implementation blocks

A likely implementation sequence is:

1. **Result-model and naming update**
   - rename body pose queries;
   - extend `Configuration` and `KinematicSolution` with optional differential state;
   - add capability flags and input differential metadata;
   - add body/point/joint differential queries.

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
   - update examples and README;
   - update version metadata;
   - document intentional breaking renames and release scope.

These blocks are planning units, not public architecture. They may be split into smaller feature branches or pull requests during implementation.

## 22. Open implementation details

The following details are intentionally left to implementation evidence and do not block the version design:

- exact private helper/function names;
- whether all differential helpers live in `_differential.py` or some remain in `_constraints.py`;
- exact private residual tolerances for velocity and acceleration acceptance;
- the exact wording of unavailable-data and differential-solve error messages;
- whether direct `Configuration` construction uses optional keyword names identical to its public properties or a slightly different constructor organization.

These should be resolved while preserving the public semantics and invariants recorded above.
