# Kimech 0.6.0 — kinematic driver study

## 1. Purpose

This study defines the initial public semantics for a first-class kinematic
driver in Kimech 0.6.0.

The goal is to replace the current solver-level input quartet

```text
input_joint
input_position
input_velocity
input_acceleration
```

with a domain object that represents one prescribed kinematic coordinate and
its requested values.

The initial public candidate is:

```python
driver = KinematicDriver(
    joint=crank_joint,
    position=theta,
    velocity=omega,
    acceleration=alpha,
)
```

followed by:

```python
solution = solve(
    mechanism,
    driver=driver,
    time=t,
    initial_guess=initial_guess,
)
```

where `time` is optional and belongs to the solve history rather than to the
driver itself.

0.6.0 remains a one-DOF planar R/P kinematics release. The purpose of this
change is not to implement general multi-DOF solving, but to make the prescribed
kinematic formulation explicit and to avoid baking the current
`input_joint` representation more deeply into the public API.

## 2. Design statement

The working definition is:

> A `KinematicDriver` prescribes the evolution of one kinematic coordinate
> used to close the current mechanism solve.

For 0.6.0, the only supported coordinate is the natural coordinate of one
`RevoluteJoint` or `PrismaticJoint`.

Mathematically, the mechanism equations are interpreted as

```text
Phi_J(q) = 0
g_D(q) - u = 0
```

where:

- `Phi_J(q)` contains the ordinary joint compatibility equations;
- `g_D(q)` is the coordinate selected by the driver;
- `u` is the prescribed driver position.

For 0.6.0,

```text
g_D(q) = c_J(q)
```

where `c_J` is the existing natural coordinate of the selected R/P joint.

The driver therefore represents a kinematic prescription, not a dynamic
actuator. It does not represent torque, force, motor dynamics, or an integration
law.

## 3. Why `KinematicDriver`

Three main names were considered:

- `JointDriver`;
- `MotionDriver`;
- `KinematicDriver`.

`JointDriver` is concrete and accurate for 0.6.0, but permanently names the
current target type. A later driver could reasonably prescribe a point
coordinate, body orientation, distance, or another mechanism-independent
kinematic quantity.

`MotionDriver` is intuitive, but suggests that the object necessarily contains
an explicit time parameterization. Kimech must continue to support purely
geometric configuration sweeps where no physical time is supplied.

`KinematicDriver` is preferred because it describes the responsibility rather
than the current target implementation:

- it can represent a position-only geometric sweep;
- it can carry physical velocity and acceleration data;
- it remains distinct from future dynamics concepts;
- it can later become a shared abstraction for more than one prescribed
  coordinate family without requiring that generalization in 0.6.0.

This study does not require a public inheritance hierarchy. In 0.6.0
`KinematicDriver` may remain one concrete class specialized internally to a
joint natural coordinate.

## 4. Proposed public construction

The initial API candidate is:

```python
KinematicDriver(
    joint,
    *,
    position,
    velocity=None,
    acceleration=None,
)
```

Equivalent keyword-only `joint=` syntax should also work if the final class
constructor follows normal Python conventions.

### Required data

`joint`:

- must be a `RevoluteJoint` or `PrismaticJoint`;
- identifies the natural coordinate prescribed by the driver;
- must belong to the mechanism supplied later to `solve()`.

`position`:

- is required;
- is the prescribed natural coordinate;
- may be a finite scalar or finite one-dimensional sequence.

### Optional differential data

`velocity`:

- is the physical time derivative of the prescribed coordinate;
- may be a finite scalar or finite one-dimensional sequence;
- a scalar broadcasts over a position history.

`acceleration`:

- is the physical second time derivative of the prescribed coordinate;
- may be a finite scalar or finite one-dimensional sequence;
- a scalar broadcasts over a position history;
- requires `velocity`.

No numerical differentiation is implied by omitted differential data.

## 5. Scalar and history semantics

The driver must support both a single requested configuration and an ordered
history.

Examples:

```python
KinematicDriver(
    crank_joint,
    position=0.5,
)
```

and:

```python
KinematicDriver(
    crank_joint,
    position=np.linspace(0.0, 2.0 * np.pi, 361),
)
```

The public semantics should match the useful behavior already established by
`solve()` in 0.5.0.

### Position coercion

A valid `position` is:

- one finite scalar; or
- a non-empty finite 1-D sequence.

A two-dimensional array or empty sequence is invalid.

Internally, the implementation may normalize scalar position to a one-element
array while retaining scalar/history metadata when useful for validation and
result construction.

### Differential broadcasting

For an `N`-sample position history:

- scalar `velocity` becomes an `N`-sample constant history;
- scalar `acceleration` becomes an `N`-sample constant history;
- a sequence must have shape `(N,)`.

For scalar position:

- scalar velocity/acceleration are valid;
- one-dimensional differential histories are not accepted merely to create
  implicit multiple position samples.

This preserves a simple invariant: position determines the number of requested
samples.

## 6. Time is not part of the driver

Physical time should remain optional metadata supplied to `solve()`:

```python
solution = solve(
    mechanism,
    driver=driver,
    time=t,
    initial_guess=initial_guess,
)
```

This is intentional.

A driver answers:

> Which kinematic coordinate is prescribed, and what values are prescribed?

The optional `time` argument answers:

> At what physical instants are the requested driver samples associated?

This preserves both major Kimech use cases.

### Geometric sweep

```python
driver = KinematicDriver(
    crank_joint,
    position=np.linspace(0.0, 2.0 * np.pi, 361),
)

solution = solve(
    mechanism,
    driver=driver,
    initial_guess=initial_guess,
)
```

This represents a configuration path

```text
q = q(u)
```

without asserting any physical timing.

### Time-aware kinematics

```python
t = np.linspace(0.0, 2.0, 201)
theta0 = np.deg2rad(30.0)

driver = KinematicDriver(
    crank_joint,
    position=theta0 + 5.0 * t,
    velocity=5.0,
    acceleration=0.0,
)

solution = solve(
    mechanism,
    driver=driver,
    time=t,
    initial_guess=initial_guess,
)
```

This associates the requested states with

```text
u = u(t)
q = q(t)
```

without changing the numerical position solve into time integration.

## 7. Time semantics

If `time` is supplied to `solve()`, it should:

- be numeric;
- contain only finite values;
- contain one value per requested driver position;
- be strictly increasing for a multi-sample history.

The exact scalar convention should be resolved during implementation, but a
single requested position should admit either no time or one finite scalar time
value.

`time` must not:

- determine continuation step size;
- replace the driver coordinate as the local continuation parameter;
- trigger automatic numerical differentiation;
- imply numerical integration;
- be used to repair inconsistent user-supplied position/velocity/acceleration
  histories.

The position continuation remains ordered by requested samples and continues to
use changes in prescribed coordinate:

```text
delta_u = u[k + 1] - u[k]
```

rather than `delta_t`.

This matters for non-monotonic laws such as `u(t) = sin(t)`: physical time can
be monotonic while the driver coordinate reverses direction.

## 8. Differential semantics

Velocity and acceleration keep their existing physical meaning:

```text
u_dot  = du/dt
u_ddot = d2u/dt2
```

They do not require explicit `time` samples to be useful.

For each accepted position state, Kimech solves the differentiated kinematic
constraints using the prescribed differential value.

A driver with:

```python
velocity=5.0
acceleration=0.0
```

asserts those physical derivative values directly. Kimech should not compare
them to finite differences of position and time in the core solver.

Future validation utilities may inspect consistency explicitly, but silent
numerical differentiation is outside the 0.6.0 driver contract.

## 9. Driver immutability and ownership

The driver should be externally immutable after construction, consistent with
the current result-oriented API style.

The driver should retain the selected joint by identity.

Mechanism ownership is validated at solve time rather than requiring the driver
to retain a separate mechanism reference. This permits construction to remain
small while ensuring that:

```python
solve(mechanism_a, driver=driver_from_mechanism_b, ...)
```

fails clearly.

Arrays exposed from the driver should not permit caller mutation of internal
state. Public array properties, if exposed, should return safe copies.

## 10. Public properties

The smallest useful public surface is expected to include:

```python
driver.joint
driver.position
driver.velocity
driver.acceleration
```

Naming may use plural properties for normalized histories if that provides a
cleaner distinction between scalar input syntax and stored arrays. This should
be decided together with the result model rather than independently.

The class should not expose numerical solver operations as part of its public
API.

In particular, 0.6.0 should avoid public methods such as:

```text
driver.jacobian(...)
driver.residual(...)
driver.solve(...)
```

These are formulation internals.

## 11. Private solver contract

The current constraints implementation already separates three driver-specific
operations:

```text
driver_residual
driver_jacobian
driver_acceleration_bias
```

This is the natural starting point for the private driver contract.

For a scalar coordinate `g_D(q)`, the solver needs the equivalent of:

### Position equation

```text
g_D(q) - u
```

### Driver Jacobian row

```text
dg_D / dq
```

### Velocity right-hand side

From

```text
(dg_D/dq) q_dot = u_dot
```

the prescribed velocity contributes directly to the driver row of the
differential right-hand side.

### Acceleration bias

For

```text
(dg_D/dq) q_ddot + b_D(q, q_dot) = u_ddot
```

the solver needs the driver-specific second-order bias `b_D`.

The public class need not expose these calculations. The important architectural
constraint is that driver-specific coordinate knowledge should become localized
rather than repeatedly unpacked as `driver.joint` throughout the solver.

A future implementation may use private methods, private helper functions, a
protocol, or composition. This study deliberately does not freeze that
implementation mechanism.

## 12. Avoiding a cosmetic refactor

0.6.0 should not merely transform code such as

```python
residual(..., input_joint, input_value)
```

into

```python
residual(..., driver.joint, driver_value)
```

everywhere.

That would rename the public API while leaving the internal architecture tied
to one natural joint coordinate.

The desired dependency direction is:

```text
solver
  |
  v
kinematic driver formulation
  |
  v
current joint-coordinate implementation
```

not:

```text
solver -> driver.joint -> special-case R/P logic everywhere
```

This is the main architectural acceptance criterion for 0.6-B.

## 13. Future generalization

The long-term mathematical abstraction may be:

```text
g_D(q) in R^m
```

rather than necessarily scalar.

This matters for possible future concepts such as:

- one component of a point position;
- both planar components of a point position;
- body orientation;
- distance or relative-coordinate prescriptions;
- other user-defined kinematic coordinates.

A possible future public organization is:

```text
KinematicDriver
├── JointDriver
├── PointDriver
└── ...
```

but inheritance is not a 0.6.0 requirement.

Composition may instead become more appropriate:

```python
KinematicDriver(
    coordinate=JointCoordinate(...),
    position=...,
)
```

or an internal Python `Protocol` may be sufficient.

0.6.0 should therefore preserve the concept without prematurely creating an
ABC, protocol hierarchy, `PointDriver`, or generic coordinate object.

## 14. One-DOF scope in 0.6.0

The public solver remains one-DOF.

Exactly one `KinematicDriver` is accepted:

```python
solve(
    mechanism,
    driver=driver,
    ...
)
```

The existing structural closure condition remains equivalent to one additional
driver equation.

Multiple drivers are intentionally deferred.

The design should nevertheless avoid choices that make a future API such as

```python
solve(
    mechanism,
    drivers=[driver_a, driver_b],
    time=t,
    ...
)
```

unnatural.

A common `time` argument at solve level is particularly useful for this
future because it avoids duplicating or reconciling time histories across
drivers.

## 15. Result-model direction

The current 0.5.0 result metadata uses:

```text
input_joint
input_position(s)
input_velocity(ies)
input_acceleration(s)
```

0.6.0 should migrate toward driver terminology.

Candidate public metadata is:

```text
solution.driver
solution.driver_positions
solution.driver_velocities
solution.driver_accelerations
solution.time
```

and per-configuration metadata:

```text
configuration.driver
configuration.driver_position
configuration.driver_velocity
configuration.driver_acceleration
configuration.time
```

The final result API should be designed together with implementation rather than
copying these names mechanically.

Snapshot behavior remains important: a solution must retain enough information
to reproduce its analysis semantics even if the source mechanism is later
extended.

## 16. Sensitivity direction

The current 0.5.0 quantity

```text
dq / du
```

already has driver semantics.

0.6.0 should consider migrating:

```text
input_sensitivity
InputSensitivity
```

toward:

```text
driver_sensitivity
DriverSensitivity
```

with derivatives explicitly interpreted with respect to the current driver
coordinate.

The sensitivity calculation remains an explicit downstream analysis and should
not be computed automatically by every solve.

This rename should be decided as part of the broader 0.6.0 public API cleanup,
not as a prerequisite for constructing `KinematicDriver`.

## 17. Diagnostics and failure terminology

Structured diagnostics and failures should gradually move away from
`input_position` terminology and describe the selected driven formulation.

Likely candidates include:

```text
driver_position
driver_index
```

or another consistently chosen driver-value vocabulary.

The important semantic rule remains unchanged: conditioning and failures
describe the selected driven formulation, not a universal mechanism
singularity classification.

## 18. Representative acceptance case

A four-bar with an input crank beginning at 30 degrees and rotating at constant
5 rad/s from 0 to 2 s is a useful 0.6.0 acceptance case.

```python
t = np.linspace(0.0, 2.0, 201)
theta0 = np.deg2rad(30.0)

driver = KinematicDriver(
    crank_joint,
    position=theta0 + 5.0 * t,
    velocity=5.0,
    acceleration=0.0,
)

solution = solve(
    mechanism,
    driver=driver,
    time=t,
    initial_guess=initial_guess,
)
```

The acceptance checks should include:

1. driver position equals `theta0 + 5*t`;
2. driver velocity equals 5 rad/s;
3. driver acceleration equals 0;
4. returned time equals the requested time history;
5. position, velocity, and acceleration results agree with the equivalent
   0.5.0 formulation evaluated at the same prescribed values;
6. the revolute coordinate remains naturally unwrapped through more than one
   revolution;
7. changing temporal sampling density does not redefine the mathematical
   meaning of the position problem.

## 19. Motion-law direction

A future analytic motion-law layer may support usage such as:

```python
driver = KinematicDriver(
    crank_joint,
    law=ConstantVelocity(
        initial_position=np.deg2rad(30.0),
        velocity=5.0,
    ),
)

solution = solve(
    mechanism,
    driver=driver,
    time=t,
    initial_guess=initial_guess,
)
```

A motion law would provide exact or explicitly defined evaluations of

```text
u(t)
u_dot(t)
u_ddot(t)
```

rather than relying on implicit numerical differentiation.

This is a natural extension but is not required for the initial 0.6-A driver
design.

## 20. Explicitly deferred

The following are not part of 0.6-A:

- multiple simultaneous drivers;
- general multi-DOF solving;
- public `JointDriver` / `PointDriver` subclasses;
- generic user-defined kinematic-coordinate protocols;
- motion-law libraries;
- numerical differentiation of sampled motion;
- automatic consistency checks between time, position, velocity, and
  acceleration;
- pseudo-arclength continuation;
- global branch enumeration;
- new joint families;
- dynamics and actuators;
- automatic initial-guess generation.

## 21. Initial design decision

The recommended 0.6-A direction is:

> Introduce one concrete public `KinematicDriver` representing the prescribed
> natural coordinate of one R/P joint together with its position and optional
> physical velocity/acceleration values. Keep physical time optional and owned
> by `solve()` / the resulting solution history. Refactor the solver to depend
> on driver semantics rather than repeatedly unpacking a joint-specific input.

The release remains one-DOF. The abstraction is intended to make the current
formulation clearer first and make later generalization possible second.

## 22. Questions to resolve during implementation

The following details remain intentionally open for 0.6-B/C:

1. Should the normalized public history properties be singular
   (`position`) or plural (`positions`)?
2. Should `KinematicSolution.driver` retain the original immutable driver or
   an explicit solve snapshot?
3. What exact private boundary best localizes coordinate-specific residual,
   Jacobian, and acceleration-bias behavior?
4. Should the sensitivity rename happen in the same breaking API block or a
   later 0.6.0 block?
5. What exact scalar `time` convention gives the cleanest behavior for a
   one-sample solve?
6. Should `time` be strictly increasing only, or merely ordered/nondecreasing
   when repeated physical instants could have a legitimate use?

These questions do not block the central driver semantics above.
