# Core concepts

## Mechanism

A {py:class}`~kimech.Mechanism` owns the ground body, mobile links, points, and
joints. Kimech currently supports planar revolute and prismatic joints.

The model is declarative: geometry and connectivity are defined first, then the
solver operates on the completed mechanism.

## Kinematic Driver

A {py:class}`~kimech.KinematicDriver` prescribes the natural coordinate of one
revolute or prismatic joint.

```python
driver = KinematicDriver(
    joint,
    position=positions,
    velocity=omega,
    acceleration=alpha,
)
```

`position` is required. `velocity` and `acceleration` are optional physical
time derivatives. Scalar differential values are broadcast over a position
history.

The Driver is **not** a motor or actuator model. It does not contain torque,
force, inertia, numerical integration, or solver settings.

## Solve and continuation

{py:func}`~kimech.solve` solves the requested Driver samples in order. Position
sweeps use predictor-corrector continuation, warm-start fallback, and bounded
adaptive subdivision when recovery is required.

The continuation parameter is the Driver coordinate itself.

## KinematicSolution

A {py:class}`~kimech.KinematicSolution` stores the accepted state history and a
snapshot of the Driver used to obtain it.

```python
solution.driver
solution.coordinates
solution.coordinate_velocities
solution.coordinate_accelerations
solution.diagnostics
solution.time
```

Indexing returns a {py:class}`~kimech.Configuration`; slicing returns another
`KinematicSolution`.

## Diagnostics

Solve-generated solutions expose {py:class}`~kimech.SolveDiagnostics`.
Condition numbers, ranks, singular values, residuals, accepted strategies, and
subdivision counts describe the **selected driven formulation**. They are not a
universal classification of mechanism singularity.

## Driver-coordinate sensitivity

Use {py:func}`~kimech.driver_sensitivity` after a successful solve:

```python
from kimech import driver_sensitivity

sensitivity = driver_sensitivity(solution)
dq_du = sensitivity.coordinate_derivatives
```

This computes local derivatives with respect to the prescribed Driver
coordinate, not derivatives with respect to time.
