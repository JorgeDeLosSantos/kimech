# Kimech 0.6.0 — release readiness

## Release theme

Kimech 0.6.0 makes prescribed kinematic motion a first-class concept while
preserving the existing one-DOF planar revolute/prismatic solver.

The release establishes three separate concepts:

```text
Mechanism
    geometry and structural constraints

KinematicDriver
    prescribed coordinate, position, velocity, acceleration

solve(..., time=...)
    numerical solve and optional global physical sample association
```

## Public API introduced or finalized

### Driver construction

```python
driver = KinematicDriver(
    joint,
    position=position,
    velocity=velocity,
    acceleration=acceleration,
)
```

For 0.6.0 the target is the natural coordinate of one revolute or prismatic
joint.

### Solve

```python
solution = solve(
    mechanism,
    driver=driver,
    initial_guess=initial_guess,
    time=time,
)
```

`time` is optional. It labels requested samples with physical instants and does
not replace the driver coordinate as the continuation parameter.

### Results

```python
solution.driver
solution.time

configuration = solution[i]
configuration.driver
configuration.time
```

The Driver is retained as immutable result metadata.

### Sensitivity

```python
sensitivity = driver_sensitivity(solution)
```

with public result type `DriverSensitivity`.

### Failure metadata

```python
context.driver_index
context.driver_position
```

## Intentional breaking changes

No compatibility aliases are retained for:

```text
solve(... input_joint=..., input_position=..., ...)
    -> solve(... driver=KinematicDriver(...))

KinematicSolution.input_*
Configuration.input_*
    -> .driver

input_sensitivity / InputSensitivity
    -> driver_sensitivity / DriverSensitivity

SolveFailureContext.input_index / input_position
    -> driver_index / driver_position
```

Kimech remains pre-1.0, so the release favors a coherent API over compatibility
aliases.

## Acceptance and regression criteria

0.6.0 is release-ready when all of the following hold:

- package and runtime metadata report `0.6.0`;
- the complete test suite passes on supported CI Python versions;
- wheel/sdist build and clean-wheel verification pass;
- existing 0.5.0 mechanism acceptance cases remain green;
- Driver scalar/history validation and differential broadcasting remain green;
- Driver-based solve, result snapshots, slicing, diagnostics, and sensitivity
  remain green;
- optional time validation and result propagation remain green;
- the constant-speed four-bar acceptance case confirms identical kinematic
  state with and without explicit `time`;
- README, current API documentation, and CHANGELOG describe the 0.6.0 API;
- no public 0.5.0 compatibility aliases are introduced accidentally.

## Representative temporal acceptance case

The permanent 0.6.0 temporal acceptance case uses:

```text
theta0 = 30 deg
omega  = 5 rad/s
alpha  = 0
0 <= t <= 2 s
```

with

```text
theta(t) = theta0 + omega t
```

The timed and untimed solves must produce the same position, velocity, and
acceleration states when evaluated at the same Driver samples.

## Explicitly deferred

The following are not release requirements for 0.6.0:

- multiple simultaneous Drivers;
- general multi-DOF solving;
- `JointDriver`, `PointDriver`, or public Driver hierarchies;
- generic user-defined kinematic-coordinate protocols;
- `MotionLaw` or motion-law libraries;
- automatic numerical differentiation of sampled motion;
- automatic consistency checking between time and prescribed derivatives;
- pseudo-arclength continuation;
- global branch enumeration;
- automatic branch-independent initial guesses;
- new joint families;
- dynamics, forces, torques, masses, or inertias;
- GUI or web-application architecture.

## Release decision

If the release branch satisfies the acceptance criteria above without introducing
new scope, tag `v0.6.0` from the merged release commit.
