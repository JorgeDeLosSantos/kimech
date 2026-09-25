# Time-aware kinematics

Kimech separates the **coordinate used to parameterize configurations** from
the optional **physical time associated with requested samples**.

For one Driver coordinate (u), a geometric sweep represents

[
q = q(u).
]

If the prescribed coordinate is known as a function of physical time,

[
u = u(t),
]

then the same samples may also represent

[
q = q(t).
]

## Constant-speed example

```python
import numpy as np

time = np.linspace(0.0, 2.0, 201)
theta0 = np.deg2rad(30.0)
omega = 5.0

driver = KinematicDriver(
    crank_joint,
    position=theta0 + omega * time,
    velocity=omega,
    acceleration=0.0,
)

solution = solve(
    mechanism,
    driver=driver,
    time=time,
    initial_guess=initial_guess,
)
```

The Driver contains the prescribed coordinate and its physical derivatives.
`time` belongs to the solve history.

## What `time` does

When supplied, `time`:

- contains one physical instant per requested Driver sample;
- is retained as `solution.time`;
- is propagated to `solution[i].time`;
- must be finite and strictly increasing for a multi-sample solve.

## What `time` does not do

Supplying `time` does **not**:

- change the continuation parameter from (u) to (t);
- numerically integrate the mechanism;
- numerically differentiate Driver positions;
- infer missing Driver velocity or acceleration;
- verify that user-supplied position, velocity, and acceleration histories are
  mutually consistent.

A timed and untimed solve evaluated at the same Driver samples therefore have
the same kinematic state. Time adds physical sample association only.
