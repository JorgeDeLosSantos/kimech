# Kimech 0.7.0 — Multi-DOF implementation design (draft)

Status: implementation underway; API and numerical changes are **not yet released**.

## Scope

Enable planar R/P mechanisms with multiple locally independent prescribed coordinates.
Preserve the simplicity of a one-driver request, maintain body-state-based
`KinematicSolution`, and reuse the existing scaled Jacobian and continuation
machinery.

## Agreed public contract

- `solve(mechanism, drivers=driver_or_sequence, initial_guess=..., time=...)`
  is now the only supported public solver interface, with no `driver=` alias.
- A single `KinematicDriver` is accepted without wrapping it in a list.
- Samples are aligned **by index**; every driver has the same sample count.
- `time` remains optional, global, and strictly increasing if supplied.
- `KinematicSolution.drivers` and `Configuration.drivers` return tuples.
- Each indexed configuration owns one sampled prescription per driver;
  slicing carries all histories, time, and diagnostics together.
- `joint_coordinates`, point/body accessors, and full generalized state
  remain unchanged.
- No duplicate solved-input histories are stored. Input residuals can be
  computed from prescribed data and the solved state.
- Dynamics and time-function-based driver laws are outside this release;
  the internal solver must not depend permanently on sampling.

## Numerics to follow

Use a combined residual `[Phi_joints(q); h(q)-u]` and stacked Jacobian.
Require a regular square determined system in this release.
Extend velocity and acceleration right-hand sides to vectors.
Generalize predictor/continuation to segments in multidimensional input space,
keeping scaling and adaptive bisection.
Separate ranks of the geometric constraint Jacobian and the full driver-augmented
Jacobian; do not misdiagnose ordinary nonlinear nonconvergence as a singularity.

## Incremental implementation

1. Normalization, ownership, sample validation, result slicing.
2. Position solver, input-scaled residual and Jacobian.
3. Velocity/acceleration.
4. Continuation and branch tracking.
5. Rank/conditioning diagnostics and error semantics.
6. Ground visualization, docs, examples, regression, release.

## Implementation status (in progress)

On branch `feat/0.7.0-multidof-contract`:

- `_drivers.py` normalizes synchronized prescribed histories.
- `solution.py` stores and slices multiple drivers together with generalized states.
- `_constraints.py` assembles driver-position rows and second-order driver biases
  in stable supplied-driver order.
- `_scaling.py` handles mixed revolute and prismatic prescriptions.
- `_differential.py` solves `J q_dot = [0; u_dot]` and
  `J q_ddot = [-joint_bias; u_ddot - driver_bias]`.
- `solver.py` now has one shared sampled-history pipeline for 1-DOF and
  Multi-DOF: position correction, tangent prediction, adaptive subdivision,
  accepted-sample diagnostics, velocity and acceleration. The vector continuation
  predictor solves the analytical linearized system for the prescribed increment.
  The old `_solve_multiple_positions()` duplicate has been removed.
- The public `solve(..., drivers=...)` API is now exclusive. All one-driver
  tests, user examples, and primary documentation have been migrated.
- Automated acceptance covers analytic serial 2R and rotating prismatic-guide
  velocity/acceleration, in addition to closed-loop five-bar kinematics.

**Differential history contract:** differential values are prescribed by the
user, not inferred from differences of position and time. If every driver
specifies velocity, compute the full generalized velocity. If every driver also
specifies acceleration, compute the full generalized acceleration. If no driver
specifies a differential order, do not compute that order. Partially prescribed
velocity or acceleration across drivers is rejected explicitly: Kimech does not
publish incomplete generalized differential states.

This branch must not be released or merged until the full regression suite,
documentation, and remaining 0.7.0 acceptance criteria are verified.

## Acceptance scenarios

- Open-chain planar 2R positions, velocities and accelerations against analytic results.
- Closed-loop five-bar branch-preserving solutions.
- Mixed revolute/prismatic inputs.
- Sample-count mismatch and duplicate/foreign driver rejection.
- Position-only solves, optional differential histories and slicing.
- Existing four-bar, slider-crank and complex mechanism regressions.

## D17 — Local rank and differential uniqueness

Let `n = 3 * len(links)`, `r = 2 * len(joints)`, and let `m`
be the number of prescribed drivers. The present release requires
`r + m = n` (a structurally square system); this count alone is
**not** a numerical proof that the chosen drivers are independent.

At each accepted position sample, evaluate the *dimensionless scaled*
Jacobian `J = [J_c; J_d]`:

- `rank(J_c) < r`: `joint_rank_loss`. Geometric joint equations
  are locally rank-deficient (e.g. at a toggle or due to redundancy).
- `rank(J_c) == r` but `rank(J) < n`: `dependent_drivers`.
  The prescribed driver set does not determine all local motions.
- `rank(J) == n`: `regular` at the numerical SVD rank tolerance.

The structural mobility estimate is `n - r`; the instantaneous
geometric mobility is `n - rank(J_c)`. They may disagree at special
configurations. `SolveDiagnostics.ranks` retains full-Jacobian ranks;
`joint_ranks` and `rank_issues` provide the decomposition, also
preserved by slicing. `summary()` includes minimum joint rank and
counts by issue type. Condition numbers and minimum singular values
remain continuous indicators of *near* singularity; a high finite
condition number is not automatically treated as exact rank loss.

A *feasible position* can be returned even if its Jacobian is singular.
Velocity/acceleration are rejected if the linearized system cannot
uniquely determine the differential state. They are never manufactured
with a pseudoinverse or least-squares solution.

A **failed nonlinear position corrector** is reported as
`failure_kind="nonconvergence"`, not as a proven kinematic singularity:
an unconverged candidate does not establish a feasible singular
configuration. Differential failures distinguish
`joint_rank_loss`, `dependent_drivers`, and other
`linear_failure` cases in the structured failure context. That
context also provides `sample_index` and an ordered tuple of
`driver_positions`; legacy singular-driver context field names
remain temporarily available until the final API cleanup.
