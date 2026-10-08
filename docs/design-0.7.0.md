# Kimech 0.7.0 — Multi-DOF implementation design (draft)

Status: implementation underway; API and numerical changes are **not yet released**.

## Scope

Enable planar R/P mechanisms with multiple locally independent prescribed coordinates.
Preserve the simplicity of a one-driver request, maintain body-state-based
`KinematicSolution`, and reuse the existing scaled Jacobian and continuation
machinery.

## Agreed public contract

- `solve(mechanism, drivers=driver_or_sequence, initial_guess=..., time=...)`
  will replace the singular `driver=` keyword in the completed release.
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

## First-block status

The branch `feat/0.7.0-multidof-contract` introduces `_drivers.py` and
adapts `solution.py` to multiple drivers with focused contract tests.
The numerical solver still has the legacy 1-DOF `driver=` signature. The
legacy tests and call sites also require coordinated migration before this
branch is ready to merge. Do not release this partial branch.

## Acceptance scenarios

- Open-chain planar 2R positions, velocities and accelerations against analytic results.
- Closed-loop five-bar branch-preserving solutions.
- Mixed revolute/prismatic inputs.
- Sample-count mismatch and duplicate/foreign driver rejection.
- Position-only solves, optional differential histories and slicing.
- Existing four-bar, slider-crank and complex mechanism regressions.
