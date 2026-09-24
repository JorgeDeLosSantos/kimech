# Kimech 0.5.0 — input-coordinate sensitivity study

## 1. Purpose

This study defines the public semantics for sensitivity of an accepted one-DOF
kinematic solution with respect to its prescribed natural joint coordinate.

For the current solve formulation,

```text
Phi(q, u) = 0
```

the local configuration tangent satisfies

```text
J(q, u) dq/du = e_driver
```

where `u` is the natural coordinate of the selected input joint.

## 2. Existing internal capability

Kimech 0.4.0 already solves `dq/du` internally to construct the first-order
continuation predictor.

That tangent is currently ephemeral:

- it is computed only when a next requested sample exists;
- predictor failure falls back to a warm start;
- the tangent is not retained in `KinematicSolution`;
- the last requested sample does not need a predictor tangent.

Therefore the existing predictor path should not simply be re-labeled as a
public sensitivity API.

## 3. Design alternatives

### A. Compute sensitivity for every solve

Advantages:

- sensitivity is always available on the result.

Disadvantages:

- every solve gains an additional linear-analysis requirement;
- a valid position solve could now fail because sensitivity is undefined at a
  driven singular state;
- users pay the computational cost even when sensitivity is not needed.

Rejected.

### B. Add a compute-sensitivity flag to solve()

Advantages:

- explicit opt-in;
- sensitivity can be stored with the solution.

Disadvantages:

- expands the core solve API for an analysis concern;
- couples a downstream analysis product to solve orchestration;
- introduces another capability flag alongside velocity/acceleration even
  though sensitivity is not a time-derivative level.

Not preferred for 0.5.0.

### C. Explicit downstream analysis of KinematicSolution

Example:

```python
from kimech import input_sensitivity

solution = solve(...)
sensitivity = input_sensitivity(solution)
```

Advantages:

- leaves position/velocity/acceleration solve semantics unchanged;
- incurs cost only when requested;
- sensitivity failure does not redefine whether the original position solution
  was accepted;
- matches Kimech's existing conceptual separation between Solver and Analysis;
- produces a reusable analysis result rather than anonymous arrays.

Recommended.

## 4. Public result

Introduce an immutable-from-the-caller's-perspective `InputSensitivity`
result.

The minimal surface is:

```python
sensitivity.coordinate_derivatives
sensitivity.body_pose_derivatives(body)
sensitivity.point_position_derivatives(point)
sensitivity.joint_coordinate_derivatives(joint)
```

All derivatives are with respect to the prescribed input coordinate `u`.

For a solution with `N` samples and `n` mobile bodies:

- `coordinate_derivatives`: shape `(N, 3*n)`;
- body-pose derivatives: shape `(N, 3)`;
- point-position derivatives: shape `(N, 2)`;
- joint-coordinate derivatives: shape `(N,)`.

The input joint itself should have unit coordinate derivative, up to numerical
precision.

## 5. Dimensional semantics

Sensitivity is not a time derivative.

For a revolute input coordinate, `u` is an angle and:

- translational generalized components have dimensions length/radian;
- angular generalized components are angle/radian.

For a prismatic input coordinate, `u` is a displacement and:

- translational generalized components are length/length;
- angular generalized components are angle/length.

Kimech does not attach a physical-unit package. Returned values follow the same
unit convention as the mechanism model and prescribed coordinate.

## 6. Numerical scaling

Sensitivity should reuse the same dimensionless scaling formulation as the
position and differential solvers.

The production calculation remains:

```text
J_hat q_hat_u = rhs_hat
```

followed by conversion back to the user's generalized-coordinate units.

The public values are therefore physical-coordinate derivatives, not scaled
internal tangents.

## 7. Result snapshots

Sensitivity analysis requires the exact body/joint layout associated with the
solution.

Kimech already retains the body layout in result objects. For this analysis,
`KinematicSolution` should also retain its joint tuple internally so a later
extension of the source `Mechanism` does not silently change the equations
used by sensitivity analysis.

This is an internal snapshot strengthening, not a new public joint-history API.

## 8. Failure semantics

Sensitivity is evaluated only when explicitly requested through
`input_sensitivity(solution)`.

If the driven Jacobian cannot support a reliable tangent at a requested sample,
the analysis raises `KinematicSolveError` with the structured failure context
already used by 0.5-C.

The original `KinematicSolution` remains valid and unchanged.

No condition-number threshold or automatic near-singularity classification is
introduced.

## 9. Future multi-DOF compatibility

The 0.5.0 API intentionally describes derivatives with respect to the one
current input coordinate.

A future multi-DOF API can add matrix/Jacobian-valued sensitivity concepts
without changing the mathematical meaning of the one-input derivative methods.

The current feature should not introduce a permanent Driver abstraction or
pretend that the one-dimensional derivative already represents general
multi-input sensitivity.

## 10. Acceptance criteria

0.5-D should verify:

1. exact one-body revolute and prismatic sensitivities;
2. the selected input joint has derivative one;
3. finite-difference agreement for a nontrivial four-bar away from singularity;
4. point and joint derived sensitivities agree with finite differences;
5. expected behavior under consistent geometry scaling;
6. slicing/mutation of returned arrays cannot mutate analysis state;
7. sensitivity failure leaves the original position solution usable;
8. sensitivity uses the joint snapshot retained by the solution rather than
   later mechanism extensions.

## 11. Design decision

> Add an explicit downstream `input_sensitivity(solution)` analysis returning
> an `InputSensitivity` snapshot. Do not compute sensitivity by default and
> do not add a solve flag in 0.5.0.

This preserves the current solve contract while making the already-established
input tangent a first-class engineering analysis quantity.
