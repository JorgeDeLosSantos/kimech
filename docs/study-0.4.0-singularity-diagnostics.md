# Study 0.4.0 — Driven singularities and toggle diagnostics

## Purpose

This study examines how Kimech's scaled-Jacobian diagnostics behave near dead-centers and toggle configurations, and what that behavior implies for continuation and future adaptive stepping.

The central object is the dimensionless Jacobian of the complete driven position problem:

    Jhat = D_phi^{-1} J D_q

which includes all geometric constraint rows and the row associated with the chosen `input_joint`.

The study therefore asks a more precise question than “is the mechanism singular?”:

> Is this configuration singular or poorly conditioned for this particular prescribed coordinate?

That distinction matters because the same physical mechanism and the same physical configuration can be regular under one input coordinate and singular under another.

## Reproducible experiment

Companion script:

    playground/singularity_diagnostics_study.py

It studies two standard mechanisms already used by Kimech: slider-crank and four-bar crank-rocker. For each case it compares two different prescribed coordinates at the same physical configurations.

The diagnostics are:

- scaled-Jacobian condition number `cond(Jhat)`;
- minimum singular value `sigma_min(Jhat)`;
- numerical rank.

No fixed “near singular” threshold is imposed.

## 1. Slider-crank

Geometry:

    crank = 0.08
    rod   = 0.24

Two physical dead-centers occur at crank angles `theta = 0` and `theta = pi`.

### Crank-driven formulation

When the crank angle is prescribed, both dead-centers remain regular solve configurations.

| configuration | driver | cond(Jhat) | sigma_min | rank |
| --- | --- | ---: | ---: | ---: |
| right dead-center | crank | ~4.90 | ~0.408 | 9 |
| left dead-center | crank | ~4.90 | ~0.408 | 9 |

This matches the geometry: crank angle remains a valid local parameter through the dead-center. The slider velocity may vanish there, but the position problem itself is not singular when parameterized by crank angle.

### Slider-driven formulation

At the same physical configurations, prescribing slider displacement produces a fold in the solve manifold.

For the right dead-center, representative values while approaching `theta = 0` are:

| |dtheta| | cond(Jhat) | sigma_min | rank |
| ---: | ---: | ---: | ---: |
| 1e-1 | ~9.95e1 | ~2.01e-2 | 9 |
| 1e-2 | ~9.93e2 | ~2.01e-3 | 9 |
| 1e-3 | ~9.93e3 | ~2.01e-4 | 9 |
| 1e-4 | ~9.93e4 | ~2.01e-5 | 9 |
| 1e-6 | ~9.93e6 | ~2.01e-7 | 9 |
| exact | ~1e17 | ~1e-17 | 8 |

The left dead-center shows the same qualitative behavior.

Two observations are especially important:

1. numerical rank remains full through a very ill-conditioned neighborhood;
2. `cond(Jhat)` and `sigma_min` provide a useful continuous warning long before the exact rank test changes.

Near the fold, the observed trend is approximately:

    cond(Jhat) ~ 1 / |dtheta|
    sigma_min  ~ |dtheta|

## 2. Four-bar crank-rocker

Geometry:

    crank   = 0.08
    coupler = 0.22
    rocker  = 0.18
    ground  = 0.30

This mechanism supports a complete crank revolution and has two rocker limit positions. At each limit, the rocker coordinate reaches a local extremum.

### Crank-driven formulation

At both rocker toggles, prescribing crank angle remains regular.

| configuration | driver | cond(Jhat) | sigma_min | rank |
| --- | --- | ---: | ---: | ---: |
| folded toggle | crank | ~10.93 | ~0.184 | 9 |
| extended toggle | crank | ~6.88 | ~0.291 | 9 |

### Rocker-driven formulation

Prescribing rocker angle at the same physical configurations changes the picture completely.

Approaching the folded toggle:

| |dtheta| | cond(Jhat) | sigma_min | rank |
| ---: | ---: | ---: | ---: |
| 1e-1 | ~2.07e2 | ~9.71e-3 | 9 |
| 1e-2 | ~2.03e3 | ~9.91e-4 | 9 |
| 1e-3 | ~2.02e4 | ~9.93e-5 | 9 |
| 1e-4 | ~2.02e5 | ~9.94e-6 | 9 |
| 1e-6 | ~2.02e7 | ~9.94e-8 | 9 |
| exact | ~1e17 | ~1e-17 | 8 |

Approaching the extended toggle gives the same trend, with condition number increasing roughly one decade whenever the crank-angle distance to the toggle decreases one decade.

Again, the physical toggle is regular under crank input and singular under rocker input.

## 3. Interpretation

### Rank loss is a late indicator

An SVD rank test is useful for identifying an exactly singular or floating-point-indistinguishable matrix, but it does not characterize the approach to singularity very well.

A system can remain numerically full rank while `cond(Jhat)` grows through 1e3, 1e5, 1e7, etc. For continuation, condition number and minimum singular value are therefore more informative than rank alone.

### There is no driver-independent scalar singularity flag yet

The experiments demonstrate:

    same mechanism
    same physical configuration
    different input_joint
    => different solve singularity

Therefore a future API should avoid ambiguous properties such as `config.is_singular` unless their semantics are very carefully defined.

The current meaning of `solution.diagnostics` is preferable: diagnostics of the chosen driven solve problem.

### A universal condition-number threshold is not justified yet

The observed divergence is clear, but a library-wide threshold such as `cond(Jhat) > 1e6 => near singular` would currently be a policy decision without enough evidence.

For 0.4.0 it is safer to expose metrics and use them internally as evidence rather than define a binary public classification.

## 4. Implications for predictor-corrector continuation

Kimech's first-order predictor uses:

    q_next_guess = q + (dq/du) * du

with the tangent obtained from the driven Jacobian.

Near a fold in the chosen input coordinate, the driven Jacobian becomes ill conditioned and the tangent magnitude can grow rapidly.

This explains two expected behaviors:

1. the predictor can become increasingly aggressive or inaccurate near the fold;
2. the tangent solve itself can fail at the exact singularity.

The current fallback to the previous accepted configuration is therefore an important robustness property.

## 5. What adaptive stepping can solve

Adaptive subdivision is useful when the requested target configuration exists but the requested input increment is too large for the local predictor/corrector.

Conceptually:

    u_k -----------> u_{k+1}   # fails
    u_k -> u_a -> u_b -> u_{k+1}  # succeeds

A sensible future implementation could:

1. attempt the requested step;
2. retry from warm start if the predictor fails;
3. if both fail, subdivide the input interval;
4. stop at a bounded subdivision depth / minimum step;
5. return only the user-requested configurations.

## 6. What adaptive stepping cannot solve

Adaptive stepping cannot cross a fold when the prescribed input coordinate ceases to be a valid parameter.

The slider-driven dead-center gives the clearest example. At maximum slider extension there is no physical configuration for a larger slider coordinate. No amount of subdivision can turn that target into a solvable configuration.

Likewise, a rocker-driven four-bar cannot continue beyond the rocker-angle extremum simply by choosing smaller rocker-angle increments.

This is not a numerical-step-size problem. It is a parameterization problem.

## 7. Pseudo-arclength continuation

To follow the configuration manifold through a fold, the continuation parameter must be decoupled from the prescribed physical coordinate.

A classical solution is pseudo-arclength continuation, where the solver follows the branch in an augmented `(q, u)` space.

That would allow continuation through points where `dq/du` becomes unbounded. However, pseudo-arclength continuation is a substantially larger architectural feature than adaptive subdivision and changes the semantics of what it means to prescribe `input_position`.

It should therefore remain outside the immediate adaptive-stepping increment unless a concrete Kimech use case requires following a branch through such a fold.

## 8. Recommendations for 0.4.0

1. Keep the current scaled diagnostics.
2. Do not add a public binary near-singularity flag yet.
3. Add adaptive subdivision only as a recovery mechanism for difficult but valid requested steps.
4. Record whether a requested step required predictor fallback or subdivision; these are useful solver diagnostics.
5. Treat rank loss / severe ill conditioning as useful failure context, not automatically as a distinct exception type yet.
6. Preserve the distinction between nonlinear convergence failure, ill conditioning/rank loss, and a target outside the reachable range of the chosen input coordinate.
7. Defer pseudo-arclength continuation until there is a concrete requirement to traverse folds with the same solve call.

## Main conclusion

The important result is semantic rather than a numerical threshold:

> Singularity in Kimech 0.4 should initially mean singularity of the chosen driven solve formulation, not an intrinsic label attached to a physical configuration.

The scaled Jacobian metrics introduced in 0.4.0 are appropriate for expressing that distinction and provide enough information to design adaptive continuation without prematurely introducing a binary singularity policy.