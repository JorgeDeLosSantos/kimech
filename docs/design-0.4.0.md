# Kimech 0.4.0 — Solver robustness design baseline

> Status: design baseline for the `0.4.0` solver-robustness release.
>
> This document records the architectural and numerical decisions introduced after `0.3.0`. The concrete public interface remains documented in [`api.md`](api.md). The numerical experiments supporting the singularity and branch-continuity decisions are recorded separately in the `0.4.0` study documents.

## 1. Purpose

Kimech `0.4.0` strengthens the position-solve path for one-DOF planar mechanisms without broadening the mechanism model.

The release focuses on three related goals:

- improve continuation between requested input samples;
- recover from locally difficult requested steps when possible;
- make numerical behavior observable through structured diagnostics.

The central design principle is that robustness should improve internally without turning the public API into a collection of solver-tuning controls.

The established high-level interface remains:

```python
solution = solve(
    mechanism,
    input_joint=input_joint,
    input_position=input_positions,
    initial_guess=initial_guess,
    input_velocity=input_velocity,
    input_acceleration=input_acceleration,
)
```

`0.4.0` does not add new public joint types, multiple prescribed inputs, a public driver hierarchy, mechanism-specific solvers, or another nonlinear backend.

## 2. Scope

The `0.4.0` solver-robustness scope contains:

- first-order predictor-corrector continuation;
- warm-start fallback when prediction/correction is not usable;
- bounded adaptive subdivision as an internal recovery mechanism;
- scaled-Jacobian conditioning and rank diagnostics;
- solve-process diagnostics;
- explicit study of driven singularities and toggles;
- explicit study of forward/reverse branch continuity;
- regression coverage for the main numerical conclusions.

The scope deliberately does not include pseudo-arclength continuation, automatic assembly-branch enumeration, globally named assembly modes, automatic branch switching, a universal near-singularity threshold, a public singularity exception, automatic initial-guess generation, public solver-tolerance controls, multiple prescribed inputs, or dynamics.

## 3. Numerical baseline inherited from 0.3.0

`0.4.0` builds on the dimensionless numerical scaling introduced in `0.3.0`.

For generalized coordinates

\[
q = D_q \hat q,
\]

and constraints

\[
\Phi = D_\Phi \hat\Phi,
\]

the scaled Jacobian is

\[
\hat J = D_\Phi^{-1} J D_q.
\]

Position solves operate in scaled coordinates, and accepted roots are verified using the scaled residual rather than raw dimensional equations. A characteristic length is inferred from solve-relevant mechanism geometry; no physical-unit package or user-supplied unit declaration is required.

All `0.4.0` continuation and diagnostic decisions are built on this scaled formulation. Condition numbers and singular values are computed from `\hat J`, not from the raw dimensional Jacobian.

## 4. Position continuation pipeline

For a user-ordered input sequence

\[
u_0,u_1,\ldots,u_N,
\]

Kimech preserves the supplied order. It does not sort, wrap, or reinterpret the input history.

For the first sample, the user-provided `initial_guess` is sent to the nonlinear corrector. For subsequent samples the preferred path is:

```text
previous accepted configuration
        |
        v
first-order tangent predictor
        |
        v
nonlinear corrector at requested target
        |
        +---- success ---> accept requested sample
        |
        +---- failure ---> retry from previous accepted configuration
                               |
                               +---- success ---> accept requested sample
                               |
                               +---- failure ---> adaptive subdivision
```

This ordering is part of the intended `0.4.0` behavior.

## 5. Predictor-corrector continuation

At an accepted configuration satisfying

\[
\Phi(q,u)=0,
\]

the local derivative with respect to the prescribed input coordinate is obtained from

\[
J(q,u)\frac{dq}{du}=e_D,
\]

where `e_D` is zero for geometric constraint rows and one in the prescribed-input row, using the sign convention of the assembled driver equation.

For an input increment `Δu`, the first-order prediction is

\[
q_{k+1}^{(0)} = q_k + \frac{dq}{du}\Delta u.
\]

The predicted state is only an initial estimate. The nonlinear corrector still solves the complete constraint system at the requested target.

If the tangent solve fails, produces invalid values, or cannot provide a distinct usable prediction, Kimech falls back to the previous accepted configuration. The tangent is therefore a continuation aid, not an independently accepted state.

## 6. Nonlinear correction and acceptance

The nonlinear corrector remains private implementation infrastructure. A backend success flag is not sufficient for acceptance.

Kimech independently recomputes the scaled residual at the candidate and accepts the configuration only if the dimensionless infinity norm satisfies the internal residual criterion.

This policy applies regardless of whether the starting estimate came from the original `initial_guess`, tangent prediction, warm start, or adaptive subdivision.

## 7. Warm-start fallback

A failed predicted correction does not immediately terminate a sweep. Kimech retries the requested target from the previous accepted configuration when that state differs from the preferred prediction.

This preserves the robustness of the pre-predictor continuation behavior and separates failure of the predictor from failure of the requested configuration itself.

If tangent prediction was unavailable and already collapsed to the previous accepted state, a duplicate retry is not performed.

## 8. Adaptive subdivision

If both direct attempts at a requested target fail and a previous accepted sample exists, Kimech may recursively bisect the input interval.

For a failed interval `[u_a,u_b]`, the midpoint

\[
u_m = \frac{u_a+u_b}{2}
\]

is solved first. Once accepted, it becomes a new local continuation anchor for the remaining interval. Subdivision depth is bounded internally.

### 8.1 Public semantics

Internal subdivision points are implementation details:

- they are not inserted into `solution.input_positions`;
- they do not appear as additional public configurations;
- they are discarded after helping reach the requested target;
- the returned solution length always matches the requested sample count.

`SolveDiagnostics.subdivision_counts` reports how many accepted hidden intermediate configurations were needed for each requested sample.

### 8.2 Failure semantics

Adaptive subdivision is a recovery mechanism, not a change in the requested problem. If recovery fails, Kimech preserves the original requested-target `KinematicSolveError` rather than replacing it with an implementation-specific midpoint or depth-exhaustion failure.

### 8.3 Fundamental limitation

Subdivision cannot cross a fold at which the selected input coordinate ceases to be a valid local parameter of the motion. Smaller input steps help only while the branch remains locally representable as `q=q(u)`.

Crossing such a fold requires a different continuation formulation, such as pseudo-arclength continuation in augmented `(q,u)` space. That is deliberately outside `0.4.0`.

## 9. Structured solve diagnostics

Solutions returned by `solve()` include `SolveDiagnostics`. Histories have one value per requested sample and are preserved when a `KinematicSolution` is sliced.

`0.4.0` separates two kinds of information.

### 9.1 State-of-problem diagnostics

For the complete scaled driven solve Jacobian `\hat J`:

- `condition_numbers`;
- `min_singular_values`;
- `ranks`.

These characterize the numerical state of the selected driven formulation. One singular-value decomposition provides all three quantities.

Rank uses the usual floating-point SVD tolerance, so it is expected to change only near machine-level deficiency. Condition number and minimum singular value are useful before exact rank loss.

Kimech intentionally does not use determinant magnitude as a singularity metric.

### 9.2 Solve-process diagnostics

The continuation/recovery path is exposed through:

- `subdivision_counts`;
- `strategies`;
- `corrector_attempts`;
- `residual_norms`.

The accepted strategy is one of `initial_guess`, `predictor`, `warm_start`, or `subdivision`.

`corrector_attempts` counts nonlinear position-corrector calls needed while obtaining a requested sample, including failed attempts and hidden subdivision solves.

`residual_norms` stores the final dimensionless scaled residual infinity norm.

These diagnostics describe Kimech behavior rather than a particular SciPy backend. Backend-specific fields such as `nfev` and `njev` are therefore not part of the public result model.

## 10. Singularity semantics

The most important singularity decision in `0.4.0` is semantic.

The Jacobian used by `solve()` contains every geometric constraint row plus the selected prescribed-input row. Its singularity therefore describes the **chosen driven solve formulation**, not automatically a driver-independent property of the physical mechanism.

The same physical configuration may be regular for one selected input and singular for another.

The slider-crank dead center is the canonical example: with crank angle prescribed, the driven system remains regular; with slider displacement prescribed at the same physical configuration, the driven Jacobian loses rank because slider position no longer locally parameterizes the branch.

The four-bar rocker extrema exhibit the same behavior. See [`study-0.4.0-singularity-diagnostics.md`](study-0.4.0-singularity-diagnostics.md).

Consequently `0.4.0` does not expose an ambiguous `config.is_singular` property and does not define a universal condition-number threshold for declaring a configuration near singular.

## 11. Branch continuity and assembly modes

Kimech uses local continuation to remain on the branch selected by the initial guess. `0.4.0` does not assign global branch identifiers or enumerate assembly modes.

The branch-continuity study exercised the Archimedes trammel, Whitworth quick-return, Watt II six-bar, Klann linkage, and Theo Jansen linkage. Complete crank revolutions solved forward and reverse reproduced corresponding configurations to solver-level precision, while maintaining periodic closure.

See [`study-0.4.0-branch-continuity.md`](study-0.4.0-branch-continuity.md).

The conclusion is intentionally narrow: the current validated one-DOF mechanism set does not justify a global assembly-mode abstraction in `0.4.0`. It does not claim branch switching is impossible in all mechanisms.

## 12. Differential kinematics interaction

Velocity and acceleration semantics introduced in `0.2.0` remain unchanged. The complete position history is solved first; only after all requested position configurations have been accepted does Kimech compute optional velocity and acceleration histories.

Therefore requesting differential state does not alter position continuation, hidden subdivision configurations are not differential output samples, and diagnostics describe the requested position history.

## 13. Error model

The public exception hierarchy remains:

```text
KimechError
├── InvalidModelError
└── KinematicSolveError
```

No singularity-specific exception is introduced. A poorly conditioned or rank-deficient state may still be represented through diagnostics if an accepted solution exists. `KinematicSolveError` means the requested result could not be accepted; it is not triggered merely by a large condition number.

## 14. Public API impact

`0.4.0` is primarily an additive robustness release. The public `solve()` signature is unchanged from `0.3.0`; the main public addition is `SolveDiagnostics` attached to `KinematicSolution`.

No public solver configuration object is added. In particular there are no public controls for predictor enable/disable, subdivision enable/disable, subdivision depth, singularity threshold, residual tolerance, or nonlinear backend selection.

These remain implementation decisions that can evolve without changing user models.

## 15. Acceptance evidence

The implementation is supported at three levels.

Focused regression tests cover tangent prediction, warm-start fallback, tangent-failure fallback, adaptive subdivision, preservation of requested-target failure, structured Jacobian diagnostics, process diagnostics, slicing, and scaling invariance.

End-to-end acceptance tests cover full-cycle four-bar forward/reverse consistency and the driver-dependent slider-crank dead-center diagnostic behavior.

Exploratory studies remain reproducible for dimensionless numerical robustness (`0.3.0`), singularities/toggles (`0.4.0`), and branch continuity (`0.4.0`). Studies provide design evidence without turning every exploratory case into a permanent per-commit gate.

## 16. Design conclusion

The `0.4.0` position-solving strategy can be summarized as:

```text
dimensionless formulation
        |
        v
tangent predictor
        |
        v
nonlinear corrector
        |
        +--> warm-start retry
                |
                +--> bounded adaptive subdivision
        |
        v
independent residual validation
        |
        v
structured Jacobian + process diagnostics
```

The design favors local robustness and observability while keeping the public surface small. The release intentionally stops before global continuation machinery.

## 17. Deferred directions

Possible later directions include pseudo-arclength continuation, branch discovery or explicit assembly-mode representation, richer diagnostic summaries/event detection, motion/time-law abstractions, serialization, topology visualization, additional joint families, multiple prescribed inputs, and dynamics.

None is required to complete the `0.4.0` solver-robustness objective.
