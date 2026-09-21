# Study 0.4.0 — Branch continuity on complex mechanisms

## Purpose

This study evaluates whether Kimech's current continuation strategy preserves the same physical assembly branch when a complete input cycle is solved in opposite directions.

The current solver already combines:

- first-order tangent prediction;
- warm-start fallback;
- bounded adaptive subdivision;
- scaled nonlinear residual acceptance.

The question is whether 0.4.0 also needs an explicit assembly-mode / branch-tracking abstraction.

## Method

The companion script is:

    playground/branch_continuity_study.py

Five mechanisms already used as Kimech acceptance/playground cases are exercised:

1. Archimedes trammel;
2. Whitworth quick-return;
3. Watt II six-bar;
4. Klann linkage;
5. Theo Jansen leg.

For each mechanism:

1. solve 181 samples over one complete input revolution, including both endpoints;
2. use the final forward configuration as the initial guess for the reverse sweep;
3. solve the identical input grid in reverse order;
4. reverse-align the second solution and compare corresponding body poses;
5. measure periodic closure, maximum normalized consecutive step, subdivision use, and Jacobian conditioning.

Translation differences are normalized by Kimech's characteristic length. Angular differences are wrapped to `[-pi, pi]`. A per-pose error is the maximum of normalized translation error and wrapped angular error.

## Results

The study was executed in CI with Python 3.12 after the full test suite passed.

| case | forward/reverse pose error | periodic closure error | max step | fwd subdivisions | rev subdivisions | max cond(Jhat) | min sigma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Archimedes trammel | 3.576e-32 | 3.701e-16 | 3.491e-02 | 0 | 0 | 6.555e+00 | 3.149e-01 |
| Whitworth | 1.377e-11 | 8.426e-16 | 3.491e-02 | 0 | 0 | 1.962e+01 | 1.067e-01 |
| Watt II six-bar | 4.717e-11 | 1.190e-13 | 3.491e-02 | 0 | 0 | 4.578e+01 | 4.611e-02 |
| Klann | 2.395e-11 | 3.995e-13 | 5.648e-02 | 0 | 0 | 5.925e+01 | 3.849e-02 |
| Theo Jansen | 1.291e-10 | 9.669e-12 | 4.668e-02 | 0 | 0 | 7.991e+01 | 2.874e-02 |

The maximum forward and reverse step metrics were equal for every mechanism at the reported precision.

## Interpretation

### 1. Direction reversal does not reveal branch switching

For all five mechanisms, corresponding forward and reverse configurations agree to approximately solver-level numerical precision.

The largest observed forward/reverse discrepancy was about `1.3e-10` in the Theo Jansen linkage. There is no evidence of a direction-dependent jump to another assembly branch in these cases.

### 2. Full-cycle closure remains strong

All mechanisms return to their initial physical configuration after one complete input revolution. Periodic closure errors remain between roughly `1e-16` and `1e-11`.

This is important because a solver could remain locally smooth yet drift onto another branch during a cycle. The closure results provide no evidence of such drift.

### 3. The difficult mechanisms remain regular under their selected crank inputs

The largest scaled-Jacobian condition number observed was about `8.0e1` for the Theo Jansen case. The corresponding minimum singular value remained about `2.9e-2`.

These values are not close to the fold behavior observed in the singularity study, where condition numbers grew by many orders of magnitude and rank was eventually lost.

### 4. Adaptive subdivision was not needed

Every forward and reverse sweep completed with zero internal subdivisions.

This means the current predictor/warm-start continuation is already sufficient for these standard complex mechanisms at a 2-degree input spacing.

Adaptive subdivision remains useful as a recovery mechanism, but it is not masking branch instability in these cases.

## Does Kimech 0.4 need explicit branch tracking?

Based on this evidence, not yet.

An explicit assembly-mode system would require defining global branch identity, branch creation/merging near singularities, and semantics for mechanisms with multiple loop closures. That is a substantial abstraction.

The current solver already exhibits the behavior we would want from a local branch tracker for the tested one-DOF mechanisms:

- continuity from the previous accepted configuration;
- direction-independent reconstruction of the same branch;
- strong periodic closure;
- no unexpected recovery/subdivision events.

Adding a public `assembly_mode` or branch identifier now would therefore solve a problem that has not yet appeared in Kimech's validated mechanism set.

## Recommended 0.4.0 direction

Rather than adding explicit branch enumeration/tracking, the next useful increment is to enrich `SolveDiagnostics` with **process diagnostics** that make continuation behavior observable.

Candidates include:

- whether predictor or warm-start produced the accepted requested sample;
- number of nonlinear corrector attempts;
- whether tangent prediction failed before fallback;
- final scaled residual norm;
- optionally nonlinear solver iteration/evaluation counts if they can be exposed without coupling the public API tightly to SciPy.

These diagnostics would let future branch-instability cases be detected empirically before Kimech commits to a global assembly-mode model.

## Regression strategy

The current evidence does not justify moving all five mechanisms into expensive per-commit branch-continuity tests.

A lighter strategy is preferable:

- keep the full study reproducible in `playground/branch_continuity_study.py`;
- retain existing mechanism-specific analytical/regression tests;
- consider one compact forward/reverse regression case if future solver changes affect continuation behavior;
- rerun this study before 0.4.0 release and after major continuation changes.

## Main conclusion

> For the current validated one-DOF mechanism set, predictor-corrector continuation plus warm-start fallback preserves branch continuity well enough that an explicit assembly-mode abstraction is not justified for 0.4.0.

The next development effort should favor richer solve-process diagnostics over global branch tracking.
