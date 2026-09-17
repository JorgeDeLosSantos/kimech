# Kimech 0.3.0 numerical robustness — A3 characteristic-length inference

## Goal

Infer the characteristic length used by the dimensionless nonlinear position
formulation without introducing unit declarations or depending on plotting-only
geometry.

## Candidate policy

For one solve problem, define the candidate set from:

1. pairwise distances between joint-referenced (structural) points belonging to
   the same body, including ground;
2. norms of structural-point local coordinates on mobile bodies;
3. the absolute prescribed natural-coordinate values when the driven joint is
   prismatic.

The characteristic length is the largest positive finite candidate.  If no
positive linear scale exists, use `1.0` as a numerical fallback.

## Why these terms

Same-body structural spans capture physical mechanism dimensions and are
invariant to translations and rotations of a body's coordinate frame.

Mobile-body structural offsets capture the lever arms multiplying angular
coordinates in the analytical Jacobian.  A translated mobile-body frame can
therefore change the numerical parameterization even when the physical
mechanism is unchanged; allowing the inferred scale to respond to that change
is intentional.

Absolute ground-point coordinates are excluded.  Translating the global frame
must not change numerical scaling.

Nonstructural points are excluded.  Adding a point of interest or visualization
point must not change solver behavior.

A pure prismatic mechanism can contain no geometric span at all while its
prescribed displacement still has physical length.  The full prescribed input
history therefore contributes to the scale when the input joint is prismatic.

## Desired properties

For ordinary geometrically scaled problems:

`L(s M) = s L(M)` for `s > 0`.

The inferred-scale dimensionless solve should retain the A2 behavior:

- all scale-study sweeps complete;
- scaled Jacobian conditioning remains approximately scale invariant;
- nonlinear evaluation counts remain approximately scale invariant;
- the original millimetre-scale regression completes without changing the
  public solver.

Additional invariants:

- translating all ground geometry does not change `L`;
- adding a distant nonstructural point does not change `L`;
- a mobile-frame offset is reflected in `L` so the scaled rotational lever arm
  remains bounded;
- a pure-prismatic solve derives its scale from prescribed displacement when no
  geometric scale exists;
- a truly zero-linear-scale revolute problem falls back to `L = 1.0`.

## Experiment

Run:

```bash
python playground/characteristic_length_study.py
```

A3 remains a study-only prototype.  No production solver behavior is changed
until the policy is validated.
