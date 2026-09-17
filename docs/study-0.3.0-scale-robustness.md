# Kimech 0.3.0 — scale-robustness study

## Purpose

Characterize the sensitivity of the current position solver to geometrically
equivalent models expressed at different linear scales before changing solver
behavior.

This study is diagnostic. It does **not** change tolerances, nonlinear methods,
continuation, residual equations, or the public API.

## Questions

The study is intended to distinguish three hypotheses:

1. **Acceptance sensitivity** — SciPy reports convergence, but Kimech rejects an
   otherwise useful candidate because `_RESIDUAL_TOL` is an absolute tolerance
   in the model's native length units.
2. **Nonlinear-solver sensitivity** — `hybr` itself changes behavior as the
   translational coordinates become much larger or smaller relative to angular
   coordinates.
3. **Continuation sensitivity** — an individual configuration remains solvable,
   but using the previously accepted configuration as the next initial guess
   loses the branch or exits the local convergence basin.

The hypotheses are not mutually exclusive.

## Scale sweep

For each mechanism, every linear geometric quantity and every translational
component of the initial guess is multiplied by

```text
1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3
```

Angular quantities are unchanged.

For each nonlinear solve the study records:

- SciPy success flag and termination message;
- Kimech's current acceptance result;
- function and Jacobian evaluation counts when available;
- raw infinity norm of the complete residual;
- linear residual infinity norm;
- linear residual normalized by a characteristic mechanism length;
- angular residual infinity norm;
- condition number of the raw analytical Jacobian;
- translational and angular movement from the supplied initial guess.

When sequential continuation first fails, the same input value is solved once
more from the original initial guess. That retry is diagnostic only and does not
alter the continuation run.

## Cases

### A. `control_revolute`

A single driven revolute link. This is a control case with a nonzero ground
location and nonzero local pivot so that translational coordinates scale while
the prescribed rotation does not.

### B. `baseline_four_bar`

The four-bar geometry already used by the differential acceptance tests. It is
a compact closed-loop mechanism representative of ordinary Kimech usage.

### C. `problem_four_bar`

The four-bar that exposed the scale issue during pre-0.3.0 exploration. Its
original numerical geometry is expressed in millimetres. The equivalent model
obtained by dividing linear quantities by 1000 was observed to complete the
input sweep where the original scale did not.

## Running the study

```bash
python playground/scale_robustness.py
```

To retain every step for later analysis:

```bash
python playground/scale_robustness.py --csv scale-robustness.csv
```

The console table is intentionally compact. The CSV is the authoritative raw
study output.

## Interpretation

Useful patterns include:

- `scipy=yes`, `status=FAIL`: evidence that current Kimech acceptance rather
  than nonlinear convergence rejected the candidate;
- strongly scale-dependent `nfev` or SciPy failures: evidence for variable or
  residual scaling inside the nonlinear problem;
- failed continuation with `retry=yes`: evidence that continuation strategy is
  contributing independently of the local nonlinear solve;
- roughly invariant `lin/L` but scale-dependent raw `res_inf`: direct evidence
  that physical accuracy and the current absolute acceptance metric do not
  scale together.

## Exit condition for 0.3-A1

0.3-A1 is complete when the study has been run over all three cases and the
results are sufficient to decide whether 0.3-A2 should address:

- residual acceptance/normalization;
- nonlinear variable scaling;
- continuation;
- or a combination of them.

No solver change should be merged as part of this study branch unless the scope
is explicitly revised after reviewing the data.
