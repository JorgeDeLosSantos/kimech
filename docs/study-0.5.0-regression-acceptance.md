# Kimech 0.5.0 — regression and acceptance expansion

## 1. Purpose

0.5-E strengthens the permanent CI baseline before larger architectural work
such as Driver and multi-DOF support.

The goal is not to move every playground mechanism into per-commit tests. The
goal is to add a small set of mechanisms that exercise structural and numerical
features not already covered strongly by four-bar and slider-crank tests.

## 2. Existing permanent coverage

Before 0.5-E, Kimech already has strong permanent coverage for:

- single revolute and prismatic mechanisms;
- four-bar position, velocity, acceleration, scaling, and full-cycle
  forward/reverse continuity;
- slider-crank with revolute and prismatic driving;
- driver-dependent dead-center diagnostics;
- adaptive subdivision and predictor/warm-start behavior;
- scale transformations from 1e-3 through 1e3;
- topology, topology visualization, structured failures, and input sensitivity.

Therefore 0.5-E should avoid duplicating these cases.

## 3. Candidate complex mechanisms

The 0.4 branch-continuity study exercised:

- Archimedes trammel;
- Whitworth quick-return;
- Watt II six-bar;
- Klann linkage;
- Theo Jansen leg.

All five are useful reproducible studies, but moving all five full-cycle studies
into CI would add cost without proportional coverage.

## 4. Permanent acceptance selection

### Archimedes trammel

Why keep it permanently:

- contains two prismatic joints plus revolute joints;
- has exact analytical slider coordinates;
- has an exact elliptical tracer path;
- tests a mechanism topology that is different from slider-crank while still
  remaining compact.

Permanent assertions:

- full-cycle solve succeeds;
- horizontal and vertical slider coordinates match analytical expressions;
- tracer path matches the analytical ellipse;
- diagnostics remain full rank with accepted residuals.

### Whitworth quick-return

Why keep it permanently:

- combines revolute and prismatic joints in a compound mechanism;
- includes a moving block in a slotted lever;
- includes a second prismatic output;
- has independent analytical slot and ram-position expressions.

Permanent assertions:

- full-cycle solve succeeds;
- moving-slot coordinate matches analytical geometry;
- ram position matches analytical geometry;
- diagnostics remain full rank with accepted residuals.

The quick-return ratio remains useful as a playground/reporting metric, but the
permanent test should prefer direct coordinate agreement over an extremum ratio
that depends on sample density.

### Watt II six-bar

Why keep it permanently:

- is the smallest currently validated compound all-revolute six-bar case;
- has two independent loops;
- contains a ternary body;
- independent circle-intersection geometry is already available;
- it is a stronger continuation case than another four-bar.

Permanent assertions:

- full-cycle point positions match independent analytical construction;
- forward and reverse sweeps reconstruct the same assembly branch;
- diagnostics remain full rank with accepted residuals.

## 5. Mechanisms retained as studies

### Klann

The Klann linkage remains valuable because it exercises a walking linkage and a
nontrivial foot trajectory. Its analytical construction is longer and overlaps
substantially with the all-revolute compound-loop coverage supplied by Watt II.

It remains in the reproducible playground/study suite.

### Theo Jansen

Theo Jansen remains the heaviest validated mechanism and is especially useful
as a release-level or solver-change study. It should not become a routine
per-commit acceptance test unless a future regression demonstrates that the
lighter cases miss an important failure mode.

## 6. Near-toggle and scale cases

No new permanent cases are needed in 0.5-E for:

- slider-crank dead-center driver dependence;
- four-bar driven toggle diagnostics;
- broad linear scaling.

These are already covered directly by existing acceptance and scaling tests.

## 7. Sampling policy

Permanent complex-mechanism tests use a moderate full-cycle grid rather than the
181/360-point exploratory grids.

The purpose is regression detection, not production of publication-quality
trajectory metrics.

A grid of roughly 73 samples per full revolution is sufficient to exercise:

- continuation around a complete cycle;
- periodic branch behavior;
- analytical path agreement;
- diagnostics over a representative trajectory.

## 8. Acceptance decision

0.5-E adds three permanent mechanism families:

1. Archimedes trammel;
2. Whitworth quick-return;
3. Watt II six-bar.

The Watt case also becomes the permanent compound-mechanism forward/reverse
branch-continuity regression.

Klann and Theo Jansen remain reproducible studies.

This gives Kimech a layered acceptance strategy:

- unit tests for local contracts;
- compact canonical mechanisms for routine kinematic acceptance;
- three representative compound mechanisms for permanent regression;
- heavier Klann/Theo-Jansen studies for release-level and solver-change
  validation.
