# Kimech 0.3.0 — dimensionless position-solver prototype

## Purpose

The first scale-robustness study showed that the current position solver is not
scale invariant.  The same physical mechanism can be accepted or rejected only
because its linear geometry is expressed at a different numerical scale, and
the raw Jacobian conditioning can change by many orders of magnitude.

This second study tests a dimensionless nonlinear formulation without changing
the public `solve()` implementation.

## Prototype formulation

Let the generalized coordinates be

\[
q = [x_1, y_1, \theta_1, x_2, y_2, \theta_2, \ldots]^T.
\]

For a positive characteristic mechanism length `L`, define a diagonal
coordinate scale

\[
D_q = \operatorname{diag}(L,L,1,L,L,1,\ldots)
\]

and solve in dimensionless coordinates

\[
q = D_q \hat q.
\]

Constraint equations are also scaled by equation type.  Linear equations use
`L`; angular equations use `1`.  This defines `D_phi` and

\[
\hat\Phi = D_\Phi^{-1}\Phi.
\]

The analytical Jacobian supplied to SciPy is transformed consistently:

\[
\hat J = \frac{\partial\hat\Phi}{\partial\hat q}
       = D_\Phi^{-1} J D_q.
\]

The prototype applies the existing numerical acceptance value `1e-9` to the
scaled residual norm rather than to a dimensional mixed residual.

## Equation scaling

For the currently supported joints:

- revolute joint: both residual rows are linear and are divided by `L`;
- prismatic joint: the transverse-displacement row is divided by `L`, while the
  relative-orientation row is unchanged;
- revolute driver: angular row is unchanged;
- prismatic driver: linear row is divided by `L`.

## Experiment

`playground/scale_robustness_dimensionless.py` reuses the three mechanism
families and seven geometric scales from the first study.  For every case it
runs both the current/raw formulation and the dimensionless prototype and
reports:

- pass/fail and number of accepted configurations;
- first failed sample;
- maximum dimensionless residual;
- maximum Jacobian condition number;
- maximum nonlinear function evaluations (`nfev`).

The key comparison is not just whether previously rejected mechanisms pass.
For geometrically equivalent mechanisms, the dimensionless formulation should
make residual quality, conditioning, and nonlinear iteration effort much less
sensitive to the numerical length scale.

## Deliberately deferred

The experiment does **not** yet define how production Kimech should infer the
characteristic length `L`.  Each study case supplies `L` explicitly.  This is
intentional: first verify the scaling mathematics independently, then design a
robust topology-based characteristic-length policy.

The study also does not change public API, continuation strategy, singularity
handling, differential kinematics, or solver tolerances in `kimech.solve()`.
